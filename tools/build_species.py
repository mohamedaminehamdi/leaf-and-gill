"""Build app/data/species.json: the most-observed plants and fungi on Earth, from GBIF.

Usage: uv run tools/build_species.py --plants 9000 --fungi 3000
Human observations only, so the list reflects what people actually meet outdoors.
Per-species lookups are cached in cache/gbif/ so reruns are cheap.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

GBIF = "https://api.gbif.org/v1"
KINGDOMS = {"plants": 6, "fungi": 5}
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus")
CACHE = Path("cache/gbif")
session = requests.Session()
session.headers["User-Agent"] = "leaf-and-gill/0.1 (https://github.com/mohamedaminehamdi/leaf-and-gill)"


def top_species_keys(kingdom_key, n):
    """Species keys ranked by number of human observations worldwide."""
    keys, offset = [], 0
    while len(keys) < n:
        r = session.get(f"{GBIF}/occurrence/search", timeout=120, params={
            "kingdomKey": kingdom_key, "basisOfRecord": "HUMAN_OBSERVATION", "limit": 0,
            "facet": "speciesKey", "facetLimit": min(1000, n - len(keys)), "facetOffset": offset})
        r.raise_for_status()
        facets = r.json()["facets"]
        counts = facets[0]["counts"] if facets else []
        if not counts:
            break
        keys += [(int(c["name"]), c["count"]) for c in counts]
        offset += len(counts)
    return keys


def cached_get(name, url):
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = session.get(url, timeout=60)
    r.raise_for_status()
    path.write_text(r.text)
    return r.json()


def english_name(key):
    names = cached_get(f"{key}-vernacular", f"{GBIF}/species/{key}/vernacularNames?limit=100")["results"]
    english = [n["vernacularName"].split(",")[0].strip().capitalize() for n in names if n.get("language") == "eng"]
    # The name most sources agree on ("Stinging nettle"), not the first one listed ("California nettle").
    return Counter(english).most_common(1)[0][0] if english else ""


def match_key(name):
    """GBIF species key for a scientific name (used to force dangerous species into the pack)."""
    m = cached_get(f"match-{name.replace(' ', '_')}", f"{GBIF}/species/match?rank=SPECIES&name={name.replace(' ', '%20')}")
    return m.get("usageKey") if m.get("rank") == "SPECIES" else None


def describe(item):
    key, count = item
    s = cached_get(str(key), f"{GBIF}/species/{key}")
    if s.get("rank") != "SPECIES" or not s.get("canonicalName"):
        return None
    return {"key": key, "name": s["canonicalName"], "common": english_name(key),
            **{rank: s.get(rank, "") for rank in RANKS}, "observations": count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=9000)
    ap.add_argument("--fungi", type=int, default=3000)
    ap.add_argument("--out", default="app/data/species.json")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    items = top_species_keys(KINGDOMS["plants"], args.plants) + top_species_keys(KINGDOMS["fungi"], args.fungi)
    # Deadly look-alikes must always be recognisable, however rarely they are photographed.
    have = {key for key, _ in items}
    danger = json.loads(Path("app/data/danger.json").read_text())["must_include"]
    items += [(key, 0) for key in map(match_key, danger) if key and key not in have]
    print(f"{len(items)} species keys; fetching names...")
    with ThreadPoolExecutor(16) as pool:
        species = [s for s in pool.map(describe, items) if s]
    seen, unique = set(), []
    for s in species:  # synonyms can collapse onto the same accepted name
        if s["name"] not in seen:
            seen.add(s["name"])
            unique.append(s)
    Path(args.out).write_text(json.dumps(unique, ensure_ascii=False, separators=(",", ":")))
    fungi = sum(s["kingdom"] == "Fungi" for s in unique)
    named = sum(bool(s["common"]) for s in unique)
    print(f"wrote {args.out}: {len(unique)} species ({len(unique) - fungi} plants, {fungi} fungi), {named} with English names")


if __name__ == "__main__":
    main()
