"""Build app/data/about.json: a short, attributed Wikipedia summary for each species.

Usage: uv run tools/build_about.py
Text is Wikipedia's own (CC BY-SA 4.0), never generated. Any sentence about eating, edibility, cooking or food
is dropped: a field guide on a phone must never suggest that something is safe to eat.
Cached in cache/wiki/ so reruns are cheap.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

DATA, CACHE = Path("app/data"), Path("cache/wiki")
API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
FOOD = re.compile(r"\b(edib\w*|eat\w*|eaten|culinar\w*|cook\w*|cuisine|food\w*|delica\w*|tast\w*|flavou?r\w*|recipe\w*|forag\w*)\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
session = requests.Session()
session.headers["User-Agent"] = "leaf-and-gill/0.1 (https://github.com/mohamedaminehamdi/leaf-and-gill)"


def summary(name):
    path = CACHE / f"{name.replace(' ', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = session.get(API + quote(name.replace(" ", "_")), timeout=60)
    data = r.json() if r.ok else {}
    path.write_text(json.dumps(data))
    return data


def safe_text(extract, limit=320):
    """The first sentences of the summary, without anything about eating, at most ~limit characters."""
    kept = []
    for sentence in SENTENCE.split(extract or ""):
        if FOOD.search(sentence):
            continue
        if kept and len(" ".join(kept + [sentence])) > limit:
            break
        kept.append(sentence.strip())
    return " ".join(kept)


def about(s):
    data = summary(s["name"])
    if data.get("type") != "standard":
        return None
    text = safe_text(data.get("extract", ""))
    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return {"text": text, "url": url} if text and url else None


def main():
    species = json.loads((DATA / "species.json").read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        entries = list(pool.map(about, species))
    out = {str(i): e for i, e in enumerate(entries) if e}
    (DATA / "about.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"about.json: {len(out)}/{len(species)} species have a Wikipedia summary")


if __name__ == "__main__":
    main()
