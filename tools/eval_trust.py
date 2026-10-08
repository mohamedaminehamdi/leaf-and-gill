"""Measure the on-device identifier on fresh iNaturalist photos, then teach TabPFN when to trust it.

Usage: uv run tools/eval_trust.py --plants 2000 --fungi 1000
Steps (each step is cached in cache/eval/, so reruns skip work already done):
  1. fetch research-grade, openly licensed observations made after BioCLIP 2's training data (since 2025-06-01)
  2. embed every photo with BioCLIP 2 (same weights as the ONNX file; parity is checked in export_model.py)
  3. score against the shipped pack exactly like app.js does (top-5 softmax)
  4. TabPFN learns P(top-1 is right) from (top-1 probability, margin, kingdom); compared with a plain threshold
Writes app/data/trust.json (a 3-D grid per kingdom the phone looks up offline), app/data/confused_with.json and metrics.md.
"""
import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

DATA, EVAL = Path("app/data"), Path("cache/eval")
INAT = "https://api.inaturalist.org/v1/observations"
SINCE = "2025-06-01"
STEPS = 21  # grid resolution per axis; the phone interpolates between points
session = requests.Session()
session.headers["User-Agent"] = "leaf-and-gill/0.1 (https://github.com/mohamedaminehamdi/leaf-and-gill)"


# ---------- 1. fresh observations ----------

def inat_page(params, tries=5):
    """One page of observations, at iNaturalist's requested ~1 request/second, retrying dropped connections."""
    for attempt in range(tries):
        time.sleep(1 + 4 * attempt)
        try:
            r = session.get(INAT, timeout=120, params=params)
            r.raise_for_status()
            return r.json()["results"]
        except requests.RequestException as err:
            print(f"  iNaturalist: {err}; retrying")
    raise RuntimeError("iNaturalist API unavailable")


def fetch_observations(iconic, n):
    path = EVAL / f"obs-{iconic}.json"
    if path.exists():
        return json.loads(path.read_text())
    seen, obs, page = set(), [], 1
    while len(obs) < n and page <= 50:
        results = inat_page({
            "iconic_taxa": iconic, "quality_grade": "research", "photos": "true", "d1": SINCE,
            "photo_license": "cc0,cc-by,cc-by-sa", "rank": "species", "per_page": 200, "page": page})
        for o in results:
            taxon = o.get("taxon") or {}
            if o["id"] in seen or taxon.get("rank") != "species" or not o.get("photos"):
                continue
            seen.add(o["id"])
            photo = o["photos"][0]
            obs.append({"id": o["id"], "name": taxon["name"], "kingdom": "Fungi" if iconic == "Fungi" else "Plantae",
                        "url": photo["url"].replace("square", "medium"), "license": photo.get("license_code"),
                        "attribution": photo.get("attribution", "")})
        page += 1
    path.write_text(json.dumps(obs[:n]))
    return obs[:n]


def download(o):
    path = EVAL / "img" / f"{o['id']}.jpg"
    if not path.exists():
        try:
            path.write_bytes(session.get(o["url"], timeout=60).content)
        except requests.RequestException:
            return None
    return path


# ---------- 2-3. embed and score like the phone ----------

def embed(paths, batch=32):
    cache = EVAL / "img_emb.npy"
    if cache.exists():
        return np.load(cache)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2", device=device)
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch]]).to(device)
            out.append(torch.nn.functional.normalize(model.encode_image(imgs), dim=-1).float().cpu().numpy())
            print(f"  embedded {min(start + batch, len(paths))}/{len(paths)}", end="\r")
    emb = np.concatenate(out)
    np.save(cache, emb)
    return emb


def score(img_emb, text_emb, logit_scale):
    """Exactly what app.js computes: top-5 softmax probabilities and their cosine similarities."""
    cos = img_emb @ text_emb.T
    logits = logit_scale * cos
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    top = np.argsort(-probs, axis=1)[:, :5]
    return top, np.take_along_axis(probs, top, axis=1), np.take_along_axis(cos, top, axis=1)


def features(top_probs, top_cos, top_kingdom_is_fungi):
    """Top-1 probability, its lead over the runner-up, how close the photo is to the best name at all, kingdom."""
    p1, p2 = top_probs[:, 0], top_probs[:, 1]
    return np.column_stack([p1, p1 - p2, top_cos[:, 0], top_kingdom_is_fungi]).astype(np.float32)


# ---------- 4. trust meter ----------

def tabpfn_classifier():
    from tabpfn import TabPFNClassifier
    try:  # v2 weights: Apache-2.0 + attribution, no login needed
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    except (ImportError, AttributeError):
        return TabPFNClassifier()


def precision_at(p, y, coverage=0.4):
    k = int(coverage * len(p))
    return float(y[np.argsort(-p)[:k]].mean())


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def confident_report(p, y, threshold):
    shown = p >= threshold
    wrong = shown & (y == 0)
    return {"coverage": float(shown.mean()), "precision": float(y[shown].mean()) if shown.any() else 0.0,
            "wrong_shown_confident": int(wrong.sum()), "wrong_rate_of_all": float(wrong.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", type=int, default=2000)
    ap.add_argument("--fungi", type=int, default=1000)
    args = ap.parse_args()
    (EVAL / "img").mkdir(parents=True, exist_ok=True)

    species = json.loads((DATA / "species.json").read_text())
    meta = json.loads((DATA / "meta.json").read_text())
    text_emb = np.fromfile(DATA / "text_emb.f16", dtype="<f2").reshape(-1, meta["dim"]).astype(np.float32)
    index = {s["name"]: i for i, s in enumerate(species)}

    obs = fetch_observations("Plantae", args.plants) + fetch_observations("Fungi", args.fungi)
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(download, obs))
    keep = [i for i, p in enumerate(paths) if p and p.stat().st_size > 1000]
    obs, paths = [obs[i] for i in keep], [paths[i] for i in keep]
    print(f"{len(obs)} fresh photos ({sum(o['kingdom'] == 'Fungi' for o in obs)} fungi), since {SINCE}")

    img_emb = embed(paths)
    top, top_probs, top_cos = score(img_emb, text_emb, meta["logit_scale"])
    truth = np.array([index.get(o["name"], -1) for o in obs])
    in_pack = truth >= 0
    correct = (top[:, 0] == truth).astype(int)
    top5 = np.array([t in row for t, row in zip(truth, top)])
    genus_ok = np.array([species[row[0]]["genus"] == o["name"].split()[0] for row, o in zip(top, obs)])
    fungi = np.array([species[row[0]]["kingdom"] == "Fungi" for row in top], dtype=np.float32)
    X = features(top_probs, top_cos, fungi)

    rng = np.random.default_rng(0)
    test = rng.random(len(obs)) < 0.4
    clf = tabpfn_classifier()
    clf.fit(X[~test], correct[~test])
    p_tabpfn = clf.predict_proba(X[test])[:, 1]
    p_raw = X[test, 0]  # baseline: trust the model's own top-1 probability

    # Pick the "Confident" cut-off on the training split so that >= 95% of Confident answers are right.
    p_train = clf.predict_proba(X[~test])[:, 1]
    confident_at = next((t for t in np.linspace(0.5, 0.99, 50) if confident_report(p_train, correct[~test], t)["precision"] >= 0.95), 0.95)
    likely_at = 0.5
    raw_at = next((t for t in np.linspace(0.5, 0.999, 100) if confident_report(X[~test, 0], correct[~test], t)["precision"] >= 0.95), 0.999)

    # Precompute TabPFN on a grid so the phone can look it up offline (trilinear interpolation in app.js).
    p_axis = np.linspace(0, 1, STEPS)
    cos_axis = np.linspace(np.percentile(X[:, 2], 0.5), np.percentile(X[:, 2], 99.5), STEPS)
    grid = np.array([[a, b, c, k] for k in (0, 1) for a in p_axis for b in p_axis for c in cos_axis], dtype=np.float32)
    grid_p = clf.predict_proba(grid)[:, 1].reshape(2, STEPS, STEPS, STEPS)
    (DATA / "trust.json").write_text(json.dumps({
        "p": p_axis.round(4).tolist(), "cos": cos_axis.round(4).tolist(),
        "confident": round(float(confident_at), 3), "likely": likely_at,
        "plants": grid_p[0].round(3).tolist(), "fungi": grid_p[1].round(3).tolist()}, separators=(",", ":")))

    # Real mix-ups: wrong top-1 where the right answer was in the pack.
    pairs = {}
    for t, row, ok in zip(truth, top, correct):
        if t >= 0 and not ok:
            pairs.setdefault(int(t), {}).setdefault(int(row[0]), 0)
            pairs[int(t)][int(row[0])] += 1
    confused = {str(t): sorted(c, key=c.get, reverse=True)[:3] for t, c in pairs.items()}
    (DATA / "confused_with.json").write_text(json.dumps(confused, separators=(",", ":")))

    tab = confident_report(p_tabpfn, correct[test], confident_at)
    raw = confident_report(p_raw, correct[test], raw_at)
    lines = [
        "# Leaf & Gill: accuracy and trust meter",
        "",
        f"Fresh iNaturalist research-grade photos observed since {SINCE} (after BioCLIP 2's training data), "
        f"openly licensed. {len(obs)} photos, {int(in_pack.sum())} of species in the pack, {int((~in_pack).sum())} not in it.",
        "",
        "| | Plants | Fungi | All |",
        "|---|---|---|---|",
    ]
    for label, arr, mask in [("Top-1 species (in pack)", correct, in_pack), ("Top-5 species (in pack)", top5, in_pack),
                             ("Top-1 genus (all photos)", genus_ok, np.ones(len(obs), bool))]:
        cells = []
        for kingdom in ("Plantae", "Fungi", None):
            m = mask & (np.array([o["kingdom"] for o in obs]) == kingdom) if kingdom else mask
            cells.append(f"{arr[m].mean():.1%}" if m.any() else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"## Trust meter (held-out {int(test.sum())} photos)",
        "",
        "| | Raw model score | TabPFN trust meter |",
        "|---|---|---|",
        f"| Separates right from wrong (AUROC, higher is better) | {roc_auc_score(correct[test], p_raw):.3f} | {roc_auc_score(correct[test], p_tabpfn):.3f} |",
        f"| Brier score (lower is better) | {brier(p_raw, correct[test]):.3f} | {brier(p_tabpfn, correct[test]):.3f} |",
        f"| Right answers among the 40% most trusted | {precision_at(p_raw, correct[test]):.1%} | {precision_at(p_tabpfn, correct[test]):.1%} |",
        f"| Answers shown as Confident | {raw['coverage']:.1%} | {tab['coverage']:.1%} |",
        f"| Confident answers that were right | {raw['precision']:.1%} | {tab['precision']:.1%} |",
        f"| Wrong answers shown as Confident | {raw['wrong_shown_confident']} | {tab['wrong_shown_confident']} |",
        "",
        f"Overall, {correct.mean():.1%} of all photos (including species outside the pack) got the right species as the top match. "
        f"Features: top-1 probability, its margin over the runner-up, top-1 cosine similarity, kingdom.",
        "",
        f"Both cut-offs were chosen on the training split to keep Confident answers at least 95% right "
        f"(raw score >= {raw_at:.3f}, TabPFN >= {confident_at:.3f}).",
    ]
    Path("metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
