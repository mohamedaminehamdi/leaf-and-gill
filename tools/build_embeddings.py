"""Encode every species name with BioCLIP 2's text tower; derive model-based look-alikes.

Usage: uv run tools/build_embeddings.py
Reads app/data/species.json, writes:
  app/data/text_emb.f16       N x 768 little-endian float16, L2-normalized, same order as species.json
  app/data/lookalikes.json    for each species, the 3 nearest species of the same kingdom (indices)
  app/data/meta.json          embedding size, species count, the model's logit scale
"""
import json
from pathlib import Path

import numpy as np
import open_clip
import torch

MODEL = "hf-hub:imageomics/bioclip-2"
DATA = Path("app/data")


def label(s):
    """BioCLIP's zero-shot text format: full taxonomy down to the binomial, plus the common name."""
    taxonomy = " ".join([s["kingdom"], s["phylum"], s["class"], s["order"], s["family"], s["name"]])
    return f"a photo of {taxonomy} with common name {s['common']}." if s["common"] else f"a photo of {taxonomy}."


def lookalikes(emb, kingdoms, k=3):
    """Nearest species of the same kingdom by cosine similarity of the text embeddings."""
    out = []
    for start in range(0, len(emb), 1024):
        sims = emb[start:start + 1024] @ emb.T
        for row, i in enumerate(range(start, min(start + 1024, len(emb)))):
            sims[row, i] = -1
            sims[row, kingdoms != kingdoms[i]] = -1
        out += np.argsort(-sims, axis=1)[:, :k].tolist()
    return out


def main():
    species = json.loads((DATA / "species.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL, device=device)
    tokenizer = open_clip.get_tokenizer(MODEL)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(species), 256):
            tokens = tokenizer([label(s) for s in species[start:start + 256]]).to(device)
            chunks.append(torch.nn.functional.normalize(model.encode_text(tokens), dim=-1).float().cpu().numpy())
    emb = np.concatenate(chunks)
    emb.astype("<f2").tofile(DATA / "text_emb.f16")
    kingdoms = np.array([s["kingdom"] for s in species])
    (DATA / "lookalikes.json").write_text(json.dumps(lookalikes(emb, kingdoms), separators=(",", ":")))
    meta = {"dim": int(emb.shape[1]), "count": len(species), "logit_scale": round(float(model.logit_scale.exp()), 4)}
    (DATA / "meta.json").write_text(json.dumps(meta))
    print(f"{len(species)} species -> text_emb.f16 ({emb.shape[1]}-d, {emb.nbytes / 2e6:.1f} MB) + lookalikes.json")


if __name__ == "__main__":
    main()
