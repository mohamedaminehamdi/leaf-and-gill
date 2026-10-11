"""Pre-compute the example photos' results so visitors can try the app before the model has downloaded.

Usage: uv run tools/build_examples.py
Same model, same species pack and same scoring as the phone (see tools/eval_trust.py score()).
Writes app/data/examples.json, tied to the pack version so a stale file is never used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_trust import score  # noqa: E402

DATA, EXAMPLES = Path("app/data"), Path("app/examples")


def main():
    meta = json.loads((DATA / "meta.json").read_text())
    text_emb = np.fromfile(DATA / "text_emb.f16", dtype="<f2").reshape(-1, meta["dim"]).astype(np.float32)
    credits = json.loads((EXAMPLES / "credits.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2", device=device)
    out = {}
    with torch.no_grad():
        for c in credits:
            img = preprocess(Image.open(EXAMPLES / c["file"]).convert("RGB"))[None].to(device)
            emb = torch.nn.functional.normalize(model.encode_image(img), dim=-1).float().cpu().numpy()
            top, probs, cos = score(emb, text_emb, meta["logit_scale"])
            out[f"examples/{c['file']}"] = [{"i": int(i), "p": round(float(p), 5), "cos": round(float(c_), 5)}
                                            for i, p, c_ in zip(top[0], probs[0], cos[0])]
    (DATA / "examples.json").write_text(json.dumps({"pack": meta["pack"], "results": out}, separators=(",", ":")))
    species = json.loads((DATA / "species.json").read_text())
    for name, matches in out.items():
        print(f"{name}: {species[matches[0]['i']]['name']} ({matches[0]['p']:.0%})")


if __name__ == "__main__":
    main()
