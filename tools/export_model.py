"""Export BioCLIP 2's image tower to ONNX (fp16), prove it matches open_clip, optionally upload to Hugging Face.

Usage:
  uv run tools/export_model.py                      # export + parity check -> cache/vision.fp16.onnx
  uv run tools/export_model.py --upload USER/leaf-and-gill-bioclip2
"""
import argparse
import io
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import requests
import torch
from onnxruntime.transformers import float16
from PIL import Image

MODEL = "hf-hub:imageomics/bioclip-2"
CACHE = Path("cache")
FP32, FP16 = CACHE / "vision.fp32.onnx", CACHE / "vision.fp16.onnx"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

CARD = """---
license: mit
base_model: imageomics/bioclip-2
tags: [onnx, biology, plants, fungi, zero-shot-image-classification]
---
# BioCLIP 2 image encoder (ONNX, fp16) for Leaf & Gill

The image tower of [imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2) (MIT), exported to ONNX
and converted to fp16 (fp32 inputs/outputs) so it runs in a phone browser with onnxruntime-web.

- Input `pixel_values`: float32 [batch, 3, 224, 224], RGB, resized (shortest side 224, bicubic), center-cropped,
  normalized with mean {mean} / std {std}.
- Output `embeds`: float32 [batch, 768], L2-normalized.
- Parity with open_clip on {n} iNaturalist photos: min cosine similarity {cos:.5f}.

Used by [Leaf & Gill](https://github.com/mohamedaminehamdi/leaf-and-gill), an offline plant and mushroom identifier.
All credit for the model to the Imageomics Institute (BioCLIP 2, NeurIPS 2025).
"""


class ImageTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return torch.nn.functional.normalize(self.model.encode_image(pixel_values), dim=-1)


def preprocess(img):
    """Same steps as the browser: shortest side -> 224 (bicubic), center crop, CLIP normalization."""
    img = img.convert("RGB")
    k = 224 / min(img.size)
    img = img.resize((max(224, round(img.width * k)), max(224, round(img.height * k))), Image.BICUBIC)
    left, top = (img.width - 224) // 2, (img.height - 224) // 2
    x = (np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]


def sample_photos(n=10):
    """A few research-grade, openly licensed iNaturalist photos of plants and fungi."""
    r = requests.get("https://api.inaturalist.org/v1/observations", timeout=60, params={
        "quality_grade": "research", "photos": "true", "photo_license": "cc0,cc-by",
        "iconic_taxa": "Plantae,Fungi", "per_page": n, "order_by": "random"})
    urls = [o["photos"][0]["url"].replace("square", "medium") for o in r.json()["results"]]
    return [Image.open(io.BytesIO(requests.get(u, timeout=60).content)) for u in urls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", metavar="REPO_ID", help="Hugging Face repo to upload the fp16 model to")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    tower = ImageTower(model).eval()
    if not FP16.exists():
        torch.onnx.export(tower, torch.randn(1, 3, 224, 224), FP32, input_names=["pixel_values"],
                          output_names=["embeds"], dynamic_axes={"pixel_values": {0: "batch"}, "embeds": {0: "batch"}},
                          opset_version=17, dynamo=False)
        onnx.save(float16.convert_float_to_float16(onnx.load(FP32), keep_io_types=True), FP16)
    print(f"{FP16}: {FP16.stat().st_size / 1e6:.0f} MB")

    photos = sample_photos()
    session = ort.InferenceSession(FP16, providers=["CPUExecutionProvider"])
    cosines = []
    with torch.no_grad():
        for img in photos:
            x = preprocess(img)
            ref = tower(torch.from_numpy(x)).numpy()[0]
            got = session.run(None, {"pixel_values": x})[0][0]
            cosines.append(float(ref @ got / np.linalg.norm(got)))
    print(f"parity on {len(cosines)} photos: min cosine {min(cosines):.5f}")
    assert min(cosines) > 0.99, "fp16 export drifted from open_clip"

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload, exist_ok=True)
        card = CARD.format(mean=MEAN.round(4).tolist(), std=STD.round(4).tolist(), n=len(cosines), cos=min(cosines))
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.upload)
        api.upload_file(path_or_fileobj=str(FP16), path_in_repo="vision.fp16.onnx", repo_id=args.upload)
        print(f"uploaded https://huggingface.co/{args.upload}")


if __name__ == "__main__":
    main()
