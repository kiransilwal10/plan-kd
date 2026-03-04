"""Parse raster floor-plan data into unified sample objects."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Any

import numpy as np
from PIL import Image


def raster_to_sample(path: pathlib.Path, image_size: int = 512) -> Dict[str, Any]:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    plan_image = np.asarray(image).transpose(2, 0, 1) / 255.0
    sample = {
        "plan_image": plan_image.astype(np.float32),
        "cad_tokens": [],
        "room_polys": [],
        "door_polys": [],
        "ocr_text": [],
        "meta": {"source": str(path), "rotation": 0},
    }
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raster floor-plan images.")
    parser.add_argument("--data_root", type=pathlib.Path, default=pathlib.Path("data/waffle"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/cache/raster_index.json"))
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for img_path in sorted(args.data_root.glob(ext)):
            try:
                sample = raster_to_sample(img_path, args.image_size)
                samples.append(sample)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {img_path}: {exc}")

    args.out.write_text(json.dumps({"count": len(samples), "samples": samples}, indent=2, default=str))
    print(f"Parsed {len(samples)} raster plans -> {args.out}")


if __name__ == "__main__":
    main()
