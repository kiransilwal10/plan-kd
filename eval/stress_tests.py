"""Stress tests for robustness (rotation, scale, JPEG, crops)."""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image, ImageOps, ImageFilter


def rotate(image: Image.Image, angle: int) -> Image.Image:
    return image.rotate(angle, expand=True)


def scale(image: Image.Image, factor: float) -> Image.Image:
    w, h = image.size
    return image.resize((int(w * factor), int(h * factor)))


def jpeg_compress(image: Image.Image, quality: int = 50) -> Image.Image:
    buf = np.asarray(image)
    pil = Image.fromarray(buf)
    pil.save("/tmp/_tmp.jpg", "JPEG", quality=quality)
    return Image.open("/tmp/_tmp.jpg")


def random_crop(image: Image.Image, ratio: float = 0.8) -> Image.Image:
    w, h = image.size
    nw, nh = int(w * ratio), int(h * ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return image.crop((left, top, left + nw, top + nh))


def run_all(image: Image.Image) -> List[Image.Image]:
    return [
        rotate(image, 90),
        scale(image, 0.9),
        jpeg_compress(image, quality=40),
        random_crop(image, 0.85),
        ImageOps.autocontrast(image.filter(ImageFilter.GaussianBlur(radius=1.5))),
    ]
