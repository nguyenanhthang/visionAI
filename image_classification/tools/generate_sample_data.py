#!/usr/bin/env python3
"""Generate a small synthetic image dataset for end-to-end testing.

Creates geometric shapes (circle / square / triangle) on noisy coloured
backgrounds, laid out as::

    <out>/circle/circle_0000.png
    <out>/square/square_0000.png
    <out>/triangle/triangle_0000.png

This lets you exercise the full train -> evaluate -> predict pipeline without
downloading any external dataset.

Example
-------
    python tools/generate_sample_data.py --out data/sample --per-class 150
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle"]


def _random_color() -> tuple:
    return tuple(random.randint(0, 255) for _ in range(3))


def make_image(shape: str, size: int) -> Image.Image:
    """Draw a single random instance of ``shape`` on a noisy background."""
    img = Image.new("RGB", (size, size), _random_color())
    draw = ImageDraw.Draw(img)
    color = _random_color()

    margin = size // 6
    x0 = random.randint(margin, size // 2)
    y0 = random.randint(margin, size // 2)
    x1 = random.randint(size // 2, size - margin)
    y1 = random.randint(size // 2, size - margin)

    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    elif shape == "triangle":
        draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=color)
    else:  # pragma: no cover - guarded by CLASSES
        raise ValueError(f"Unknown shape: {shape}")

    # Sprinkle a little pixel noise so the task isn't trivially separable.
    arr = np.asarray(img).astype(np.int16)
    arr += np.random.randint(-20, 21, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic shapes dataset.")
    parser.add_argument("--out", default="data/sample", help="Output directory.")
    parser.add_argument("--per-class", type=int, default=150, help="Images per class.")
    parser.add_argument("--image-size", type=int, default=180, help="Square image side.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    total = 0
    for shape in CLASSES:
        class_dir = os.path.join(args.out, shape)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(args.per_class):
            make_image(shape, args.image_size).save(
                os.path.join(class_dir, f"{shape}_{i:04d}.png")
            )
            total += 1

    print(f"Wrote {total} images across {len(CLASSES)} classes to '{args.out}'.")


if __name__ == "__main__":
    main()
