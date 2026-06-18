#!/usr/bin/env python3
"""Run inference on a single image or a folder of images.

The exported model already contains its own preprocessing, so images are fed in
at their raw ``[0, 255]`` scale — only resizing is needed here.

Examples
--------
    python predict.py --run-dir outputs/run_20240101-120000 --input cat.jpg
    python predict.py --run-dir outputs/run_20240101-120000 --input some_folder --top-k 3
"""

from __future__ import annotations

import argparse
import os

import keras
import numpy as np

from imgcls.utils import load_json

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify images with a trained model.")
    p.add_argument("--run-dir", required=True, help="Run directory from train.py.")
    p.add_argument("--input", required=True, help="Image file or directory of images.")
    p.add_argument("--model", help="Model path (default: <run-dir>/best_model.keras).")
    p.add_argument("--top-k", type=int, default=3, help="How many predictions to show.")
    return p.parse_args()


def list_images(path: str):
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith(_IMAGE_EXTS)
        ]
        if not files:
            raise SystemExit(f"No images found in directory: {path}")
        return files
    return [path]


def load_image(path: str, size):
    """Load and resize an image into a single-element batch of shape (1, H, W, 3)."""
    img = keras.utils.load_img(path, target_size=size)
    arr = keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def main() -> None:
    args = parse_args()
    model_path = args.model or os.path.join(args.run_dir, "best_model.keras")
    class_names = load_json(os.path.join(args.run_dir, "class_names.json"))["class_names"]

    model = keras.models.load_model(model_path)
    # Input spatial size is encoded in the model: (None, H, W, 3).
    _, height, width, _ = model.input_shape
    size = (height, width)
    top_k = min(args.top_k, len(class_names))

    for image_path in list_images(args.input):
        batch = load_image(image_path, size)
        probs = model.predict(batch, verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        print(f"\n{image_path}")
        for rank, idx in enumerate(order, start=1):
            print(f"  {rank}. {class_names[idx]:<20s} {probs[idx] * 100:6.2f}%")


if __name__ == "__main__":
    main()
