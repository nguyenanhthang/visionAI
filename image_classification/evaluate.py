#!/usr/bin/env python3
"""Evaluate a trained model and report metrics.

Loads the model and the class names from a run directory produced by
``train.py``, runs it over the validation (or test) split defined in the run's
config, and writes a classification report plus a confusion-matrix plot.

Example
-------
    python evaluate.py --run-dir outputs/run_20240101-120000 --split val
"""

from __future__ import annotations

import argparse
import os

import keras
import numpy as np

from imgcls.config import load_config
from imgcls.data import build_datasets
from imgcls.utils import ensure_dir, load_json, plot_confusion_matrix, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    p.add_argument("--run-dir", required=True, help="Run directory from train.py.")
    p.add_argument("--model", help="Model path (default: <run-dir>/best_model.keras).")
    p.add_argument("--config", help="Config path (default: <run-dir>/config.yaml).")
    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate (falls back to val if no test set).",
    )
    return p.parse_args()


def collect_predictions(model, dataset):
    """Run the model over a dataset and return ``(y_true, y_pred, y_prob)``."""
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def main() -> None:
    args = parse_args()
    model_path = args.model or os.path.join(args.run_dir, "best_model.keras")
    config_path = args.config or os.path.join(args.run_dir, "config.yaml")
    class_names = load_json(os.path.join(args.run_dir, "class_names.json"))["class_names"]

    cfg = load_config(config_path)
    print(f"[eval] Loading model: {model_path}")
    model = keras.models.load_model(model_path)

    _train_ds, val_ds, test_ds, _ = build_datasets(cfg)
    eval_ds = test_ds if (args.split == "test" and test_ds is not None) else val_ds
    if eval_ds is None:
        raise SystemExit("No evaluation dataset available (configure val_dir/test_dir).")

    loss, acc = model.evaluate(eval_ds, verbose=1)
    print(f"[eval] loss={loss:.4f}  accuracy={acc:.4f}")

    # Detailed metrics via scikit-learn.
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred, _ = collect_predictions(model, eval_ds)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )
    print("\n" + report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out_dir = ensure_dir(os.path.join(args.run_dir, "eval"))
    plot_confusion_matrix(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))

    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    save_json(
        {"loss": float(loss), "accuracy": float(acc), "report": report_dict},
        os.path.join(out_dir, "metrics.json"),
    )
    print(f"[eval] Wrote metrics + confusion matrix to {out_dir}")


if __name__ == "__main__":
    main()
