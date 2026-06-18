"""Small helpers: seeding, JSON IO and plotting.

Matplotlib uses the non-interactive ``Agg`` backend so plots can be written to
disk on headless machines without a display.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Keras/TensorFlow RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # keras is optional at import time (e.g. data-generation tooling)
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - defensive
        pass


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def plot_history(history: Dict[str, List[float]], out_path: str) -> None:
    """Plot loss & accuracy curves from a Keras history dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.get("loss", [])) + 1)

    axes[0].plot(epochs, history.get("loss", []), label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.get("accuracy", []), label="train")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], out_path: str, normalize: bool = True
) -> None:
    """Render a confusion matrix heatmap to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.5
    fmt = ".2f" if normalize else ".0f"
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
