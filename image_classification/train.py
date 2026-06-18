#!/usr/bin/env python3
"""Train an image classifier with Keras.

Examples
--------
Train on the bundled synthetic dataset using all defaults::

    python train.py

Use your own data and a different backbone::

    python train.py --dataset-dir data/flowers --backbone resnet50 --epochs 30

Start from a YAML file and override a couple of values on top::

    python train.py --config configs/default.yaml --epochs 5 --no-fine-tune

Artifacts (best checkpoint, final model, class names, config snapshot, history
and training-curve plot) are written to ``<output_dir>/<run_name>_<timestamp>``.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from imgcls.callbacks import build_callbacks
from imgcls.config import load_config, save_config, update_config
from imgcls.data import build_datasets
from imgcls.models import AVAILABLE_BACKBONES, build_model, compile_model, enable_fine_tuning
from imgcls.utils import ensure_dir, plot_history, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an image classification model with Keras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="Path to a YAML config file (optional).")

    g_data = p.add_argument_group("data")
    g_data.add_argument("--dataset-dir", help="Folder of class sub-folders to auto-split.")
    g_data.add_argument("--train-dir", help="Pre-split training folder.")
    g_data.add_argument("--val-dir", help="Pre-split validation folder.")
    g_data.add_argument("--test-dir", help="Pre-split test folder.")
    g_data.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"))
    g_data.add_argument("--batch-size", type=int)

    g_model = p.add_argument_group("model")
    g_model.add_argument("--backbone", choices=AVAILABLE_BACKBONES)
    g_model.add_argument("--weights", help='"imagenet" or "none" for random init.')

    g_train = p.add_argument_group("training")
    g_train.add_argument("--epochs", type=int)
    g_train.add_argument("--learning-rate", type=float)
    g_train.add_argument("--no-augment", action="store_true", help="Disable augmentation.")
    g_train.add_argument("--no-fine-tune", action="store_true", help="Skip the fine-tune phase.")

    g_io = p.add_argument_group("output")
    g_io.add_argument("--output-dir")
    g_io.add_argument("--run-name")
    return p.parse_args()


def args_to_overrides(args: argparse.Namespace) -> dict:
    """Translate CLI flags into config overrides (skipping unset values)."""
    overrides = {
        "dataset_dir": args.dataset_dir,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "test_dir": args.test_dir,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
    }
    if args.weights is not None:
        overrides["weights"] = None if args.weights.lower() == "none" else args.weights
    if args.no_augment:
        overrides["augment"] = False
    if args.no_fine_tune:
        overrides["fine_tune"] = False
    return {k: v for k, v in overrides.items() if v is not None}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = update_config(cfg, args_to_overrides(args))
    set_seed(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(os.path.join(cfg.output_dir, f"{cfg.run_name}_{timestamp}"))
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    print(f"[train] Run directory: {run_dir}")

    # --- data -------------------------------------------------------------
    train_ds, val_ds, _test_ds, class_names = build_datasets(cfg)
    num_classes = len(class_names)
    save_json({"class_names": class_names}, os.path.join(run_dir, "class_names.json"))
    print(f"[train] {num_classes} classes: {class_names}")

    # --- model ------------------------------------------------------------
    model = build_model(cfg, num_classes)
    compile_model(model, cfg)
    model.summary()

    # --- phase 1: train the head (backbone frozen for transfer learning) --
    callbacks, checkpoint = build_callbacks(cfg, run_dir)
    print(f"\n[train] Phase 1: training for up to {cfg.epochs} epochs")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks
    )
    merged = {k: list(v) for k, v in history.history.items()}

    # --- phase 2: optional fine-tuning -----------------------------------
    if cfg.fine_tune:
        if enable_fine_tuning(model, cfg):
            compile_model(model, cfg, fine_tune=True)  # recompile after un-freezing
            epochs_done = len(merged.get("loss", []))
            total_epochs = epochs_done + cfg.fine_tune_epochs
            ft_callbacks, _ = build_callbacks(cfg, run_dir, checkpoint=checkpoint)
            print(
                f"\n[train] Phase 2: fine-tuning for up to "
                f"{cfg.fine_tune_epochs} more epochs"
            )
            ft_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=total_epochs,
                initial_epoch=epochs_done,
                callbacks=ft_callbacks,
            )
            for key, values in ft_history.history.items():
                merged.setdefault(key, []).extend(values)
        else:
            print("[train] Backbone has no base model to fine-tune; skipping phase 2.")

    # --- save artifacts ---------------------------------------------------
    final_path = os.path.join(run_dir, "final_model.keras")
    model.save(final_path)
    save_json(merged, os.path.join(run_dir, "history.json"))
    plot_history(merged, os.path.join(run_dir, "training_curves.png"))

    best_val = max(merged.get("val_accuracy", [0.0]), default=0.0)
    print("\n[train] Done.")
    print(f"[train]   best val_accuracy : {best_val:.4f}")
    print(f"[train]   best checkpoint   : {os.path.join(run_dir, 'best_model.keras')}")
    print(f"[train]   final model       : {final_path}")
    print(f"[train]   artifacts in      : {run_dir}")


if __name__ == "__main__":
    main()
