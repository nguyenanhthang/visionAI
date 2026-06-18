"""Keras callbacks for a training run.

A single best-model checkpoint is shared across the (optional) two training
phases so it always reflects the global best, while early-stopping / LR
scheduling are recreated per phase so their patience counters start fresh.
"""

from __future__ import annotations

import importlib.util
import os
from typing import List, Optional, Tuple

import keras

from .config import Config


def build_callbacks(
    cfg: Config,
    run_dir: str,
    *,
    monitor: str = "val_accuracy",
    checkpoint: Optional[keras.callbacks.ModelCheckpoint] = None,
) -> Tuple[List[keras.callbacks.Callback], keras.callbacks.ModelCheckpoint]:
    """Return ``(callbacks, checkpoint)`` for one ``model.fit`` phase.

    Pass the returned ``checkpoint`` back in for a subsequent phase so that
    "best so far" tracking persists across both phases.
    """
    if checkpoint is None:
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(run_dir, "best_model.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        )

    callbacks: List[keras.callbacks.Callback] = [
        checkpoint,
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=cfg.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv"), append=True),
    ]
    # TensorBoard logging is optional: it needs the `tensorboard` package, which
    # minimal installs (e.g. tensorflow-cpu) may not pull in. Only enable it when
    # available so training never crashes just for the sake of logging.
    if importlib.util.find_spec("tensorboard") is not None:
        callbacks.append(
            keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "tensorboard"))
        )
    return callbacks, checkpoint
