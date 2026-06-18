"""Input pipelines.

Images are loaded straight from folders with
``keras.utils.image_dataset_from_directory`` (one sub-folder per class), then
cached and prefetched for throughput. Pixel values are left in the ``[0, 255]``
range here — each model applies its own preprocessing as its first layers (see
:mod:`imgcls.models`), which keeps the saved model self-contained.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import keras
import tensorflow as tf

from .config import Config

AUTOTUNE = tf.data.AUTOTUNE


def _load_dir(
    directory: str,
    cfg: Config,
    *,
    subset: Optional[str] = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Wrapper around ``image_dataset_from_directory`` honouring the config."""
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        validation_split=cfg.validation_split if subset else None,
        subset=subset,
        seed=cfg.seed,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
    )


def build_datasets(
    cfg: Config,
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], Optional[tf.data.Dataset], List[str]]:
    """Construct datasets for a run.

    Returns ``(train_ds, val_ds, test_ds, class_names)``. ``val_ds`` / ``test_ds``
    may be ``None`` when no corresponding data is configured.
    """
    if cfg.train_dir:
        # Explicit, pre-split folders.
        train_ds = _load_dir(cfg.train_dir, cfg, shuffle=True)
        val_ds = _load_dir(cfg.val_dir, cfg, shuffle=False) if cfg.val_dir else None
    else:
        if not cfg.dataset_dir:
            raise ValueError("Set either `dataset_dir` or `train_dir` in the config.")
        # Single folder split into train/validation on the fly.
        train_ds = _load_dir(cfg.dataset_dir, cfg, subset="training")
        val_ds = _load_dir(cfg.dataset_dir, cfg, subset="validation", shuffle=False)

    # ``class_names`` is only available before the dataset is transformed.
    class_names = list(train_ds.class_names)

    test_ds = _load_dir(cfg.test_dir, cfg, shuffle=False) if cfg.test_dir else None

    # Throughput: cache decoded images and overlap data loading with training.
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    if val_ds is not None:
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
