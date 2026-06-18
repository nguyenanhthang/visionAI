"""Configuration handling.

A single flat :class:`Config` dataclass holds every knob the project exposes.
Keeping it flat (rather than nested) makes command-line overrides trivial: a
CLI flag maps one-to-one onto a field name.

Typical usage::

    cfg = load_config("configs/default.yaml")     # start from a YAML file
    cfg = update_config(cfg, {"epochs": 5})       # override individual fields
    save_config(cfg, "outputs/run/config.yaml")   # snapshot the exact config
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    """All training/evaluation settings in one place."""

    # ------------------------------------------------------------------ data
    # Point ``dataset_dir`` at a folder of class sub-folders; it will be split
    # into train/val using ``validation_split``. Alternatively provide already
    # split folders via ``train_dir`` / ``val_dir`` / ``test_dir`` — when
    # ``train_dir`` is set it takes priority over ``dataset_dir``.
    dataset_dir: Optional[str] = "data/sample"
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None

    image_size: List[int] = field(default_factory=lambda: [180, 180])
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = 1337

    # ----------------------------------------------------------------- model
    # backbone: simple_cnn | mobilenetv2 | resnet50 | efficientnetb0
    backbone: str = "mobilenetv2"
    dropout: float = 0.2
    weights: Optional[str] = "imagenet"   # "imagenet" or None (random init)
    trainable_base: bool = False          # freeze the backbone for phase 1

    # ---------------------------------------------------------- augmentation
    augment: bool = True
    random_flip: bool = True
    random_rotation: float = 0.10
    random_zoom: float = 0.10
    random_contrast: float = 0.10

    # -------------------------------------------------------------- training
    epochs: int = 20
    learning_rate: float = 1.0e-3
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    # Optional second phase: unfreeze part of the backbone and train slowly.
    fine_tune: bool = True
    fine_tune_epochs: int = 10
    fine_tune_at: float = 0.7    # fraction of backbone layers to keep frozen
    fine_tune_lr: float = 1.0e-5

    # -------------------------------------------------------------------- io
    output_dir: str = "outputs"
    run_name: str = "run"

    # ----------------------------------------------------------- convenience
    @property
    def img_size(self) -> tuple:
        """``image_size`` as a ``(height, width)`` tuple of ints."""
        return (int(self.image_size[0]), int(self.image_size[1]))

    def to_dict(self) -> dict:
        return asdict(self)


_FIELD_NAMES = {f.name for f in fields(Config)}


def load_config(path: Optional[str] = None) -> Config:
    """Build a :class:`Config`, optionally overlaying values from a YAML file."""
    cfg = Config()
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = update_config(cfg, data)
    return cfg


def update_config(cfg: Config, overrides: dict) -> Config:
    """Set fields on ``cfg`` from a mapping, ignoring ``None`` values.

    Raises ``KeyError`` for unknown keys so typos in a YAML file or CLI flag
    surface immediately instead of being silently ignored.
    """
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in _FIELD_NAMES:
            raise KeyError(f"Unknown config key: {key!r}")
        setattr(cfg, key, value)
    return cfg


def save_config(cfg: Config, path: str) -> None:
    """Write ``cfg`` to ``path`` as YAML, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
