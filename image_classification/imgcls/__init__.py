"""imgcls — a small, reusable Keras image-classification toolkit.

The package is split into focused modules so the training / evaluation /
prediction entry-points at the project root stay thin:

    config     dataclass-based configuration (YAML + CLI overrides)
    data       tf.data pipelines built from folders of images
    models     a simple CNN plus transfer-learning backbones
    callbacks  Keras callbacks wired up for a training run
    utils      seeding, IO helpers and plotting
"""

__version__ = "0.1.0"

__all__ = ["config", "data", "models", "callbacks", "utils"]
