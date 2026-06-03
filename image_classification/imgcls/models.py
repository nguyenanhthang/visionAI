"""Model architectures.

Every model is built as a single ``keras.Model`` that takes raw ``[0, 255]``
RGB images and outputs class probabilities, so it is fully self-contained:

    inputs -> [augmentation] -> preprocessing -> backbone -> head -> softmax

Putting augmentation *inside* the model means it is automatically disabled at
inference time, and putting preprocessing inside means the exported ``.keras``
file can be used on raw images without remembering backbone-specific scaling.
"""

from __future__ import annotations

from typing import Optional

import keras
from keras import layers

from .config import Config

# backbone name -> (application constructor, preprocess_input fn)
_BACKBONES = {
    "mobilenetv2": (
        keras.applications.MobileNetV2,
        keras.applications.mobilenet_v2.preprocess_input,
    ),
    "resnet50": (
        keras.applications.ResNet50,
        keras.applications.resnet50.preprocess_input,
    ),
    "efficientnetb0": (
        keras.applications.EfficientNetB0,
        keras.applications.efficientnet.preprocess_input,
    ),
}

AVAILABLE_BACKBONES = ("simple_cnn",) + tuple(_BACKBONES)


def build_augmentation(cfg: Config) -> keras.Sequential:
    """A small stack of random image augmentations (active only in training)."""
    aug = []
    if cfg.random_flip:
        aug.append(layers.RandomFlip("horizontal"))
    if cfg.random_rotation:
        aug.append(layers.RandomRotation(cfg.random_rotation))
    if cfg.random_zoom:
        aug.append(layers.RandomZoom(cfg.random_zoom))
    if cfg.random_contrast:
        aug.append(layers.RandomContrast(cfg.random_contrast))
    return keras.Sequential(aug, name="data_augmentation")


def _simple_cnn_features(x, cfg: Config):
    """A compact VGG-style CNN trained from scratch."""
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.dropout)(x)
    return x


def _transfer_features(x, cfg: Config):
    """Features from a pretrained ImageNet backbone."""
    app_cls, preprocess = _BACKBONES[cfg.backbone]
    x = preprocess(x)
    base = app_cls(
        include_top=False,
        weights=cfg.weights,
        input_shape=(*cfg.img_size, 3),
    )
    base.trainable = cfg.trainable_base
    # ``training=False`` keeps BatchNorm in inference mode while the base is
    # frozen — important so its running statistics are not destroyed.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.dropout)(x)
    return x


def build_model(cfg: Config, num_classes: int) -> keras.Model:
    """Assemble the full classifier for ``num_classes`` classes."""
    inputs = keras.Input(shape=(*cfg.img_size, 3), name="image")
    x = inputs
    if cfg.augment:
        x = build_augmentation(cfg)(x)

    if cfg.backbone == "simple_cnn":
        x = _simple_cnn_features(x, cfg)
    elif cfg.backbone in _BACKBONES:
        x = _transfer_features(x, cfg)
    else:
        raise ValueError(
            f"Unknown backbone {cfg.backbone!r}. "
            f"Choose one of: {', '.join(AVAILABLE_BACKBONES)}"
        )

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"{cfg.backbone}_classifier")


def compile_model(model: keras.Model, cfg: Config, *, fine_tune: bool = False) -> keras.Model:
    """Compile with Adam + sparse categorical cross-entropy."""
    lr = cfg.fine_tune_lr if fine_tune else cfg.learning_rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def find_base_model(model: keras.Model) -> Optional[keras.Model]:
    """Return the nested pretrained backbone, if the model has one.

    The augmentation block is also a nested (``Sequential``) model, but it
    carries no weights — so we look for the nested model that actually has
    parameters, which is the backbone.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.weights:
            return layer
    return None


def enable_fine_tuning(model: keras.Model, cfg: Config) -> bool:
    """Unfreeze the top portion of the backbone for the fine-tuning phase.

    BatchNorm layers are kept frozen throughout (a common best practice), and
    the bottom ``fine_tune_at`` fraction of layers stays frozen. Returns
    ``False`` when there is no backbone to fine-tune (e.g. ``simple_cnn``).
    """
    base = find_base_model(model)
    if base is None:
        return False

    base.trainable = True
    freeze_until = int(len(base.layers) * cfg.fine_tune_at)
    for i, layer in enumerate(base.layers):
        trainable = i >= freeze_until
        if isinstance(layer, layers.BatchNormalization):
            trainable = False  # always keep BN frozen while fine-tuning
        layer.trainable = trainable
    return True
