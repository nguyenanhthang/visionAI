"""Image processing pipeline.

Stores the original image and a stack of adjustment parameters. Every call to
``render`` produces a fresh result by applying the current parameters to the
original, so adjustments are non-destructive until the user explicitly bakes a
snapshot.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


FILTERS = (
    "none",
    "grayscale",
    "sepia",
    "invert",
    "blur",
    "sharpen",
    "edge",
    "emboss",
    "cool",
    "warm",
)


@dataclass
class Adjustments:
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    rotation: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    filter_name: str = "none"

    def is_default(self) -> bool:
        return (
            self.brightness == 1.0
            and self.contrast == 1.0
            and self.saturation == 1.0
            and self.sharpness == 1.0
            and self.rotation == 0
            and not self.flip_horizontal
            and not self.flip_vertical
            and self.filter_name == "none"
        )


def _apply_sepia(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.split()[3]
    grey = ImageOps.grayscale(rgba)
    sepia_rgb = ImageOps.colorize(grey, black="#1a0d04", white="#ffe2b3").convert("RGB")
    sr, sg, sb = sepia_rgb.split()
    return Image.merge("RGBA", (sr, sg, sb, alpha))


def _apply_color_temperature(img: Image.Image, warm: bool) -> Image.Image:
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()
    if warm:
        r = r.point(lambda v: min(int(v * 1.10), 255))
        b = b.point(lambda v: int(v * 0.88))
    else:
        b = b.point(lambda v: min(int(v * 1.12), 255))
        r = r.point(lambda v: int(v * 0.90))
    return Image.merge("RGBA", (r, g, b, a))


def _apply_invert(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    inverted = ImageOps.invert(rgb)
    ir, ig, ib = inverted.split()
    return Image.merge("RGBA", (ir, ig, ib, a))


def _apply_filter(img: Image.Image, name: str) -> Image.Image:
    if name == "grayscale":
        grey = ImageOps.grayscale(img)
        return grey.convert("RGBA")
    if name == "sepia":
        return _apply_sepia(img)
    if name == "invert":
        return _apply_invert(img)
    if name == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=4))
    if name == "sharpen":
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    if name == "edge":
        edges = img.convert("RGB").filter(ImageFilter.FIND_EDGES)
        return edges.convert("RGBA")
    if name == "emboss":
        embossed = img.convert("RGB").filter(ImageFilter.EMBOSS)
        return embossed.convert("RGBA")
    if name == "cool":
        return _apply_color_temperature(img, warm=False)
    if name == "warm":
        return _apply_color_temperature(img, warm=True)
    return img


@dataclass
class Snapshot:
    """A point-in-time copy of the working image used for undo/redo."""

    image: Image.Image
    adjustments: Adjustments


class ImageProcessor:
    """High level facade used by the UI."""

    MAX_HISTORY = 24

    def __init__(self) -> None:
        self._original: Optional[Image.Image] = None
        self._source_path: Optional[str] = None
        self.adjustments = Adjustments()

        self._history: List[Snapshot] = []
        self._history_index: int = -1

    # ------------------------------------------------------------------ I/O
    def load(self, path: str) -> Image.Image:
        with Image.open(path) as raw:
            raw.load()
            img = raw.convert("RGBA")
        self._original = img
        self._source_path = path
        self.adjustments = Adjustments()
        self._history = [Snapshot(image=img.copy(), adjustments=Adjustments())]
        self._history_index = 0
        return img

    def save(self, path: str) -> None:
        rendered = self.render()
        if rendered is None:
            raise RuntimeError("No image loaded.")
        if path.lower().endswith((".jpg", ".jpeg", ".bmp")):
            rendered = rendered.convert("RGB")
        rendered.save(path)

    @property
    def has_image(self) -> bool:
        return self._original is not None

    @property
    def source_path(self) -> Optional[str]:
        return self._source_path

    @property
    def size(self) -> Optional[tuple]:
        return self._original.size if self._original else None

    # ---------------------------------------------------------------- Render
    def render(self) -> Optional[Image.Image]:
        if self._original is None:
            return None

        img = self._original.copy()
        adj = self.adjustments

        if adj.filter_name and adj.filter_name != "none":
            img = _apply_filter(img, adj.filter_name)

        if adj.brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(adj.brightness)
        if adj.contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(adj.contrast)
        if adj.saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(adj.saturation)
        if adj.sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(adj.sharpness)

        if adj.flip_horizontal:
            img = ImageOps.mirror(img)
        if adj.flip_vertical:
            img = ImageOps.flip(img)
        if adj.rotation:
            img = img.rotate(-adj.rotation, expand=True, resample=Image.BICUBIC)

        return img

    # ----------------------------------------------------------- Adjustments
    def set_filter(self, name: str) -> None:
        if name not in FILTERS:
            name = "none"
        self.adjustments.filter_name = name

    def set_brightness(self, value: float) -> None:
        self.adjustments.brightness = max(0.0, value)

    def set_contrast(self, value: float) -> None:
        self.adjustments.contrast = max(0.0, value)

    def set_saturation(self, value: float) -> None:
        self.adjustments.saturation = max(0.0, value)

    def set_sharpness(self, value: float) -> None:
        self.adjustments.sharpness = max(0.0, value)

    def set_rotation(self, degrees: int) -> None:
        self.adjustments.rotation = int(degrees) % 360

    def toggle_flip_horizontal(self) -> bool:
        self.adjustments.flip_horizontal = not self.adjustments.flip_horizontal
        return self.adjustments.flip_horizontal

    def toggle_flip_vertical(self) -> bool:
        self.adjustments.flip_vertical = not self.adjustments.flip_vertical
        return self.adjustments.flip_vertical

    def reset_adjustments(self) -> None:
        self.adjustments = Adjustments()

    # ----------------------------------------------------------- History API
    def commit(self) -> None:
        """Bake current adjustments into the working image as a new snapshot."""
        if self._original is None:
            return
        rendered = self.render()
        if rendered is None:
            return
        # Drop forward history beyond current pointer
        self._history = self._history[: self._history_index + 1]
        self._history.append(
            Snapshot(image=rendered.copy(), adjustments=deepcopy(self.adjustments))
        )
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY :]
        self._history_index = len(self._history) - 1
        # The rendered output becomes the new baseline.
        self._original = rendered.copy()
        self.adjustments = Adjustments()

    def can_undo(self) -> bool:
        return self._history_index > 0

    def can_redo(self) -> bool:
        return self._history_index >= 0 and self._history_index < len(self._history) - 1

    def undo(self) -> Optional[Image.Image]:
        if not self.can_undo():
            return None
        self._history_index -= 1
        snap = self._history[self._history_index]
        self._original = snap.image.copy()
        self.adjustments = deepcopy(snap.adjustments)
        return self.render()

    def redo(self) -> Optional[Image.Image]:
        if not self.can_redo():
            return None
        self._history_index += 1
        snap = self._history[self._history_index]
        self._original = snap.image.copy()
        self.adjustments = deepcopy(snap.adjustments)
        return self.render()

    def revert_to_load(self) -> Optional[Image.Image]:
        if not self._history:
            return None
        self._history_index = 0
        snap = self._history[0]
        self._original = snap.image.copy()
        self.adjustments = Adjustments()
        return self.render()
