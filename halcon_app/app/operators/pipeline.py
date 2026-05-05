"""Pipeline engine: chuỗi tool kéo-thả như VisionPro ToolBlock.

Cấu trúc:
- `Param`: schema cho 1 tham số (cho ParamDialog dựng widget)
- `ToolSpec`: metadata + runner cho 1 tool (id / display / icon / chain / runner / params)
- `TOOLS`: registry tất cả tool khả dụng
- `PipelineContext`: tài nguyên ngoài (template, reference, mask, segment, color_roi)
- `PipelineNode`: 1 node trong chuỗi (params, enabled, last_image, last_metrics, …)
- `Pipeline`: list node + run() chạy tuần tự, mỗi node nhận output của node trước nếu spec.chain=True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .halcon_engine import (
    OperatorResult,
    adaptive_threshold,
    apply_filter,
    apply_mask,
    color_stats,
    contour_analysis,
    decode_codes,
    edges_sub_pix,
    histogram,
    image_diff,
    measure_pairs,
    morphology,
    rotate_image,
    shape_match,
    threshold_blob,
)


# =============================================================================
# Schema
# =============================================================================

@dataclass
class Param:
    name: str
    label: str
    kind: str               # "int" | "float" | "choice" | "bool"
    default: Any
    rng: Optional[tuple] = None
    step: Optional[float] = None
    choices: Optional[list[str]] = None


@dataclass
class ToolSpec:
    id: str
    display: str
    icon: str
    chain: bool                       # True = output ảnh được chuyển làm input node sau
    needs: list[str]                  # ["template", "reference", "segment", "color_roi"]
    params: list[Param]
    runner: Callable[..., OperatorResult]

    def default_params(self) -> dict:
        return {p.name: p.default for p in self.params}


@dataclass
class PipelineContext:
    """Tài nguyên ngoài node có thể cần."""
    template: Optional[np.ndarray] = None
    reference: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    segment: Optional[tuple[int, int, int, int]] = None
    color_roi: Optional[tuple[int, int, int, int]] = None


@dataclass
class PipelineNode:
    tool_id: str
    params: dict
    enabled: bool = True
    label: str = ""
    last_image: Optional[np.ndarray] = field(default=None, repr=False)
    last_metrics: Optional[dict] = None
    last_log: Optional[list[str]] = None
    last_thumbnail: Optional[np.ndarray] = field(default=None, repr=False)
    error: Optional[str] = None


# =============================================================================
# Runners
# =============================================================================

def _r_filter(img, p, ctx): return apply_filter(img, **p)
def _r_morph(img, p, ctx): return morphology(img, **p)
def _r_rotate(img, p, ctx): return rotate_image(img, **p)
def _r_blob(img, p, ctx): return threshold_blob(img, **p)
def _r_adaptive(img, p, ctx): return adaptive_threshold(img, **p)
def _r_edges(img, p, ctx): return edges_sub_pix(img, **p)
def _r_contours(img, p, ctx): return contour_analysis(img, **p)


def _r_match(img, p, ctx):
    if ctx.template is None:
        raise RuntimeError("Pattern Match cần template (Pick ROI hoặc Load file).")
    return shape_match(img, ctx.template, **p)


def _r_caliper(img, p, ctx):
    if ctx.segment is None:
        raise RuntimeError("Caliper cần segment (vẽ trên ảnh).")
    r1, c1, r2, c2 = ctx.segment
    return measure_pairs(img, r1, c1, r2, c2, **p)


def _r_histogram(img, p, ctx): return histogram(img)
def _r_idread(img, p, ctx): return decode_codes(img)


def _r_color(img, p, ctx):
    if ctx.color_roi:
        x, y, w, h = ctx.color_roi
        return color_stats(img, x, y, w, h)
    return color_stats(img)


def _r_diff(img, p, ctx):
    if ctx.reference is None:
        raise RuntimeError("Image Diff cần reference (golden) image.")
    return image_diff(img, ctx.reference, **p)


# =============================================================================
# Tool registry
# =============================================================================

TOOLS: dict[str, ToolSpec] = {
    "filter": ToolSpec(
        id="filter", display="Filter", icon="🪄", chain=True, needs=[],
        params=[
            Param("method", "Method", "choice", "gauss",
                  choices=["gauss", "median", "mean", "sharpen"]),
            Param("ksize", "Kernel", "int", 5, rng=(3, 51), step=2),
            Param("sigma", "Sigma", "float", 1.5, rng=(0.1, 10.0), step=0.1),
        ],
        runner=_r_filter,
    ),
    "morphology": ToolSpec(
        id="morphology", display="Morphology", icon="🧱", chain=True, needs=[],
        params=[
            Param("op", "Operation", "choice", "dilate",
                  choices=["dilate", "erode", "open", "close", "gradient", "tophat", "blackhat"]),
            Param("shape", "Kernel shape", "choice", "rect",
                  choices=["rect", "ellipse", "cross"]),
            Param("ksize", "Kernel size", "int", 5, rng=(1, 51)),
            Param("iterations", "Iterations", "int", 1, rng=(1, 20)),
        ],
        runner=_r_morph,
    ),
    "rotate": ToolSpec(
        id="rotate", display="Rotate", icon="↻", chain=True, needs=[],
        params=[
            Param("angle", "Angle (°)", "float", 0.0, rng=(-360.0, 360.0), step=0.5),
            Param("scale", "Scale", "float", 1.0, rng=(0.1, 10.0), step=0.05),
            Param("interpolation", "Interpolation", "choice", "linear",
                  choices=["nearest", "linear", "cubic", "lanczos"]),
            Param("expand", "Expand canvas", "bool", True),
            Param("border", "Border", "choice", "constant",
                  choices=["constant", "reflect", "replicate"]),
        ],
        runner=_r_rotate,
    ),
    "blob": ToolSpec(
        id="blob", display="Blob", icon="⬛", chain=False, needs=[],
        params=[
            Param("min_gray", "Min gray", "int", 0, rng=(0, 255)),
            Param("max_gray", "Max gray", "int", 128, rng=(0, 255)),
            Param("min_area", "Min area", "int", 100, rng=(1, 10_000_000)),
            Param("max_area", "Max area", "int", 10_000_000, rng=(1, 100_000_000)),
        ],
        runner=_r_blob,
    ),
    "adaptive": ToolSpec(
        id="adaptive", display="Adaptive Threshold", icon="🌓", chain=False, needs=[],
        params=[
            Param("method", "Method", "choice", "mean", choices=["mean", "gaussian"]),
            Param("block_size", "Block size", "int", 15, rng=(3, 99), step=2),
            Param("offset", "Offset", "int", 5, rng=(-50, 50)),
        ],
        runner=_r_adaptive,
    ),
    "edges": ToolSpec(
        id="edges", display="Edges", icon="✶", chain=False, needs=[],
        params=[
            Param("method", "Method", "choice", "canny",
                  choices=["canny", "sobel", "deriche2", "lanser2"]),
            Param("alpha", "Alpha", "float", 1.0, rng=(0.1, 10.0), step=0.1),
            Param("low", "Low", "int", 40, rng=(0, 255)),
            Param("high", "High", "int", 120, rng=(0, 255)),
        ],
        runner=_r_edges,
    ),
    "contours": ToolSpec(
        id="contours", display="Contours", icon="〜", chain=False, needs=[],
        params=[
            Param("min_area", "Min area", "int", 50, rng=(1, 10_000_000)),
            Param("max_area", "Max area", "int", 10_000_000, rng=(1, 100_000_000)),
            Param("approx_eps", "Approx ε", "float", 0.01, rng=(0.001, 0.5), step=0.005),
        ],
        runner=_r_contours,
    ),
    "match": ToolSpec(
        id="match", display="Pattern Match", icon="🎯", chain=False, needs=["template"],
        params=[
            Param("min_score", "Min score", "float", 0.6, rng=(0.1, 1.0), step=0.05),
            Param("num_matches", "Num matches", "int", 5, rng=(1, 100)),
            Param("angle_start", "Angle start", "float", -0.39, rng=(-3.14, 3.14), step=0.05),
            Param("angle_extent", "Angle extent", "float", 0.78, rng=(0.0, 6.28), step=0.05),
        ],
        runner=_r_match,
    ),
    "caliper": ToolSpec(
        id="caliper", display="Caliper", icon="📐", chain=False, needs=["segment"],
        params=[
            Param("sigma", "Sigma", "float", 1.0, rng=(0.1, 10.0), step=0.1),
            Param("threshold", "Edge threshold", "int", 30, rng=(1, 255)),
        ],
        runner=_r_caliper,
    ),
    "histogram": ToolSpec(
        id="histogram", display="Histogram", icon="📊", chain=False, needs=[],
        params=[],
        runner=_r_histogram,
    ),
    "idread": ToolSpec(
        id="idread", display="ID Read", icon="🔢", chain=False, needs=[],
        params=[],
        runner=_r_idread,
    ),
    "color": ToolSpec(
        id="color", display="Color stats", icon="🎨", chain=False, needs=[],
        params=[],
        runner=_r_color,
    ),
    "diff": ToolSpec(
        id="diff", display="Image Diff", icon="📋", chain=False, needs=["reference"],
        params=[
            Param("threshold", "Diff threshold", "int", 30, rng=(1, 255)),
            Param("blur", "Blur kernel", "int", 5, rng=(1, 25), step=2),
        ],
        runner=_r_diff,
    ),
}


# =============================================================================
# Helpers
# =============================================================================

def make_thumbnail(img: np.ndarray, size: int = 64) -> np.ndarray:
    if img is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas = np.full((size, size, 3), 22, dtype=np.uint8)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


# =============================================================================
# Pipeline
# =============================================================================

class Pipeline:
    def __init__(self):
        self.nodes: list[PipelineNode] = []

    def add(self, tool_id: str, params: dict | None = None,
            label: str = "") -> PipelineNode:
        spec = TOOLS[tool_id]
        node = PipelineNode(
            tool_id=tool_id,
            params=params if params is not None else spec.default_params(),
            label=label or spec.display,
        )
        self.nodes.append(node)
        return node

    def remove(self, idx: int) -> None:
        if 0 <= idx < len(self.nodes):
            del self.nodes[idx]

    def move(self, src: int, dst: int) -> None:
        if not (0 <= src < len(self.nodes)):
            return
        node = self.nodes.pop(src)
        dst = max(0, min(dst, len(self.nodes)))
        self.nodes.insert(dst, node)

    def clear(self) -> None:
        self.nodes.clear()

    def run(self, image: np.ndarray, ctx: PipelineContext) -> None:
        if image is None:
            return
        current = image
        if ctx.mask is not None:
            current = apply_mask(current, ctx.mask)
        for node in self.nodes:
            if not node.enabled:
                node.last_image = None
                node.last_metrics = None
                node.last_log = ["[skipped]"]
                node.last_thumbnail = None
                node.error = None
                continue
            spec = TOOLS[node.tool_id]
            try:
                result = spec.runner(current, node.params, ctx)
                node.last_image = result.image
                node.last_metrics = result.metrics
                node.last_log = result.log
                node.last_thumbnail = make_thumbnail(result.image)
                node.error = None
                if spec.chain:
                    current = result.image
            except Exception as exc:
                node.last_image = None
                node.last_metrics = {"error": str(exc)}
                node.last_log = [f"[ERROR] {exc}"]
                node.last_thumbnail = None
                node.error = str(exc)
