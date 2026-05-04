"""HALCON engine wrapper.

Bao bọc các operator HALCON thường dùng. Nếu thư viện `halcon` (mvtec-halcon)
không có sẵn, fallback sang OpenCV/numpy để app vẫn chạy được trong môi
trường dev/CI. Tất cả hàm public trả về numpy.ndarray (BGR/GRAY) để GUI hiển thị
và `dict` chứa kết quả số liệu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

try:
    import halcon as ha  # type: ignore

    HALCON_AVAILABLE = True
    HALCON_VERSION = getattr(ha, "__version__", "unknown")
except Exception:  # ImportError, license error, etc.
    ha = None  # type: ignore
    HALCON_AVAILABLE = False
    HALCON_VERSION = ""


@dataclass
class OperatorResult:
    """Kết quả trả về từ một operator."""

    image: np.ndarray  # ảnh hiển thị (đã vẽ overlay)
    metrics: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_image(path: str) -> np.ndarray:
    """Đọc ảnh: ưu tiên HALCON `read_image`, fallback OpenCV."""
    if HALCON_AVAILABLE:
        try:
            himg = ha.read_image(path)
            return _himage_to_numpy(himg)
        except Exception:
            pass
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _himage_to_numpy(himg) -> np.ndarray:
    """Chuyển HALCON HImage sang numpy (BGR)."""
    channels = ha.count_channels(himg)
    if channels == 1:
        ptr, type_, w, h = ha.get_image_pointer1(himg)
        arr = np.frombuffer(
            (ha.HTuple(ptr).S if False else None) or b"", dtype=np.uint8
        )
        # API thực dùng get_image_pointer1 trả về numpy trực tiếp ở binding mới
        arr = np.array(ha.get_image_pointer1(himg)[0]).reshape(h, w)
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # 3-channel
    r, g, b, type_, w, h = ha.get_image_pointer3(himg)
    rr = np.array(r).reshape(h, w).astype(np.uint8)
    gg = np.array(g).reshape(h, w).astype(np.uint8)
    bb = np.array(b).reshape(h, w).astype(np.uint8)
    return cv2.merge([bb, gg, rr])  # OpenCV BGR


def _numpy_to_himage(img: np.ndarray):
    """Chuyển numpy (BGR/GRAY) sang HALCON HImage."""
    if not HALCON_AVAILABLE:
        raise RuntimeError("HALCON không khả dụng")
    if img.ndim == 2:
        h, w = img.shape
        return ha.gen_image1("byte", w, h, img.tobytes())
    h, w, _ = img.shape
    b, g, r = cv2.split(img)
    return ha.gen_image3(
        "byte", w, h, r.tobytes(), g.tobytes(), b.tobytes()
    )


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 1. Threshold + Blob analysis
# ---------------------------------------------------------------------------

def threshold_blob(
    img: np.ndarray,
    min_gray: int = 0,
    max_gray: int = 128,
    min_area: int = 100,
    max_area: int = 10_000_000,
) -> OperatorResult:
    """Threshold + connection + select_shape (area).

    Tương đương HALCON:
        threshold(Image, Region, MinGray, MaxGray)
        connection(Region, ConnectedRegions)
        select_shape(ConnectedRegions, Selected, 'area', 'and', MinArea, MaxArea)
    """
    gray = to_gray(img)
    log: list[str] = []

    if HALCON_AVAILABLE:
        try:
            himg = _numpy_to_himage(gray)
            region = ha.threshold(himg, min_gray, max_gray)
            connected = ha.connection(region)
            selected = ha.select_shape(
                connected, "area", "and", min_area, max_area
            )
            count = ha.count_obj(selected)
            log.append(f"[HALCON] threshold({min_gray},{max_gray}) -> {count} blob")
            blobs = []
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for i in range(1, count + 1):
                obj = ha.select_obj(selected, i)
                area = float(ha.area_center(obj)[0][0])
                row, col = ha.area_center(obj)[1][0], ha.area_center(obj)[2][0]
                r1, c1, r2, c2 = ha.smallest_rectangle1(obj)
                cv2.rectangle(
                    overlay, (int(c1), int(r1)), (int(c2), int(r2)), (0, 220, 80), 2
                )
                cv2.circle(overlay, (int(col), int(row)), 3, (0, 220, 255), -1)
                blobs.append(
                    {"id": i, "area": area, "cx": float(col), "cy": float(row)}
                )
            return OperatorResult(
                image=overlay,
                metrics={"count": count, "blobs": blobs},
                log=log,
            )
        except Exception as exc:  # pragma: no cover
            log.append(f"[HALCON] error -> fallback OpenCV: {exc}")

    # ---- OpenCV fallback ----
    _, binary = cv2.threshold(gray, max_gray, 255, cv2.THRESH_BINARY_INV)
    binary[gray < min_gray] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blobs = []
    kept = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        kept += 1
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 220, 80), 2)
        cv2.circle(overlay, (int(cx), int(cy)), 3, (0, 220, 255), -1)
        blobs.append({"id": kept, "area": area, "cx": float(cx), "cy": float(cy)})
    log.append(
        f"[OpenCV fallback] threshold({min_gray},{max_gray}) -> {kept} blob"
    )
    return OperatorResult(
        image=overlay, metrics={"count": kept, "blobs": blobs}, log=log
    )


# ---------------------------------------------------------------------------
# 2. Edge detection
# ---------------------------------------------------------------------------

def edges_sub_pix(
    img: np.ndarray,
    method: str = "canny",
    alpha: float = 1.0,
    low: int = 40,
    high: int = 120,
) -> OperatorResult:
    """Phát hiện cạnh.

    Tương đương HALCON: `edges_sub_pix(Image, Edges, 'canny', Alpha, Low, High)`.
    """
    gray = to_gray(img)
    log: list[str] = []

    if HALCON_AVAILABLE:
        try:
            himg = _numpy_to_himage(gray)
            edges = ha.edges_sub_pix(himg, method, alpha, low, high)
            edge_count = ha.count_obj(edges)
            # render: convert sub-pix XLD edges -> binary image
            edge_region = ha.gen_region_contour_xld(edges, "filled")
            edge_img = ha.region_to_bin(
                edge_region, 255, 0, gray.shape[1], gray.shape[0]
            )
            mask = _himage_to_numpy(edge_img)
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            overlay[mask_gray > 0] = (0, 220, 255)
            log.append(
                f"[HALCON] edges_sub_pix({method}, alpha={alpha}) -> {edge_count} contour"
            )
            return OperatorResult(
                image=overlay,
                metrics={"edge_count": int(edge_count)},
                log=log,
            )
        except Exception as exc:  # pragma: no cover
            log.append(f"[HALCON] error -> fallback OpenCV: {exc}")

    # ---- OpenCV fallback ----
    if method == "canny":
        edges = cv2.Canny(gray, low, high)
    elif method == "sobel":
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        edges = np.uint8(np.clip(mag * alpha, 0, 255))
        edges = (edges > low).astype(np.uint8) * 255
    else:
        edges = cv2.Canny(gray, low, high)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[edges > 0] = (0, 220, 255)
    log.append(f"[OpenCV fallback] {method}(low={low}, high={high})")
    return OperatorResult(
        image=overlay,
        metrics={"edge_pixels": int(np.count_nonzero(edges))},
        log=log,
    )


# ---------------------------------------------------------------------------
# 3. Shape-based template matching
# ---------------------------------------------------------------------------

def shape_match(
    img: np.ndarray,
    template: np.ndarray,
    min_score: float = 0.6,
    num_matches: int = 5,
    angle_start: float = -0.39,  # ~ -22 deg
    angle_extent: float = 0.78,  #  ~ 45 deg total
) -> OperatorResult:
    """Tìm template trong ảnh.

    HALCON: `create_shape_model` + `find_shape_model`.
    Fallback: cv2.matchTemplate.
    """
    gray = to_gray(img)
    tpl_gray = to_gray(template)
    log: list[str] = []

    if HALCON_AVAILABLE:
        try:
            himg = _numpy_to_himage(gray)
            htpl = _numpy_to_himage(tpl_gray)
            model_id = ha.create_shape_model(
                htpl,
                "auto",
                angle_start,
                angle_extent,
                "auto",
                "auto",
                "use_polarity",
                "auto",
                "auto",
            )
            row, col, angle, score = ha.find_shape_model(
                himg,
                model_id,
                angle_start,
                angle_extent,
                min_score,
                num_matches,
                0.5,
                "least_squares",
                0,
                0.9,
            )
            ha.clear_shape_model(model_id)
            overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            matches = []
            th, tw = tpl_gray.shape[:2]
            for i, s in enumerate(score):
                r, c, a = float(row[i]), float(col[i]), float(angle[i])
                _draw_rotated_box(overlay, c, r, tw, th, a)
                matches.append({"row": r, "col": c, "angle": a, "score": float(s)})
            log.append(f"[HALCON] find_shape_model -> {len(matches)} match")
            return OperatorResult(
                image=overlay,
                metrics={"matches": matches, "count": len(matches)},
                log=log,
            )
        except Exception as exc:  # pragma: no cover
            log.append(f"[HALCON] error -> fallback OpenCV: {exc}")

    # ---- OpenCV fallback (no rotation, NCC) ----
    res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    th, tw = tpl_gray.shape[:2]
    matches: list[dict[str, Any]] = []
    res_copy = res.copy()
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for _ in range(num_matches):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_copy)
        if max_val < min_score:
            break
        x, y = max_loc
        cv2.rectangle(overlay, (x, y), (x + tw, y + th), (0, 220, 80), 2)
        cx, cy = x + tw / 2, y + th / 2
        cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 220, 255), -1)
        matches.append(
            {"row": float(cy), "col": float(cx), "angle": 0.0, "score": float(max_val)}
        )
        # suppress region
        x0 = max(0, x - tw // 2)
        y0 = max(0, y - th // 2)
        x1 = min(res_copy.shape[1], x + tw // 2)
        y1 = min(res_copy.shape[0], y + th // 2)
        res_copy[y0:y1, x0:x1] = -1.0
    log.append(f"[OpenCV fallback] matchTemplate(NCC) -> {len(matches)} match")
    return OperatorResult(
        image=overlay, metrics={"matches": matches, "count": len(matches)}, log=log
    )


def _draw_rotated_box(img, cx, cy, w, h, angle_rad):
    box = cv2.boxPoints(((cx, cy), (w, h), -np.degrees(angle_rad)))
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, (0, 220, 80), 2)
    cv2.circle(img, (int(cx), int(cy)), 4, (0, 220, 255), -1)


# ---------------------------------------------------------------------------
# 4. Measure 1D - đo cạnh dọc theo segment
# ---------------------------------------------------------------------------

def measure_pairs(
    img: np.ndarray,
    row1: int,
    col1: int,
    row2: int,
    col2: int,
    sigma: float = 1.0,
    threshold: int = 30,
) -> OperatorResult:
    """Đo cặp cạnh (rising-falling) dọc segment.

    HALCON: `gen_measure_rectangle2` / `measure_pairs`.
    Fallback: lấy profile dọc segment, tìm điểm có |gradient| > threshold.
    """
    gray = to_gray(img)
    log: list[str] = []
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(overlay, (col1, row1), (col2, row2), (255, 200, 0), 1)

    # Sample profile theo Bresenham-ish
    length = int(np.hypot(col2 - col1, row2 - row1))
    if length < 2:
        return OperatorResult(image=overlay, metrics={}, log=["segment quá ngắn"])
    xs = np.linspace(col1, col2, length).astype(np.int32)
    ys = np.linspace(row1, row2, length).astype(np.int32)
    xs = np.clip(xs, 0, gray.shape[1] - 1)
    ys = np.clip(ys, 0, gray.shape[0] - 1)
    profile = gray[ys, xs].astype(np.float32)

    # Gaussian smooth
    k = max(3, int(sigma * 6) | 1)
    profile_s = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), sigma).ravel()
    grad = np.gradient(profile_s)

    edges_pos = []
    i = 1
    while i < len(grad) - 1:
        g = grad[i]
        if abs(g) >= threshold:
            polarity = "rising" if g > 0 else "falling"
            edges_pos.append((i, polarity, float(g)))
            # skip a few samples to avoid duplicate
            i += 3
        else:
            i += 1

    pairs = []
    for j in range(0, len(edges_pos) - 1):
        p1, pol1, g1 = edges_pos[j]
        p2, pol2, g2 = edges_pos[j + 1]
        if pol1 != pol2:
            pairs.append({"start": p1, "end": p2, "width_px": p2 - p1})

    for idx, (i, pol, g) in enumerate(edges_pos):
        x, y = int(xs[i]), int(ys[i])
        color = (0, 255, 100) if pol == "rising" else (0, 100, 255)
        cv2.circle(overlay, (x, y), 5, color, -1)
        cv2.putText(
            overlay,
            f"{idx}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    log.append(
        f"[Measure] {len(edges_pos)} edge, {len(pairs)} pair (sigma={sigma}, th={threshold})"
    )
    return OperatorResult(
        image=overlay,
        metrics={
            "edges": [
                {"index": int(p), "polarity": pol, "amplitude": amp}
                for p, pol, amp in edges_pos
            ],
            "pairs": pairs,
        },
        log=log,
    )
