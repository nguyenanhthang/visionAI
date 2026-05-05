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


# ---------------------------------------------------------------------------
# 5. Pre-process filters
# ---------------------------------------------------------------------------

def apply_filter(
    img: np.ndarray,
    method: str = "gauss",
    ksize: int = 5,
    sigma: float = 1.5,
) -> OperatorResult:
    """Smoothing / sharpen filter.

    HALCON: gauss_filter / median_image / mean_image / emphasize.
    """
    log: list[str] = []
    k = max(3, int(ksize) | 1)

    if HALCON_AVAILABLE:
        try:
            himg = _numpy_to_himage(img if img.ndim == 2 else to_gray(img))
            if method == "gauss":
                out = ha.gauss_filter(himg, k)
            elif method == "median":
                out = ha.median_image(himg, "circle", max(1, k // 2), "mirrored")
            elif method == "mean":
                out = ha.mean_image(himg, k, k)
            elif method == "sharpen":
                out = ha.emphasize(himg, k, k, 1.0)
            else:
                out = himg
            arr = _himage_to_numpy(out)
            log.append(f"[HALCON] filter={method} ksize={k}")
            return OperatorResult(image=arr, metrics={"method": method, "ksize": k}, log=log)
        except Exception as exc:  # pragma: no cover
            log.append(f"[HALCON] error -> fallback OpenCV: {exc}")

    # OpenCV fallback
    if method == "gauss":
        out = cv2.GaussianBlur(img, (k, k), sigma)
    elif method == "median":
        out = cv2.medianBlur(img, k)
    elif method == "mean":
        out = cv2.blur(img, (k, k))
    elif method == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(img, -1, kernel)
    else:
        out = img.copy()
    log.append(f"[OpenCV] filter={method} ksize={k}")
    return OperatorResult(image=out, metrics={"method": method, "ksize": k}, log=log)


# ---------------------------------------------------------------------------
# 6. Histogram analysis
# ---------------------------------------------------------------------------

def histogram(img: np.ndarray) -> OperatorResult:
    """Tính histogram + thống kê + render overlay biểu đồ."""
    gray = to_gray(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    mean_v = float(np.mean(gray))
    std_v = float(np.std(gray))
    min_v = int(np.min(gray))
    max_v = int(np.max(gray))
    median_v = int(np.median(gray))

    # Render: image + histogram strip dưới
    h, w = gray.shape
    strip_h = max(120, h // 4)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = np.full((h + strip_h + 8, w, 3), 28, dtype=np.uint8)
    canvas[:h] = overlay

    if hist.max() > 0:
        norm = (hist / hist.max() * (strip_h - 14)).astype(np.int32)
        bar_w = max(1, w // 256)
        for i in range(256):
            x0 = i * bar_w
            x1 = min(w, (i + 1) * bar_w)
            y0 = h + 8 + (strip_h - 14 - norm[i])
            cv2.rectangle(canvas, (x0, y0), (x1, h + 8 + strip_h - 14),
                          (54, 197, 214), -1)
        # mean/median markers
        for v, color, label in (
            (int(mean_v), (108, 217, 137), "mean"),
            (int(median_v), (255, 180, 84), "median"),
        ):
            x = int(v * w / 256)
            cv2.line(canvas, (x, h + 8), (x, h + 8 + strip_h - 14), color, 1)

    return OperatorResult(
        image=canvas,
        metrics={
            "mean": mean_v, "std": std_v, "median": median_v,
            "min": min_v, "max": max_v,
            "p1": int(np.percentile(gray, 1)),
            "p99": int(np.percentile(gray, 99)),
        },
        log=[f"[Histogram] mean={mean_v:.1f} std={std_v:.1f} min={min_v} max={max_v}"],
    )


# ---------------------------------------------------------------------------
# 7. ID Read (QR / Data Matrix / Barcode)
# ---------------------------------------------------------------------------

def decode_codes(img: np.ndarray) -> OperatorResult:
    """Decode QR + barcode bằng OpenCV detector. (HALCON: find_data_code_2d.)"""
    gray = to_gray(img)
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    found: list[dict[str, Any]] = []

    # QR
    try:
        qr = cv2.QRCodeDetector()
        ok, decoded, points, _ = qr.detectAndDecodeMulti(gray)
        if ok and points is not None:
            for txt, poly in zip(decoded, points):
                pts = np.intp(poly)
                cv2.polylines(overlay, [pts], True, (108, 217, 137), 2)
                if txt:
                    cv2.putText(overlay, f"QR: {txt}", tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (108, 217, 137), 2)
                found.append({"type": "QR", "data": txt, "polygon": pts.tolist()})
    except Exception:
        pass

    # Barcode (1D)
    try:
        bd = cv2.barcode.BarcodeDetector()  # type: ignore[attr-defined]
        ok, decoded, types, points = bd.detectAndDecode(gray)
        if ok and points is not None:
            for txt, t, poly in zip(decoded, types, points):
                pts = np.intp(poly)
                cv2.polylines(overlay, [pts], True, (255, 180, 84), 2)
                cv2.putText(overlay, f"{t}: {txt}", tuple(pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 84), 2)
                found.append({"type": t, "data": txt, "polygon": pts.tolist()})
    except Exception:
        pass

    return OperatorResult(
        image=overlay,
        metrics={"count": len(found), "codes": found},
        log=[f"[ID Read] {len(found)} code(s) detected"],
    )


# ---------------------------------------------------------------------------
# 8. Color stats trong ROI (hoặc full image)
# ---------------------------------------------------------------------------

def color_stats(
    img: np.ndarray,
    x: int = 0, y: int = 0, w: int = 0, h: int = 0,
) -> OperatorResult:
    """Thống kê màu trong ROI (mean RGB/HSV, dominant)."""
    H, W = img.shape[:2]
    if w <= 0 or h <= 0:
        x, y, w, h = 0, 0, W, H
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = min(w, W - x); h = min(h, H - y)
    roi = img[y:y + h, x:x + w]
    if roi.ndim == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    bgr_mean = roi.reshape(-1, 3).mean(axis=0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)

    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (54, 197, 214), 2)
    swatch_color = tuple(int(v) for v in bgr_mean)
    cv2.rectangle(overlay, (x, y - 24), (x + 80, y - 4), swatch_color, -1)
    cv2.rectangle(overlay, (x, y - 24), (x + 80, y - 4), (255, 255, 255), 1)

    return OperatorResult(
        image=overlay,
        metrics={
            "roi": {"x": x, "y": y, "w": w, "h": h},
            "bgr_mean": [round(float(v), 2) for v in bgr_mean],
            "rgb_mean": [round(float(bgr_mean[2]), 2),
                         round(float(bgr_mean[1]), 2),
                         round(float(bgr_mean[0]), 2)],
            "hsv_mean": [round(float(v), 2) for v in hsv_mean],
        },
        log=[f"[Color] BGR≈({bgr_mean[0]:.0f},{bgr_mean[1]:.0f},{bgr_mean[2]:.0f}) "
             f"HSV≈({hsv_mean[0]:.0f},{hsv_mean[1]:.0f},{hsv_mean[2]:.0f})"],
    )


# ---------------------------------------------------------------------------
# 9. Morphology (dilation / erosion / opening / closing)
# ---------------------------------------------------------------------------

def morphology(
    img: np.ndarray,
    op: str = "dilate",
    ksize: int = 5,
    shape: str = "rect",
    iterations: int = 1,
) -> OperatorResult:
    """Morphology trên grayscale.

    HALCON: dilation_circle / erosion_circle / opening_circle / closing_circle.
    """
    gray = to_gray(img)
    log: list[str] = []
    k = max(1, int(ksize))

    if HALCON_AVAILABLE:
        try:
            himg = _numpy_to_himage(gray)
            radius = float(k)
            if op == "dilate":
                out = ha.dilation_circle(ha.threshold(himg, 1, 255), radius)
            elif op == "erode":
                out = ha.erosion_circle(ha.threshold(himg, 1, 255), radius)
            elif op == "open":
                out = ha.opening_circle(ha.threshold(himg, 1, 255), radius)
            elif op == "close":
                out = ha.closing_circle(ha.threshold(himg, 1, 255), radius)
            else:
                out = ha.threshold(himg, 1, 255)
            mask_img = ha.region_to_bin(out, 255, 0, gray.shape[1], gray.shape[0])
            arr = _himage_to_numpy(mask_img)
            log.append(f"[HALCON] {op}_circle radius={radius}")
            return OperatorResult(image=arr, metrics={"op": op, "ksize": k}, log=log)
        except Exception as exc:  # pragma: no cover
            log.append(f"[HALCON] error -> fallback OpenCV: {exc}")

    cv_shape = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}.get(
        shape, cv2.MORPH_RECT
    )
    kernel = cv2.getStructuringElement(cv_shape, (k, k))
    if op == "dilate":
        out = cv2.dilate(gray, kernel, iterations=iterations)
    elif op == "erode":
        out = cv2.erode(gray, kernel, iterations=iterations)
    elif op == "open":
        out = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op == "close":
        out = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif op == "gradient":
        out = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif op == "tophat":
        out = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif op == "blackhat":
        out = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    else:
        out = gray.copy()
    log.append(f"[OpenCV] morphology={op} ksize={k} iter={iterations}")
    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return OperatorResult(image=out_bgr, metrics={"op": op, "ksize": k}, log=log)


# ---------------------------------------------------------------------------
# 10. Adaptive threshold
# ---------------------------------------------------------------------------

def adaptive_threshold(
    img: np.ndarray,
    block_size: int = 15,
    offset: int = 5,
    method: str = "mean",
) -> OperatorResult:
    """Adaptive threshold (HALCON: dyn_threshold)."""
    gray = to_gray(img)
    bs = max(3, int(block_size) | 1)
    cv_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    binary = cv2.adaptiveThreshold(
        gray, 255, cv_method, cv2.THRESH_BINARY_INV, bs, int(offset)
    )
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[binary > 0] = (54, 197, 214)
    return OperatorResult(
        image=overlay,
        metrics={
            "method": method,
            "block_size": bs,
            "offset": int(offset),
            "fg_pixels": int(np.count_nonzero(binary)),
        },
        log=[f"[Adaptive] {method} block={bs} offset={offset}"],
    )


# ---------------------------------------------------------------------------
# 11. Color segment → mask (HSV range)
# ---------------------------------------------------------------------------

def color_segment(
    img: np.ndarray,
    h_min: int = 0, h_max: int = 179,
    s_min: int = 0, s_max: int = 255,
    v_min: int = 0, v_max: int = 255,
) -> OperatorResult:
    """Phân vùng theo HSV range; trả về ảnh overlay + mask trong metrics."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Overlay: tô màu accent lên vùng matched
    overlay = img.copy()
    accent = np.zeros_like(img); accent[:, :] = (54, 197, 214)
    blend = cv2.addWeighted(overlay, 0.6, accent, 0.4, 0)
    overlay[mask > 0] = blend[mask > 0]
    return OperatorResult(
        image=overlay,
        metrics={
            "hsv_lower": [h_min, s_min, v_min],
            "hsv_upper": [h_max, s_max, v_max],
            "mask_pixels": int(np.count_nonzero(mask)),
            "mask_ratio": round(float(np.count_nonzero(mask)) / mask.size, 4),
            "_mask": mask,  # GUI sẽ rút ra để set làm current mask
        },
        log=[f"[ColorSegment] H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}] "
             f"-> {np.count_nonzero(mask)} px"],
    )


# ---------------------------------------------------------------------------
# 12. Contour analysis
# ---------------------------------------------------------------------------

def contour_analysis(
    img: np.ndarray,
    min_area: int = 50,
    max_area: int = 10_000_000,
    approx_eps: float = 0.01,
) -> OperatorResult:
    """Tìm contour, xấp xỉ polygon, vẽ + thống kê (perimeter/area/circularity)."""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    items = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        peri = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        eps = max(1.0, approx_eps * peri)
        approx = cv2.approxPolyDP(c, eps, True)
        cv2.drawContours(overlay, [c], -1, (108, 217, 137), 1)
        cv2.drawContours(overlay, [approx], -1, (54, 197, 214), 2)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
        cv2.putText(
            overlay, f"#{len(items)+1}", (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 84), 1, cv2.LINE_AA,
        )
        items.append({
            "id": len(items) + 1,
            "area": float(area),
            "perimeter": float(peri),
            "circularity": round(float(circularity), 4),
            "vertices": int(len(approx)),
            "cx": cx, "cy": cy,
        })
    return OperatorResult(
        image=overlay,
        metrics={"count": len(items), "contours": items},
        log=[f"[Contours] {len(items)} (min_area={min_area}, eps={approx_eps})"],
    )


# ---------------------------------------------------------------------------
# 13. Image diff (so sánh với reference / golden image)
# ---------------------------------------------------------------------------

def image_diff(
    img: np.ndarray,
    reference: np.ndarray,
    threshold: int = 30,
    blur: int = 5,
) -> OperatorResult:
    """Diff image với reference (golden template), highlight defect."""
    if img.shape != reference.shape:
        reference = cv2.resize(reference, (img.shape[1], img.shape[0]))
    g1 = to_gray(img)
    g2 = to_gray(reference)
    if blur > 1:
        b = max(3, int(blur) | 1)
        g1 = cv2.GaussianBlur(g1, (b, b), 0)
        g2 = cv2.GaussianBlur(g2, (b, b), 0)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    overlay = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (84, 96, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 5:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (84, 96, 255), 2)
        defects.append({"x": x, "y": y, "w": w, "h": h, "area": float(a)})
    return OperatorResult(
        image=overlay,
        metrics={
            "defect_count": len(defects),
            "defect_pixels": int(np.count_nonzero(mask)),
            "defects": defects,
        },
        log=[f"[Diff] threshold={threshold} -> {len(defects)} defect"],
    )


# ---------------------------------------------------------------------------
# 14. Mask helpers
# ---------------------------------------------------------------------------

def mask_from_gray_range(
    img: np.ndarray, min_gray: int = 0, max_gray: int = 128
) -> np.ndarray:
    gray = to_gray(img)
    return cv2.inRange(gray, int(min_gray), int(max_gray))


def mask_from_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    mask[y0:y1, x0:x1] = 255
    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Áp mask: ngoài mask = 0 (HALCON: reduce_domain)."""
    if mask is None:
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        return cv2.bitwise_and(img, img, mask=mask)
    return cv2.bitwise_and(img, img, mask=mask)
