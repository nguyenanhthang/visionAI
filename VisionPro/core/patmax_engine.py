"""
core/patmax_engine.py — v2
Fix: thuật toán search đáng tin cậy hơn.
Dùng 3 phương pháp song song:
  1. Raw patch NCC (nhanh, chính xác với ánh sáng đồng đều)
  2. Edge-on-edge (bất biến với thay đổi màu/sáng)
  3. Gradient orientation (Cognex-style, robust nhất)
Lấy score = weighted max của 3 phương pháp.
Debug: in score tối đa ra console để người dùng điều chỉnh threshold.
"""
from __future__ import annotations
import cv2
import numpy as np
import json, os, math, hashlib
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class PatMaxResult:
    found: bool
    score: float
    x: float
    y: float
    angle: float
    scale: float
    width: float
    height: float
    corners: List[Tuple[float, float]] = field(default_factory=list)
    # Origin (điểm tham chiếu) đã transform theo angle/scale của result
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class PatMaxModel:
    trained: bool = False
    train_roi: Optional[Tuple[int,int,int,int]] = None
    origin_x: float = 0.0
    origin_y: float = 0.0
    pattern_w: int = 0
    pattern_h: int = 0
    # Raw patch (BGR) — dùng cho NCC
    patch_bgr: Optional[np.ndarray] = None
    # Gray patch
    patch_gray: Optional[np.ndarray] = None
    # Canny edges
    edge_image: Optional[np.ndarray] = None
    # Thumbnail hiển thị
    thumbnail: Optional[np.ndarray] = None
    edge_count: int = 0
    model_hash: str = ""
    # Shape info: "rect" | "circle" | "ellipse" | "polygon"
    shape_type: str = "rect"
    shape_data: Optional[dict] = None
    # Mask (h × w) — uint8 0/255, áp lên patch khi train với non-rect shape
    mask: Optional[np.ndarray] = None
    # Train mode: "evaluate" (DOFs at runtime) | "create" (precomputed templates)
    train_mode: str = "evaluate"
    # Precomputed templates khi train_mode == "create"
    precomputed_templates: Optional[list] = None
    # Search params
    accept_threshold: float = 0.5
    angle_low: float = 0.0
    angle_high: float = 0.0
    angle_step: float = 5.0
    scale_low: float = 1.0
    scale_high: float = 1.0
    scale_step: float = 0.1
    num_results: int = 1
    overlap_threshold: float = 0.5
    # Canny params dùng lúc train
    canny_low: int = 50
    canny_high: int = 150

    def is_valid(self) -> bool:
        return (self.trained
                and self.patch_gray is not None
                and self.pattern_w > 4
                and self.pattern_h > 4)


# ═══════════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════════
def _build_template(patch_gray: np.ndarray,
                     edge_image: np.ndarray,
                     angle: float,
                     scale: float) -> dict:
    """Tạo template (rotated/scaled) từ patch base + edge base."""
    pw = max(4, int(patch_gray.shape[1] * scale))
    ph = max(4, int(patch_gray.shape[0] * scale))
    patch = cv2.resize(patch_gray, (pw, ph))
    edge_p = cv2.resize(edge_image, (pw, ph))
    if abs(angle) > 0.1:
        rad = math.radians(angle)
        cos_a = abs(math.cos(rad)); sin_a = abs(math.sin(rad))
        nW = int(ph * sin_a + pw * cos_a) + 2
        nH = int(ph * cos_a + pw * sin_a) + 2
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        M[0, 2] += (nW - pw) / 2; M[1, 2] += (nH - ph) / 2
        patch_rot = cv2.warpAffine(patch, M, (nW, nH),
                                    borderMode=cv2.BORDER_REPLICATE)
        edge_rot  = cv2.warpAffine(edge_p, M, (nW, nH))
    else:
        nW, nH = pw, ph
        patch_rot = patch
        edge_rot  = edge_p
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dil = cv2.dilate(edge_rot, kernel)
    return {"angle": float(angle), "scale": float(scale),
            "patch": patch_rot, "edge_dil": edge_dil,
            "nW": nW, "nH": nH}


def _build_search_grid(angle_low, angle_high, angle_step,
                        scale_low, scale_high, scale_step):
    if abs(angle_high - angle_low) < 0.5:
        angles = [0.0]
    else:
        step = max(0.5, angle_step)
        angles = list(np.arange(angle_low, angle_high + step * 0.5, step))
        if 0.0 not in [round(a, 2) for a in angles]:
            angles.append(0.0)
    if abs(scale_high - scale_low) < 0.01:
        scales = [1.0]
    else:
        step_s = max(0.01, scale_step)
        scales = list(np.arange(scale_low, scale_high + step_s * 0.5, step_s))
    return angles, scales


def precompute_templates(model: PatMaxModel) -> int:
    """Build precomputed_templates từ ranges đã lưu trong model.
    Trả về số lượng template đã build."""
    if not model.is_valid():
        return 0
    angles, scales = _build_search_grid(
        model.angle_low, model.angle_high, model.angle_step,
        model.scale_low, model.scale_high, 0.1)
    tmpls = []
    for sc in scales:
        for ang in angles:
            tmpls.append(_build_template(model.patch_gray, model.edge_image, ang, sc))
    model.precomputed_templates = tmpls
    return len(tmpls)


def train_patmax(image: np.ndarray,
                 roi: Tuple[int,int,int,int],
                 origin_offset: Tuple[float,float] = (0.5, 0.5),
                 canny_low: int = 50,
                 canny_high: int = 150,
                 shape_type: str = "rect",
                 shape_data: Optional[dict] = None,
                 train_mode: str = "evaluate",
                 angle_low: float = 0.0,
                 angle_high: float = 0.0,
                 angle_step: float = 5.0,
                 scale_low: float = 1.0,
                 scale_high: float = 1.0,
                 scale_step: float = 0.1) -> PatMaxModel:

    x, y, w, h = roi
    H, W = image.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))

    bgr  = image[y:y+h, x:x+w].copy()
    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray_smooth, canny_low, canny_high)

    # Build mask trong toạ độ patch (h × w) cho non-rect shapes
    mask = _build_shape_mask(shape_type, shape_data, x, y, w, h)
    if mask is not None:
        # Outside-of-shape: set gray = mean (giảm ảnh hưởng NCC), edges = 0
        mean_val = int(gray_smooth[mask > 0].mean()) if np.count_nonzero(mask) else 127
        gray_smooth = np.where(mask > 0, gray_smooth, mean_val).astype(np.uint8)
        edges = np.where(mask > 0, edges, 0).astype(np.uint8)

    # Thumbnail 80×80
    th = cv2.resize(bgr, (80, 80))
    e_small = cv2.resize(edges, (80, 80))
    th[e_small > 0] = [0, 220, 80]

    ox = w * origin_offset[0]
    oy = h * origin_offset[1]
    model_hash = hashlib.md5(gray.tobytes()).hexdigest()[:8]

    print(f"[PatMax Train] ROI=({x},{y},{w},{h}) shape={shape_type} "
          f"mode={train_mode}  edges={int(np.count_nonzero(edges))}  hash={model_hash}")

    model = PatMaxModel(
        trained=True,
        train_roi=(x, y, w, h),
        origin_x=ox, origin_y=oy,
        pattern_w=w, pattern_h=h,
        patch_bgr=bgr,
        patch_gray=gray_smooth,
        edge_image=edges,
        thumbnail=th,
        edge_count=int(np.count_nonzero(edges)),
        model_hash=model_hash,
        canny_low=canny_low,
        canny_high=canny_high,
        shape_type=shape_type,
        shape_data=dict(shape_data) if shape_data else None,
        mask=mask,
        train_mode=train_mode if train_mode in ("evaluate", "create") else "evaluate",
        angle_low=angle_low, angle_high=angle_high, angle_step=angle_step,
        scale_low=scale_low, scale_high=scale_high,
    )
    if model.train_mode == "create":
        n = precompute_templates(model)
        print(f"[PatMax Train] precomputed {n} DOF templates")
    return model


def _build_shape_mask(shape_type: str, shape_data: Optional[dict],
                       roi_x: int, roi_y: int, w: int, h: int
                       ) -> Optional[np.ndarray]:
    """Tạo mask (h × w) uint8 0/255 cho non-rect shape, toạ độ ảnh → patch."""
    if not shape_data or shape_type == "rect":
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    if shape_type == "ellipse":
        cx = int((shape_data.get("x", roi_x) - roi_x) + shape_data.get("w", w) / 2)
        cy = int((shape_data.get("y", roi_y) - roi_y) + shape_data.get("h", h) / 2)
        ax = max(1, int(shape_data.get("w", w) / 2))
        ay = max(1, int(shape_data.get("h", h) / 2))
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    elif shape_type == "circle":
        cx = int(shape_data.get("cx", roi_x + w / 2) - roi_x)
        cy = int(shape_data.get("cy", roi_y + h / 2) - roi_y)
        r = max(1, int(shape_data.get("r", min(w, h) / 2)))
        cv2.circle(mask, (cx, cy), r, 255, -1)
    elif shape_type == "polygon":
        pts = shape_data.get("pts") or []
        if len(pts) >= 3:
            arr = np.array([[int(px - roi_x), int(py - roi_y)] for px, py in pts],
                            dtype=np.int32)
            cv2.fillPoly(mask, [arr], 255)
        else:
            return None
    else:
        return None
    return mask


# ═══════════════════════════════════════════════════════════════
#  SEARCH — single angle/scale attempt
# ═══════════════════════════════════════════════════════════════
def _match_template(gray_img: np.ndarray,
                     img_edges: np.ndarray,
                     t: Dict,
                     max_locations: int = 1) -> List[Dict]:
    """
    Match một template đã build sẵn — trả TOP-K peaks trên score map.
    Cho phép multi-object detection (mỗi peak ≈ một object). Caller
    sau đó phân biệt overlap qua NMS.
    """
    patch_rot = t["patch"]
    edge_rot_d = t["edge_dil"]
    nW = t["nW"]; nH = t["nH"]
    angle = t["angle"]; scale = t["scale"]
    H, W = gray_img.shape[:2]

    if nW >= W - 2 or nH >= H - 2:
        return []

    # Score maps đầy đủ
    try:
        res_ncc = cv2.matchTemplate(gray_img, patch_rot, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return []
    img_e_f   = img_edges.astype(np.float32) / 255.0
    templ_e_f = edge_rot_d.astype(np.float32) / 255.0
    try:
        res_edge = cv2.matchTemplate(img_e_f, templ_e_f, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        res_edge = np.zeros_like(res_ncc)
    try:
        res_sq = cv2.matchTemplate(gray_img, patch_rot, cv2.TM_SQDIFF_NORMED)
        res_sq_inv = 1.0 - res_sq          # nhỏ = tốt → đảo
    except cv2.error:
        res_sq_inv = np.zeros_like(res_ncc)

    s_ncc_map  = np.maximum(res_ncc, 0.0).astype(np.float32)
    s_edge_map = np.maximum(res_edge, 0.0).astype(np.float32)
    s_sq_map   = np.clip(res_sq_inv, 0.0, 1.0).astype(np.float32)
    score_map  = 0.5 * s_ncc_map + 0.3 * s_edge_map + 0.2 * s_sq_map

    # Local NMS: suppress peaks gần nhau bằng cách lấy max trong cửa sổ
    win = max(3, min(nW, nH) // 2)
    if win % 2 == 0: win += 1
    kernel = np.ones((win, win), dtype=np.uint8)
    local_max = cv2.dilate(score_map, kernel)
    peaks_mask = (score_map == local_max) & (score_map > 0)

    ys, xs = np.where(peaks_mask)
    if len(ys) == 0:
        return []
    scores = score_map[ys, xs]

    # Lấy top-K peaks
    K = max(1, int(max_locations))
    if len(scores) > K:
        idx = np.argpartition(-scores, K)[:K]
        idx = idx[np.argsort(-scores[idx])]
    else:
        idx = np.argsort(-scores)

    out: List[Dict] = []
    for i in idx:
        ty = int(ys[i]); tx = int(xs[i])
        out.append({
            "score":  float(scores[i]),
            "s_ncc":  float(s_ncc_map[ty, tx]),
            "s_edge": float(s_edge_map[ty, tx]),
            "s_sq":   float(s_sq_map[ty, tx]),
            "cx": float(tx + nW / 2),
            "cy": float(ty + nH / 2),
            "tx": tx, "ty": ty,
            "nW": nW, "nH": nH,
            "angle": angle, "scale": scale,
        })
    return out


# ═══════════════════════════════════════════════════════════════
#  RUN PATMAX — main search
# ═══════════════════════════════════════════════════════════════
def run_patmax(image: np.ndarray,
               model: PatMaxModel,
               accept_threshold: float = 0.5,
               angle_low: float = 0.0,
               angle_high: float = 0.0,
               angle_step: float = 5.0,
               scale_low: float = 1.0,
               scale_high: float = 1.0,
               scale_step: float = 0.1,
               num_results: int = 1,
               overlap_threshold: float = 0.5,
               ) -> Tuple[List[PatMaxResult], np.ndarray]:

    if not model.is_valid():
        return [], _empty_vis(image)

    bgr      = image if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W     = gray_img.shape

    canny_lo = model.canny_low
    canny_hi = model.canny_high
    img_edges = cv2.Canny(gray_img, canny_lo, canny_hi)

    # Chọn nguồn templates: precomputed (mode "create") hoặc build on-the-fly
    use_precomputed = (model.train_mode == "create"
                       and model.precomputed_templates)

    if use_precomputed:
        templates = model.precomputed_templates
        print(f"[PatMax Search] using {len(templates)} precomputed templates  "
              f"threshold={accept_threshold}")
    else:
        angles, scales = _build_search_grid(angle_low, angle_high, angle_step,
                                              scale_low, scale_high, scale_step)
        templates = [_build_template(model.patch_gray, model.edge_image, ang, sc)
                     for sc in scales for ang in angles]
        print(f"[PatMax Search] evaluate-DOF: {len(templates)} templates "
              f"(angles={len(angles)} × scales={len(scales)})  "
              f"threshold={accept_threshold}")

    candidates: List[Dict] = []
    score_map = np.zeros((H, W), dtype=np.float32)
    best_any = 0.0

    # Oversample peaks per template — global NMS sau đó dedupe vị trí trùng
    peaks_per_template = max(num_results * 4, 8)

    for t in templates:
        results_t = _match_template(gray_img, img_edges, t, peaks_per_template)
        for result in results_t:
            score = result["score"]
            best_any = max(best_any, score)

            tx, ty = result["tx"], result["ty"]
            nW, nH = result["nW"], result["nH"]
            s_for_map = result["s_ncc"]
            if (0 <= ty < H - nH) and (0 <= tx < W - nW):
                cy_i = min(int(result["cy"]), H - 1)
                cx_i = min(int(result["cx"]), W - 1)
                if score_map[cy_i, cx_i] < s_for_map:
                    score_map[cy_i, cx_i] = s_for_map

            if score >= accept_threshold:
                candidates.append(result)

    print(f"[PatMax Search] best_score={best_any:.4f}  "
          f"candidates_above_threshold={len(candidates)}  "
          f"threshold={accept_threshold}")

    if best_any < accept_threshold:
        print(f"[PatMax Search] ⚠  Không tìm thấy. "
              f"Thử giảm threshold xuống {best_any * 0.85:.2f} "
              f"hoặc retrain với Canny phù hợp hơn.")

    # NMS
    candidates.sort(key=lambda d: -d["score"])
    kept: List[Dict] = []
    for cand in candidates:
        overlap = False
        for k in kept:
            dist = math.hypot(cand["cx"] - k["cx"], cand["cy"] - k["cy"])
            min_dim = min(cand["nW"], cand["nH"], k["nW"], k["nH"])
            if dist < min_dim * overlap_threshold:
                overlap = True; break
        if not overlap:
            kept.append(cand)
        if len(kept) >= num_results:
            break

    # Origin offset từ tâm pattern (toạ độ pattern, không clamp)
    pdx = float(model.origin_x) - float(model.pattern_w) / 2.0
    pdy = float(model.origin_y) - float(model.pattern_h) / 2.0

    # Build results
    results: List[PatMaxResult] = []
    for d in kept:
        corners = _rotated_corners(d["cx"], d["cy"], d["nW"], d["nH"], d["angle"])
        rad_o = math.radians(-d["angle"])
        ca = math.cos(rad_o); sa = math.sin(rad_o)
        s = d["scale"] if d["scale"] else 1.0
        ox_t = float(d["cx"]) + s * (pdx * ca - pdy * sa)
        oy_t = float(d["cy"]) + s * (pdx * sa + pdy * ca)
        results.append(PatMaxResult(
            found=True, score=d["score"],
            x=d["cx"], y=d["cy"],
            angle=d["angle"], scale=d["scale"],
            width=float(d["nW"]), height=float(d["nH"]),
            corners=corners,
            origin_x=ox_t, origin_y=oy_t,
        ))

    # Score map visualization (blur để đẹp hơn)
    sm_blurred = cv2.GaussianBlur(score_map, (15, 15), 0)
    sm_vis = _score_map_vis(bgr, sm_blurred)

    return results, sm_vis


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def _rotated_corners(cx, cy, w, h, angle_deg):
    hw, hh = w/2, h/2
    pts = [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    rad = math.radians(-angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    return [(cx + p[0]*ca - p[1]*sa,
             cy + p[0]*sa + p[1]*ca) for p in pts]


def _score_map_vis(image: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    vis = image.copy() if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if score_map.max() > 0.01:
        sm_u8 = (np.clip(score_map / score_map.max(), 0, 1) * 255).astype(np.uint8)
        heat  = cv2.applyColorMap(sm_u8, cv2.COLORMAP_JET)
        cv2.addWeighted(vis, 0.55, heat, 0.45, 0, vis)
    return vis


def _empty_vis(image: np.ndarray) -> np.ndarray:
    vis = image.copy() if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis, "[PatMax] No model trained",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,150,255), 2)
    cv2.putText(vis, "Double-click node -> Draw ROI -> Train",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,180), 1)
    return vis


def draw_patmax_results(image: np.ndarray,
                         results: List[PatMaxResult],
                         model: Optional[PatMaxModel] = None) -> np.ndarray:
    vis = image.copy() if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Origin offset t\u1eeb t\u00e2m pattern (to\u1ea1 \u0111\u1ed9 pattern, c\u00f3 th\u1ec3 \u00e2m ho\u1eb7c >w/h)
    has_origin = False
    pdx = pdy = 0.0
    if model is not None and model.is_valid():
        pw = float(model.pattern_w); ph = float(model.pattern_h)
        pdx = float(model.origin_x) - pw / 2.0
        pdy = float(model.origin_y) - ph / 2.0
        has_origin = True

    for i, r in enumerate(results):
        if not r.found:
            continue
        col = (0,220,80) if r.score >= 0.7 else (0,200,160) if r.score >= 0.5 else (0,150,220)

        # Rotated bounding box
        if r.corners and len(r.corners) == 4:
            pts = np.array(r.corners, dtype=np.int32)
            cv2.polylines(vis, [pts], True, col, 2)

        # T\u00ednh origin (transformed) \u2014 \u0111\u00e2y l\u00e0 MARKER CH\u00cdNH (tham chi\u1ebfu)
        if has_origin:
            rad_o = math.radians(-r.angle)
            ca = math.cos(rad_o); sa = math.sin(rad_o)
            s  = r.scale if r.scale else 1.0
            ox = float(r.x) + s * (pdx * ca - pdy * sa)
            oy = float(r.y) + s * (pdx * sa + pdy * ca)
        else:
            ox = float(r.x); oy = float(r.y)
        ox_i = int(round(ox)); oy_i = int(round(oy))

        # Marker tham chi\u1ebfu: v\u00f2ng xanh (theo score) + X v\u00e0ng \u0111\u00e8 l\u00ean \u2014 g\u1ed9p 1 v\u1ecb tr\u00ed
        sz = max(14, int(min(r.width, r.height) * 0.18))
        cv2.circle(vis, (ox_i, oy_i), 11, col, 2, cv2.LINE_AA)
        cv2.circle(vis, (ox_i, oy_i), 3, col, -1, cv2.LINE_AA)
        o_col = (0, 215, 255)   # v\u00e0ng (BGR)
        cv2.line(vis, (ox_i - 9, oy_i - 9), (ox_i + 9, oy_i + 9),
                 o_col, 2, cv2.LINE_AA)
        cv2.line(vis, (ox_i - 9, oy_i + 9), (ox_i + 9, oy_i - 9),
                 o_col, 2, cv2.LINE_AA)

        # Angle arrow \u2014 ph\u00e1t t\u1eeb marker (origin)
        if abs(r.angle) > 0.5:
            rad = math.radians(-r.angle)
            ex  = int(ox_i + math.cos(rad) * sz * 2)
            ey  = int(oy_i + math.sin(rad) * sz * 2)
            cv2.arrowedLine(vis, (ox_i, oy_i), (ex, ey),
                             (255, 210, 0), 2, tipLength=0.3)

        # Label: score + to\u1ea1 \u0111\u1ed9 origin
        lx = max(0, int(r.x - r.width/2))
        ly = max(16, int(r.y - r.height/2) - 8)
        label = f"#{i+1} {r.score:.3f}"
        if abs(r.angle) > 0.5:
            label += f" {r.angle:+.1f}\u00b0"
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(vis, f"({ox:.0f},{oy:.0f})",
                    (ox_i + 14, oy_i + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, o_col, 1, cv2.LINE_AA)

    n = len(results)
    st_col = (0,220,80) if n > 0 else (0,60,255)
    cv2.putText(vis, f"PatMax: {n} found", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, st_col, 2)
    return vis


# ═══════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ═══════════════════════════════════════════════════════════════
def save_model(model: PatMaxModel, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    base = os.path.splitext(path)[0]

    np_data = {}
    for attr in ("patch_bgr","patch_gray","edge_image","thumbnail","mask"):
        arr = getattr(model, attr, None)
        if arr is not None:
            np_data[attr] = arr
    if np_data:
        np.savez_compressed(base + ".npz", **np_data)

    meta = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in model.__dict__.items()
            if not isinstance(getattr(model,k,None), np.ndarray)
            and not isinstance(getattr(model,k,None), type(None))
            or k in ("train_roi",)}
    # Remove numpy arrays + non-serializable từ meta
    for key in ("patch_bgr","patch_gray","edge_image","thumbnail","mask",
                "precomputed_templates"):
        meta.pop(key, None)

    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(path: str) -> Optional[PatMaxModel]:
    base  = os.path.splitext(path)[0]
    jpath = base + ".json"
    npath = base + ".npz"
    if not os.path.exists(jpath):
        return None
    try:
        with open(jpath) as f:
            meta = json.load(f)
        model = PatMaxModel(
            trained          = meta.get("trained", False),
            train_roi        = tuple(meta["train_roi"]) if meta.get("train_roi") else None,
            origin_x         = meta.get("origin_x", 0.0),
            origin_y         = meta.get("origin_y", 0.0),
            pattern_w        = meta.get("pattern_w", 0),
            pattern_h        = meta.get("pattern_h", 0),
            edge_count       = meta.get("edge_count", 0),
            model_hash       = meta.get("model_hash", ""),
            accept_threshold = meta.get("accept_threshold", 0.5),
            angle_low        = meta.get("angle_low", 0.0),
            angle_high       = meta.get("angle_high", 0.0),
            angle_step       = meta.get("angle_step", 5.0),
            scale_low        = meta.get("scale_low", 1.0),
            scale_high       = meta.get("scale_high", 1.0),
            scale_step       = meta.get("scale_step", 0.1),
            num_results      = meta.get("num_results", 1),
            overlap_threshold= meta.get("overlap_threshold", 0.5),
            canny_low        = meta.get("canny_low", 50),
            canny_high       = meta.get("canny_high", 150),
            shape_type       = meta.get("shape_type", "rect"),
            shape_data       = meta.get("shape_data"),
            train_mode       = meta.get("train_mode", "evaluate"),
        )
        if os.path.exists(npath):
            npz = np.load(npath)
            for attr in ("patch_bgr","patch_gray","edge_image","thumbnail","mask"):
                if attr in npz:
                    setattr(model, attr, npz[attr])
        # Regenerate precomputed templates nếu mode == "create"
        if model.train_mode == "create" and model.is_valid():
            n = precompute_templates(model)
            print(f"[PatMax Load] regenerated {n} DOF templates")
        return model
    except Exception as e:
        print(f"[PatMax] load_model error: {e}")
        return None
