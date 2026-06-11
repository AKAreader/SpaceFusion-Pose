# make_v3_center_triad.py  (v3.2)
# Fixes vs v3.1:
# 1) RGB-only CSV: allow rows with ONLY q_pred (no q_gt), so C panel can work even if gt not exported
# 2) More robust id / angle column name candidates
# 3) If C has no angle_deg, compute using q_gt from A at runtime
#
# Output: crop-only qualitative figures (A=IR, B=RGBIR, optional C=RGB-only)

import os
import csv
import json
import math
import ast
import random
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np


# ===================== USER CONFIG =====================
DATA_ROOT = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
TEST_JSON = os.path.join(DATA_ROOT, "test_pose.json")

DIR_IR_A = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports\ir_out320_ex1.30_best_ir"
DIR_RGBIR_B = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports\rgbir_out320_ex1.30_best_rgbir"
DIR_RGB_C = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA_RGB\test_exports_rgb"  # optional

OUT_DIR = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA_RGB\paper_figs_v3_center_triad_v32"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SIZE = 320
ROI_EXPAND = 1.30

TOPK_WORST = 10
TOPK_DELTA = 10
RANDOM_N = 10

USE_RGB_BG_FOR_B = True
ENABLE_RGB_ONLY_PANEL = True

# Triad placement & size
TRIAD_LEN_RATIO = 0.22
TRIAD_CENTER_OFFSET_XY = (0, -60)  # move upward more; tune -40/-60/-80
GT_THICK = 4
PR_THICK = 4
DASH_LEN = 10
GAP_LEN = 7
# =======================================================


# ===================== Visual Style =====================
FONT = cv2.FONT_HERSHEY_SIMPLEX
X_COL = (40, 40, 255)
Y_COL = (40, 255, 40)
Z_COL = (255, 80, 80)
PRED_DIM = 0.55

COL_TXT = (230, 230, 230)
COL_BORDER = (120, 120, 120)
OUTLINE_COL = (20, 20, 20)
OUTLINE_EXTRA = 2

LEG_W = 220
LEG_H = 78
LEG_ALPHA = 0.82
LEG_MARGIN = 18

TOPBOX_ALPHA = 0.70
TOPBOX_BG = (10, 10, 10)
TOPBOX_PAD = 4
TEXT_DY = 18
TEXT_Y0 = 44
# =======================================================


# ===================== IO Helpers =====================
def imread_any(path: str, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def imwrite_any(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode: {path}")
    buf.tofile(path)

def to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        m = float(np.max(img)) if np.max(img) > 0 else 65535.0
        return img.astype(np.float32) / m
    img = img.astype(np.float32)
    m = float(np.max(img)) if np.max(img) > 1.5 else 1.0
    if m > 1.0:
        img = img / m
    return np.clip(img, 0.0, 1.0)

def _clip_box(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(1, min(W, x2)))
    y2 = int(max(1, min(H, y2)))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def crop_with_roi(img: np.ndarray, roi_xyxy, expand: float):
    x1, y1, x2, y2 = [float(v) for v in roi_xyxy]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw2 = bw * float(expand)
    bh2 = bh * float(expand)

    nx1 = cx - bw2 / 2.0
    ny1 = cy - bh2 / 2.0
    nx2 = cx + bw2 / 2.0
    ny2 = cy + bh2 / 2.0

    H, W = img.shape[:2]
    x1i, y1i, x2i, y2i = _clip_box(nx1, ny1, nx2, ny2, W=W, H=H)
    if img.ndim == 2:
        return img[y1i:y2i, x1i:x2i]
    return img[y1i:y2i, x1i:x2i, :]

def resize_square(img: np.ndarray, out_size: int) -> np.ndarray:
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

def _basename_noext(s: str) -> str:
    s = str(s).strip()
    s = os.path.basename(s)
    if "." in s:
        s = os.path.splitext(s)[0]
    return s

def to_id_str(x) -> str:
    s = _basename_noext(str(x))
    if s.isdigit():
        return s.zfill(4)
    digits = "".join([c for c in s if c.isdigit()])
    if digits.isdigit() and len(digits) > 0:
        return digits.zfill(4)
    return s

def parse_list_str(s) -> List[float]:
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(v) for v in s]
    if s is None:
        return []
    s = str(s).strip()

    # handle "tensor([...])"
    if s.lower().startswith("tensor("):
        s = s[s.find("(") + 1: s.rfind(")")]
        s = s.strip()

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
    except Exception:
        pass

    s = s.strip("[]() ")
    s = s.replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out
# =======================================================


# ===================== Pose Math =====================
def quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    n = float(np.linalg.norm(q) + 1e-12)
    q = q / n
    if q[0] < 0:
        q = -q
    return q

def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.astype(np.float32).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)]
    ], dtype=np.float32)

def quat_to_euler_zyx_deg(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = q.tolist()
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * x + y * z)
    t4 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t3, t4)

    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))

def angle_deg_from_two_quat(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
    q_pred = quat_wxyz_normalize(q_pred)
    q_gt = quat_wxyz_normalize(q_gt)
    dot = float(abs(np.dot(q_pred, q_gt)))
    dot = max(-1.0, min(1.0, dot))
    ang = 2.0 * math.degrees(math.acos(dot))
    return ang
# =======================================================


# ===================== Projection (fixed view) =====================
_VIEW_DIR = np.array([1.0, 1.0, 0.9], dtype=np.float32)
_VIEW_DIR = _VIEW_DIR / (np.linalg.norm(_VIEW_DIR) + 1e-12)
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
_RIGHT = np.cross(_WORLD_UP, _VIEW_DIR)
if np.linalg.norm(_RIGHT) < 1e-6:
    _WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    _RIGHT = np.cross(_WORLD_UP, _VIEW_DIR)
_RIGHT = _RIGHT / (np.linalg.norm(_RIGHT) + 1e-12)
_UP = np.cross(_VIEW_DIR, _RIGHT)
_UP = _UP / (np.linalg.norm(_UP) + 1e-12)

def project_vec(v3: np.ndarray):
    u = float(np.dot(v3, _RIGHT))
    v = float(np.dot(v3, _UP))
    return np.array([u, v], dtype=np.float32)
# =======================================================


# ===================== CSV / JSON loaders =====================
def find_pred_csv(pred_dir: str) -> str:
    cands = []
    for fn in os.listdir(pred_dir):
        if fn.lower().endswith(".csv") and fn.lower().startswith("pred_test_full"):
            cands.append(os.path.join(pred_dir, fn))
    if not cands:
        cands = [os.path.join(pred_dir, fn) for fn in os.listdir(pred_dir) if fn.lower().endswith(".csv")]
    if not cands:
        raise FileNotFoundError(f"No csv found in: {pred_dir}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def load_test_items(test_json: str) -> Dict[str, dict]:
    with open(test_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    mp = {}
    for it in items:
        sid = to_id_str(it["id"])
        mp[sid] = it
    return mp

def _first_existing_key(row: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return k
    return None

def _extract_quat(row: dict, prefix: str) -> Optional[np.ndarray]:
    k1 = _first_existing_key(row, [f"{prefix}_wxyz", f"{prefix}_quat_wxyz", f"{prefix}_quat", prefix])
    if k1 is not None:
        arr = parse_list_str(row.get(k1))
        if len(arr) == 4:
            return quat_wxyz_normalize(np.array(arr, dtype=np.float32))

    kw = _first_existing_key(row, [f"{prefix}_w", f"{prefix}w", f"{prefix}_0"])
    kx = _first_existing_key(row, [f"{prefix}_x", f"{prefix}x", f"{prefix}_1"])
    ky = _first_existing_key(row, [f"{prefix}_y", f"{prefix}y", f"{prefix}_2"])
    kz = _first_existing_key(row, [f"{prefix}_z", f"{prefix}z", f"{prefix}_3"])
    if kw and kx and ky and kz:
        try:
            arr = [float(row[kw]), float(row[kx]), float(row[ky]), float(row[kz])]
            return quat_wxyz_normalize(np.array(arr, dtype=np.float32))
        except Exception:
            return None
    return None

def _extract_angle(row: dict) -> Optional[float]:
    k = _first_existing_key(row, [
        "angle_deg", "ang_deg", "err_deg", "rot_err_deg", "error_deg",
        "rot_angle_deg", "rotation_error_deg"
    ])
    if k is None:
        return None
    try:
        return float(row[k])
    except Exception:
        return None

def _extract_id(row: dict) -> Optional[str]:
    k = _first_existing_key(row, [
        "id", "sid", "frame_id", "name",
        "index", "idx", "frame", "img", "image", "filename", "file"
    ])
    if k is None:
        return None
    return to_id_str(row[k])

def load_preds(csv_path: str, allow_missing_qgt: bool = False) -> Dict[str, dict]:
    """
    allow_missing_qgt=False: require q_gt and q_pred (A/B)
    allow_missing_qgt=True : allow only q_pred (C panel)
    """
    out: Dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        cols = [c.strip() for c in (r.fieldnames or [])]
        if not cols:
            raise RuntimeError(f"Empty CSV header: {csv_path}")

        print(f"[DBG] CSV header({os.path.basename(csv_path)}): {cols}")

        for row in r:
            sid = _extract_id(row)
            if sid is None:
                continue

            qg = _extract_quat(row, "q_gt")
            qp = _extract_quat(row, "q_pred")

            if qg is None:
                qg = _extract_quat(row, "gt")
            if qp is None:
                qp = _extract_quat(row, "pred")

            if qp is None:
                continue
            if (not allow_missing_qgt) and (qg is None):
                continue

            ang = _extract_angle(row)  # may be None for RGB-only
            eg = list(quat_to_euler_zyx_deg(qg)) if qg is not None else None
            ep = list(quat_to_euler_zyx_deg(qp))

            out[sid] = {
                "id": sid,
                "angle_deg": float(ang) if ang is not None else None,
                "q_gt": qg,
                "q_pred": qp,
                "e_gt": eg,
                "e_pred": ep,
            }

    print(f"[DBG] parsed preds: {len(out)} from {csv_path}")
    return out
# =======================================================


# ===================== Drawing =====================
def draw_panel(img, x0, y0, w, h, fill_color, alpha, border_color=None, border_thick=1):
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), fill_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_color is not None:
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), border_color, border_thick)

def draw_top_text_box(canvas, lines, x=10, y=TEXT_Y0):
    widths = []
    heights = []
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, FONT, 0.55, 1)
        widths.append(tw)
        heights.append(th)
    box_w = max(widths) + TOPBOX_PAD * 2
    box_h = len(lines) * TEXT_DY + TOPBOX_PAD * 2

    box_x0 = x - TOPBOX_PAD
    box_y0 = y - heights[0] - TOPBOX_PAD

    draw_panel(canvas, box_x0, box_y0, box_w, box_h, TOPBOX_BG, TOPBOX_ALPHA,
               border_color=COL_BORDER, border_thick=1)

    yy = y
    for ln in lines:
        cv2.putText(canvas, ln, (x, yy), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)
        yy += TEXT_DY

def dim_color(bgr, k):
    return (int(bgr[0] * k), int(bgr[1] * k), int(bgr[2] * k))

def draw_dashed_arrow(img, p0, p1, color, thickness=2, dash_len=10, gap_len=7):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    dist = float(math.hypot(dx, dy))
    if dist < 1e-6:
        return
    vx = dx / dist
    vy = dy / dist
    cur = 0.0
    while cur < dist:
        seg_start = cur
        seg_end = min(dist, cur + dash_len)
        sx = int(round(x0 + vx * seg_start))
        sy = int(round(y0 + vy * seg_start))
        ex = int(round(x0 + vx * seg_end))
        ey = int(round(y0 + vy * seg_end))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        cur += dash_len + gap_len
    tail = (int(round(x1 - vx * 12)), int(round(y1 - vy * 12)))
    cv2.arrowedLine(img, tail, (x1, y1), color, thickness, cv2.LINE_AA, tipLength=0.35)

def draw_arrow_with_outline(img, p0, p1, color, thickness, is_dashed=False):
    t_out = thickness + OUTLINE_EXTRA
    if is_dashed:
        draw_dashed_arrow(img, p0, p1, OUTLINE_COL, t_out, DASH_LEN, GAP_LEN)
        draw_dashed_arrow(img, p0, p1, color, thickness, DASH_LEN, GAP_LEN)
    else:
        cv2.arrowedLine(img, p0, p1, OUTLINE_COL, t_out, cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.18)

def triad_endpoints(q: np.ndarray, origin_xy, length_px: int):
    ox, oy = origin_xy
    if q is None:
        return None
    R = quat_to_rotmat_wxyz(q)
    ex = R @ np.array([1, 0, 0], dtype=np.float32)
    ey = R @ np.array([0, 1, 0], dtype=np.float32)
    ez = R @ np.array([0, 0, 1], dtype=np.float32)
    px = project_vec(ex)
    py = project_vec(ey)
    pz = project_vec(ez)

    def to_pt(p2):
        u, v = float(p2[0]), float(p2[1])
        return (int(round(ox + u * length_px)), int(round(oy - v * length_px)))

    return to_pt(px), to_pt(py), to_pt(pz)

def draw_center_triad(img_bgr: np.ndarray, q_gt: np.ndarray, q_pred: np.ndarray):
    H, W = img_bgr.shape[:2]
    dx, dy = TRIAD_CENTER_OFFSET_XY
    origin = (int(W // 2 + dx), int(H // 2 + dy))

    triad_len = int(TRIAD_LEN_RATIO * min(H, W))
    triad_len = max(30, min(triad_len, int(0.45 * min(H, W))))

    cv2.circle(img_bgr, origin, 2, COL_TXT, -1, cv2.LINE_AA)

    gt_pts = triad_endpoints(q_gt, origin, triad_len)
    pr_pts = triad_endpoints(q_pred, origin, triad_len)

    if gt_pts is not None:
        px, py, pz = gt_pts
        draw_arrow_with_outline(img_bgr, origin, px, X_COL, GT_THICK, is_dashed=False)
        draw_arrow_with_outline(img_bgr, origin, py, Y_COL, GT_THICK, is_dashed=False)
        draw_arrow_with_outline(img_bgr, origin, pz, Z_COL, GT_THICK, is_dashed=False)

    if pr_pts is not None:
        px, py, pz = pr_pts
        draw_arrow_with_outline(img_bgr, origin, px, dim_color(X_COL, PRED_DIM), PR_THICK, is_dashed=True)
        draw_arrow_with_outline(img_bgr, origin, py, dim_color(Y_COL, PRED_DIM), PR_THICK, is_dashed=True)
        draw_arrow_with_outline(img_bgr, origin, pz, dim_color(Z_COL, PRED_DIM), PR_THICK, is_dashed=True)

def draw_global_legend_bottom_right(canvas: np.ndarray):
    H, W = canvas.shape[:2]
    x0 = W - LEG_W - LEG_MARGIN
    y0 = H - LEG_H - LEG_MARGIN

    draw_panel(canvas, x0, y0, LEG_W, LEG_H, fill_color=(10, 10, 10),
               alpha=LEG_ALPHA, border_color=COL_BORDER, border_thick=1)

    y = y0 + 24
    cv2.rectangle(canvas, (x0 + 10, y - 10), (x0 + 24, y + 4), X_COL, -1)
    cv2.putText(canvas, "X", (x0 + 30, y), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (x0 + 58, y - 10), (x0 + 72, y + 4), Y_COL, -1)
    cv2.putText(canvas, "Y", (x0 + 78, y), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (x0 + 106, y - 10), (x0 + 120, y + 4), Z_COL, -1)
    cv2.putText(canvas, "Z", (x0 + 126, y), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)

    y2 = y0 + 54
    cv2.line(canvas, (x0 + 10, y2), (x0 + 44, y2), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (x0 + 50, y2 + 5), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)

    px0 = x0 + 96
    for k in range(4):
        cv2.line(canvas, (px0 + k * 10, y2), (px0 + k * 10 + 6, y2), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Pred", (px0 + 45, y2 + 5), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)
# =======================================================


# ===================== Image prep =====================
def load_crop_images(item: dict) -> Tuple[np.ndarray, np.ndarray]:
    roi = item["roi_xyxy"]

    ir_path = os.path.join(DATA_ROOT, item["ir_path"])
    ir = imread_any(ir_path, flag=cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir01 = to_float01(ir)
    ir_crop = crop_with_roi(ir01, roi, expand=ROI_EXPAND)
    ir_crop = resize_square(ir_crop, OUT_SIZE)
    ir_u8 = (ir_crop * 255.0).clip(0, 255).astype(np.uint8)
    ir_bgr = cv2.cvtColor(ir_u8, cv2.COLOR_GRAY2BGR)

    rgb_path = os.path.join(DATA_ROOT, item["rgb_path"])
    rgb = imread_any(rgb_path, flag=cv2.IMREAD_COLOR)
    rgb01 = to_float01(rgb)
    rgb_crop = crop_with_roi(rgb01, roi, expand=ROI_EXPAND)
    rgb_crop = resize_square(rgb_crop, OUT_SIZE)
    rgb_u8 = (rgb_crop * 255.0).clip(0, 255).astype(np.uint8)

    return ir_bgr, rgb_u8
# =======================================================


def make_panel(base_img_bgr: np.ndarray, q_gt: np.ndarray, q_pred: np.ndarray, title: str) -> np.ndarray:
    img = base_img_bgr.copy()
    cv2.putText(img, title, (10, 26), FONT, 0.72, COL_TXT, 2, cv2.LINE_AA)
    draw_center_triad(img, q_gt, q_pred)
    return img

def concat_panels(panels: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(panels, axis=1)


def main():
    items = load_test_items(TEST_JSON)
    print(f"[DBG] test_json items: {len(items)}")

    csv_ir = find_pred_csv(DIR_IR_A)
    csv_rgbir = find_pred_csv(DIR_RGBIR_B)
    print(f"[DBG] csv_ir   : {csv_ir}")
    print(f"[DBG] csv_rgbir: {csv_rgbir}")

    pred_ir = load_preds(csv_ir, allow_missing_qgt=False)
    pred_rgbir = load_preds(csv_rgbir, allow_missing_qgt=False)

    pred_rgb = {}
    csv_rgb = None
    if ENABLE_RGB_ONLY_PANEL and os.path.isdir(DIR_RGB_C):
        try:
            csv_rgb = find_pred_csv(DIR_RGB_C)
            print(f"[DBG] csv_rgb  : {csv_rgb}")
            pred_rgb = load_preds(csv_rgb, allow_missing_qgt=True)  # <<< key change
        except Exception as e:
            print(f"[WARN] RGB-only cannot load, continue A/B only. err={e}")
            pred_rgb = {}

    common_ab = set(items.keys()) & set(pred_ir.keys()) & set(pred_rgbir.keys())
    if len(common_ab) == 0:
        raise RuntimeError("No common ids among IR/RGBIR predictions and test_json.")

    common_ids = sorted(list(common_ab))
    common_abc = (set(common_ids) & set(pred_rgb.keys())) if pred_rgb else set()

    print(f"[DBG] common(A,B): {len(common_ids)}")
    print(f"[DBG] common(A,B,C): {len(common_abc)}")

    worst = sorted(common_ids, key=lambda sid: pred_rgbir[sid]["angle_deg"], reverse=True)[:min(TOPK_WORST, len(common_ids))]
    deltas = [(sid, pred_rgbir[sid]["angle_deg"] - pred_ir[sid]["angle_deg"]) for sid in common_ids]
    A_much_better = [sid for sid, _d in sorted(deltas, key=lambda x: x[1], reverse=True)[:min(TOPK_DELTA, len(deltas))]]
    B_much_better = [sid for sid, _d in sorted(deltas, key=lambda x: x[1])[:min(TOPK_DELTA, len(deltas))]]

    random.seed(42)
    rand_ids = random.sample(common_ids, k=min(RANDOM_N, len(common_ids)))

    packs = [
        ("worst_by_B", worst),
        ("A_much_better", A_much_better),
        ("B_much_better", B_much_better),
        ("random", rand_ids),
    ]

    for tag, id_list in packs:
        out_sub = os.path.join(OUT_DIR, tag)
        os.makedirs(out_sub, exist_ok=True)

        for sid in id_list:
            it = items[sid]
            ir_crop_bgr, rgb_crop_bgr = load_crop_images(it)

            # A/B required
            q_gt = pred_ir[sid]["q_gt"]
            q_a = pred_ir[sid]["q_pred"]
            q_b = pred_rgbir[sid]["q_pred"]
            a_err = float(pred_ir[sid]["angle_deg"])
            b_err = float(pred_rgbir[sid]["angle_deg"])
            zyx_gt = pred_ir[sid]["e_gt"]
            zyx_a = pred_ir[sid]["e_pred"]
            zyx_b = pred_rgbir[sid]["e_pred"]

            panel_a_bg = ir_crop_bgr
            panel_b_bg = rgb_crop_bgr if USE_RGB_BG_FOR_B else ir_crop_bgr

            panel_a = make_panel(panel_a_bg, q_gt, q_a, title="A: IR-only")
            panel_b = make_panel(panel_b_bg, q_gt, q_b, title="B: RGBIR")
            panels = [panel_a, panel_b]

            # C optional
            c_ok = bool(pred_rgb) and (sid in pred_rgb) and (pred_rgb[sid].get("q_pred") is not None)
            c_err = None
            zyx_c = None
            q_c = None

            if ENABLE_RGB_ONLY_PANEL and c_ok:
                q_c = pred_rgb[sid]["q_pred"]
                # if csv has no angle_deg, compute on the fly using A's q_gt
                if pred_rgb[sid].get("angle_deg") is None:
                    c_err = angle_deg_from_two_quat(q_c, q_gt)
                else:
                    c_err = float(pred_rgb[sid]["angle_deg"])
                zyx_c = list(quat_to_euler_zyx_deg(q_c))
                panel_c = make_panel(rgb_crop_bgr, q_gt, q_c, title="C: RGB-only")
                panels.append(panel_c)

            big = concat_panels(panels)

            top_pad = 68
            canvas = np.zeros((big.shape[0] + top_pad, big.shape[1], 3), dtype=np.uint8)
            canvas[top_pad:, :, :] = big

            # concise header (no "??")
            if c_ok:
                header1 = f"A={a_err:.3f}deg  B={b_err:.3f}deg  C={c_err:.3f}deg"
                header2 = ("ZYX(deg)  GT=[{:.1f},{:.1f},{:.1f}]  "
                           "A=[{:.1f},{:.1f},{:.1f}]  "
                           "B=[{:.1f},{:.1f},{:.1f}]  "
                           "C=[{:.1f},{:.1f},{:.1f}]").format(
                    zyx_gt[0], zyx_gt[1], zyx_gt[2],
                    zyx_a[0], zyx_a[1], zyx_a[2],
                    zyx_b[0], zyx_b[1], zyx_b[2],
                    zyx_c[0], zyx_c[1], zyx_c[2],
                )
            else:
                if ENABLE_RGB_ONLY_PANEL and csv_rgb is not None:
                    print(f"[WARN] sid={sid} not found/parsed in RGB-only preds. csv={csv_rgb}")
                header1 = f"A_ERR={a_err:.3f}°   B_ERR={b_err:.3f}°   C_ERR=N/A"
                header2 = ("ZYX(deg)  GT=[{:.1f},{:.1f},{:.1f}]  "
                           "A=[{:.1f},{:.1f},{:.1f}]  "
                           "B=[{:.1f},{:.1f},{:.1f}]").format(
                    zyx_gt[0], zyx_gt[1], zyx_gt[2],
                    zyx_a[0], zyx_a[1], zyx_a[2],
                    zyx_b[0], zyx_b[1], zyx_b[2],
                )

            draw_top_text_box(canvas, [header1, header2], x=10, y=TEXT_Y0)
            draw_global_legend_bottom_right(canvas)

            out_path = os.path.join(out_sub, f"{sid}_A{a_err:.3f}_B{b_err:.3f}.png")
            imwrite_any(out_path, canvas)

        print(f"[OK] {tag}: saved {len(id_list)} -> {out_sub}")

    print(f"[DONE] All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
