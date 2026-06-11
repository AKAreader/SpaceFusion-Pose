# make_ab_crop_compare.py
import os
import re
import json
import csv
import math
import numpy as np
import cv2

# 假设这些工具函数在您的 utils_img.py 中
from utils_img import imread_any, to_float01, crop_with_roi

# ===================== 路径配置 =====================
DATA_ROOT = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
TEST_JSON = os.path.join(DATA_ROOT, "test_pose.json")

A_CSV = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports_A\pred_test_full.csv"
B_CSV = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports_B\pred_test_full.csv"

OUT_DIR = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\ab_crop_compare_global_legend_v3"
# ===================================================

TOPK = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ========= 视觉参数 =========
TEXT_Y0 = 44
TEXT_DY = 18

# inset 参数（每个 half 一块）
INSET_W = 160
INSET_H = 100
INSET_MARGIN_X = 10
INSET_MARGIN_Y = 95
INSET_ALPHA = 0

TRIAD_ORIGIN_IN_INSET = (100, 40)  # inset 内原点
TRIAD_LEN = 68  # 三轴长度
GT_THICK = 4
PR_THICK = 4
DASH_LEN = 10
GAP_LEN = 7

# 轴颜色（更亮，RGB）
X_COL = (40, 40, 255)  # BGR: Red
Y_COL = (40, 255, 40)  # BGR: Green
Z_COL = (255, 80, 80)  # BGR: Blue

# Pred 颜色（降低亮度以区分）
PRED_DIM = 0.55

# ===============================================
# [修改]：配色调整 (更柔和专业)
# ===============================================
COL_TXT = (230, 230, 230)  # 米白色文字
COL_BORDER = (120, 120, 120)  # 中灰色边框
COL_INSET_BG = (25, 25, 25)  # 深色背景

# 描边颜色 (配合亮色文字，使用深色描边增加对比)
OUTLINE_COL = (20, 20, 20)
OUTLINE_EXTRA = 2

# 全局 Legend（右下角，极简）
LEG_W = 180
LEG_H = 70  # 高度调小，更紧凑
LEG_ALPHA = 0.82
LEG_MARGIN = 18

# 顶部文本底板
TOPBOX_ALPHA = 0.70
TOPBOX_BG = (10, 10, 10)
TOPBOX_PAD = 3


# ===============================================


def _to_id_str(x):
    try:
        i = int(x)
        return f"{i:04d}"
    except Exception:
        s = str(x)
        return s.zfill(4) if s.isdigit() else s


def parse_summary(summary_path: str):
    cfg = {"out_size": None, "roi_expand": None}
    if not os.path.exists(summary_path):
        return cfg
    with open(summary_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def _get(key):
        m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", txt, flags=re.M)
        return m.group(1).strip() if m else None

    out_size = _get("out_size")
    roi_expand = _get("roi_expand")
    try:
        cfg["out_size"] = int(float(out_size)) if out_size is not None else None
    except Exception:
        cfg["out_size"] = None
    try:
        cfg["roi_expand"] = float(roi_expand) if roi_expand is not None else None
    except Exception:
        cfg["roi_expand"] = None
    return cfg


def parse_quat_str(s: str):
    if s is None:
        return None
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if len(nums) != 4:
        return None
    q = np.array([float(x) for x in nums], dtype=np.float32)
    q = q / (float(np.linalg.norm(q)) + 1e-12)
    if q[0] < 0:
        q = -q
    return q


def quat_to_euler_zyx_deg(q: np.ndarray):
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


def quat_to_rotmat_wxyz(q: np.ndarray):
    w, x, y, z = q.astype(np.float32).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ], dtype=np.float32)


# ======= 固定视角投影 =======
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


# =====================================================


def read_pred_csv(path: str):
    assert os.path.exists(path), f"CSV not found: {path}"
    out = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = _to_id_str(row.get("id"))
            out[sid] = {
                "angle_deg": float(row.get("angle_deg")),
                "q_gt": parse_quat_str(row.get("q_gt_wxyz")),
                "q_pred": parse_quat_str(row.get("q_pred_wxyz")),
            }
    return out


def imwrite(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode: {path}")
    buf.tofile(path)


def make_ir_crop(item: dict, out_size: int, roi_expand: float):
    roi = item["roi_xyxy"]
    ir_path = os.path.join(DATA_ROOT, item["ir_path"])
    ir = imread_any(ir_path, flag=cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir01 = to_float01(ir)
    crop = crop_with_roi(ir01, roi, expand=float(roi_expand), jitter=0.0, rng=np.random)
    crop = cv2.resize(crop, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA)
    return (crop * 255.0).clip(0, 255).astype(np.uint8)


def side_by_side(a_u8, b_u8, title_left, title_right):
    a = cv2.cvtColor(a_u8, cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(b_u8, cv2.COLOR_GRAY2BGR)
    H = max(a.shape[0], b.shape[0])
    W = a.shape[1] + b.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:a.shape[1] + b.shape[1]] = b

    # 标题文字也使用新配色
    cv2.putText(canvas, title_left, (10, 24), FONT, 0.72, COL_TXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, title_right, (a.shape[1] + 10, 24), FONT, 0.72, COL_TXT, 2, cv2.LINE_AA)
    return canvas


def draw_panel(img, x0, y0, w, h, fill_color, alpha, border_color=None, border_thick=1):
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), fill_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_color is not None:
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), border_color, border_thick)


def dim_color(bgr, k):
    return (int(bgr[0] * k), int(bgr[1] * k), int(bgr[2] * k))


def draw_arrow(img, p0, p1, color, thickness=2):
    cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.18)


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
        draw_arrow(img, p0, p1, OUTLINE_COL, t_out)
        draw_arrow(img, p0, p1, color, thickness)


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


def draw_global_legend_bottom_right(canvas):
    """
    右下角，极简 legend
    """
    H, W = canvas.shape[:2]
    x0 = W - LEG_W - LEG_MARGIN
    y0 = H - LEG_H - LEG_MARGIN

    draw_panel(canvas, x0, y0, LEG_W, LEG_H, fill_color=(10, 10, 10),
               alpha=LEG_ALPHA, border_color=COL_BORDER, border_thick=1)

    # 第一行：X Y Z 色块
    y = y0 + 22
    # X
    cv2.rectangle(canvas, (x0 + 10, y - 10), (x0 + 24, y + 4), X_COL, -1)
    cv2.putText(canvas, "X", (x0 + 28, y), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)
    # Y
    cv2.rectangle(canvas, (x0 + 56, y - 10), (x0 + 70, y + 4), Y_COL, -1)
    cv2.putText(canvas, "Y", (x0 + 74, y), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)
    # Z
    cv2.rectangle(canvas, (x0 + 102, y - 10), (x0 + 116, y + 4), Z_COL, -1)
    cv2.putText(canvas, "Z", (x0 + 120, y), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)

    # 第二行：GT / Pred 示例
    y2 = y0 + 50
    # GT 白实线 (这里用白色表示GT本身，但描边用了深色)
    cv2.line(canvas, (x0 + 10, y2), (x0 + 44, y2), (255, 255, 255), 3, cv2.LINE_AA)
    # ========================================================

    cv2.putText(canvas, "GT", (x0 + 50, y2 + 5), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)

    # Pred 黄虚线
    px0 = x0 + 96
    for k in range(4):
        cv2.line(canvas, (px0 + k * 10, y2), (px0 + k * 10 + 6, y2), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Pred", (px0 + 45, y2 + 5), FONT, 0.55, COL_TXT, 1, cv2.LINE_AA)


def draw_triad_in_inset(half_img_bgr, q_gt, q_pred, half_label="A"):
    x0 = INSET_MARGIN_X
    y0 = INSET_MARGIN_Y

    # [修改] border_color=None (去掉方框线)，只保留半透明底色区域逻辑(alpha=0时完全不显示)
    draw_panel(half_img_bgr, x0, y0, INSET_W, INSET_H, fill_color=COL_INSET_BG,
               alpha=INSET_ALPHA, border_color=None, border_thick=1)

    # [修改] 文字移到区域下方，避免与箭头重叠
    text_y_pos = y0 + INSET_H + 20
    cv2.putText(half_img_bgr, f"{half_label}: triad", (x0, text_y_pos),
                FONT, 0.60, COL_TXT, 1, cv2.LINE_AA)

    ox_in, oy_in = TRIAD_ORIGIN_IN_INSET
    origin = (x0 + ox_in, y0 + oy_in)
    cv2.circle(half_img_bgr, origin, 2, COL_TXT, -1, cv2.LINE_AA)

    gt_pts = triad_endpoints(q_gt, origin, TRIAD_LEN)
    pr_pts = triad_endpoints(q_pred, origin, TRIAD_LEN)

    if gt_pts is not None:
        px, py, pz = gt_pts
        draw_arrow_with_outline(half_img_bgr, origin, px, X_COL, GT_THICK, is_dashed=False)
        draw_arrow_with_outline(half_img_bgr, origin, py, Y_COL, GT_THICK, is_dashed=False)
        draw_arrow_with_outline(half_img_bgr, origin, pz, Z_COL, GT_THICK, is_dashed=False)

    if pr_pts is not None:
        px, py, pz = pr_pts
        draw_arrow_with_outline(half_img_bgr, origin, px, dim_color(X_COL, PRED_DIM), PR_THICK, is_dashed=True)
        draw_arrow_with_outline(half_img_bgr, origin, py, dim_color(Y_COL, PRED_DIM), PR_THICK, is_dashed=True)
        draw_arrow_with_outline(half_img_bgr, origin, pz, dim_color(Z_COL, PRED_DIM), PR_THICK, is_dashed=True)


def draw_top_text_box(canvas, lines, x=10, y=TEXT_Y0):
    """
    顶部文本
    """
    widths = []
    heights = []
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, FONT, 0.52, 1)
        widths.append(tw)
        heights.append(th)
    box_w = max(widths) + TOPBOX_PAD * 2
    box_h = len(lines) * TEXT_DY + TOPBOX_PAD * 2

    box_x0 = x - TOPBOX_PAD
    box_y0 = y - heights[0] - TOPBOX_PAD

    # 顶部文本框也使用新的边框色
    draw_panel(canvas, box_x0, box_y0, box_w, box_h, TOPBOX_BG, TOPBOX_ALPHA,
               border_color=COL_BORDER, border_thick=1)

    yy = y
    for ln in lines:
        cv2.putText(canvas, ln, (x, yy), FONT, 0.52, COL_TXT, 1, cv2.LINE_AA)
        yy += TEXT_DY


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.exists(TEST_JSON), f"test_pose.json not found: {TEST_JSON}"

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)
    id2item = {_to_id_str(it["id"]): it for it in samples}

    A = read_pred_csv(A_CSV)
    B = read_pred_csv(B_CSV)
    common_ids = sorted(set(A.keys()) & set(B.keys()) & set(id2item.keys()))
    assert len(common_ids) > 0, "No common ids among A/B CSV and test_pose.json"

    a_sum = parse_summary(os.path.join(os.path.dirname(A_CSV), "summary_test.txt"))
    b_sum = parse_summary(os.path.join(os.path.dirname(B_CSV), "summary_test.txt"))
    a_out = a_sum["out_size"] or 320
    a_ex = a_sum["roi_expand"] or 1.30
    b_out = b_sum["out_size"] or 320
    b_ex = b_sum["roi_expand"] or 1.30

    rows = []
    for sid in common_ids:
        errA = float(A[sid]["angle_deg"])
        errB = float(B[sid]["angle_deg"])
        rows.append((sid, errA, errB, errB - errA))
    rows.sort(key=lambda t: t[3], reverse=True)

    improved_ids = [t[0] for t in rows[:TOPK]]
    out_sub = os.path.join(OUT_DIR, "improved_top")
    os.makedirs(out_sub, exist_ok=True)

    for sid in improved_ids:
        item = id2item[sid]
        errA = float(A[sid]["angle_deg"])
        errB = float(B[sid]["angle_deg"])
        d = errB - errA

        qg = A[sid]["q_gt"] if A[sid]["q_gt"] is not None else B[sid]["q_gt"]
        qa = A[sid]["q_pred"]
        qb = B[sid]["q_pred"]

        yg, pg, rg = quat_to_euler_zyx_deg(qg) if qg is not None else (float("nan"),) * 3
        ya, pa, ra = quat_to_euler_zyx_deg(qa) if qa is not None else (float("nan"),) * 3
        yb, pb, rb = quat_to_euler_zyx_deg(qb) if qb is not None else (float("nan"),) * 3

        cropA = make_ir_crop(item, out_size=a_out, roi_expand=a_ex)
        cropB = make_ir_crop(item, out_size=b_out, roi_expand=b_ex)

        canvas = side_by_side(
            cropA, cropB,
            title_left=f"A (out={a_out}, ex={a_ex:.2f})",
            title_right=f"B (out={b_out}, ex={b_ex:.2f})"
        )

        draw_global_legend_bottom_right(canvas)

        wA = cropA.shape[1]
        left = canvas[:, :wA].copy()
        right = canvas[:, wA:wA + cropB.shape[1]].copy()

        draw_triad_in_inset(left, qg, qa, half_label="A")
        draw_triad_in_inset(right, qg, qb, half_label="B")

        canvas[:, :wA] = left
        canvas[:, wA:wA + cropB.shape[1]] = right

        lines = [
            f"GT zyx: [{yg:.1f},{pg:.1f},{rg:.1f}]",
            f"A  zyx: [{ya:.1f},{pa:.1f},{ra:.1f}]",
            f"B  zyx: [{yb:.1f},{pb:.1f},{rb:.1f}]",
        ]
        draw_top_text_box(canvas, lines, x=10, y=TEXT_Y0)

        out_path = os.path.join(out_sub, f"{sid}_A{errA:.3f}_B{errB:.3f}_d{d:.3f}.png")
        imwrite(out_path, canvas)

    print(f"[OK] exported {len(improved_ids)} improved cases to: {out_sub}")


if __name__ == "__main__":
    main()