# roi_tools.py
import os
import json
import numpy as np
import cv2

def cv_imread_cn(path, flag=cv2.IMREAD_COLOR):
    """支持中文路径的 imread"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    return img

def clip_square_box_keep_size(cx, cy, side, W, H):
    """方形框贴边时不缩小：clamp 到 [0, W-side] / [0, H-side]"""
    side = float(side)
    side = max(2.0, min(side, float(min(W, H))))
    side_i = int(round(side))

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))

    if W - side_i <= 0:
        x1 = 0
        x2 = W
    else:
        x1 = max(0, min(W - side_i, x1))
        x2 = x1 + side_i

    if H - side_i <= 0:
        y1 = 0
        y2 = H
    else:
        y1 = max(0, min(H - side_i, y1))
        y2 = y1 + side_i

    x2 = max(x2, x1 + 2)
    y2 = max(y2, y1 + 2)
    return int(x1), int(y1), int(x2), int(y2)

def sample_square_from_roi(roi_xyxy, H, W, expand, jitter, rng):
    x1, y1, x2, y2 = [float(v) for v in roi_xyxy]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    side = max(bw, bh) * float(expand)
    if jitter > 0:
        dx = (rng.rand() * 2.0 - 1.0) * jitter * side
        dy = (rng.rand() * 2.0 - 1.0) * jitter * side
        cx += dx
        cy += dy

    return clip_square_box_keep_size(cx, cy, side, W, H)

def mask_to_bbox(mask_u8, min_area=200):
    """从二值 mask 找 bbox；过滤星点（小面积）"""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        return [int(x), int(y), int(x + w), int(y + h)]
    return None

def build_mask_from_saliency(sal, q=0.85):
    """sal: float32 [0,1]"""
    thr = float(np.quantile(sal, q))
    m = (sal >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)
    return m

def build_mask_from_ir(ir, thr=0.02):
    """ir: float32 [0,1]"""
    m = (ir >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)
    return m

def ensure_full_object(box, obj_mask_u8, W, H, max_iter=4, grow=1.18, border_tol=2):
    """
    如果 obj_mask 在 crop 内触边，说明可能截断：自动放大 box。
    box: [x1,y1,x2,y2] (方形更稳)
    """
    x1, y1, x2, y2 = box
    for _ in range(max_iter):
        crop = obj_mask_u8[y1:y2, x1:x2]
        if crop.size == 0:
            break
        # 检查触边
        top = crop[:border_tol, :].max() > 0
        bot = crop[-border_tol:, :].max() > 0
        lef = crop[:, :border_tol].max() > 0
        rig = crop[:, -border_tol:].max() > 0
        if not (top or bot or lef or rig):
            return [x1, y1, x2, y2]  # OK

        # 放大
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * grow
        x1, y1, x2, y2 = clip_square_box_keep_size(cx, cy, side, W, H)

    return [x1, y1, x2, y2]

def load_roi_cache(path):
    if path is None or (not os.path.exists(path)):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种格式：dict{id:box} 或 list of {"id":..,"roi":..}
    if isinstance(data, dict):
        return {int(k): v for k, v in data.items()}
    out = {}
    for it in data:
        out[int(it["id"])] = it["roi_xyxy"]
    return out
