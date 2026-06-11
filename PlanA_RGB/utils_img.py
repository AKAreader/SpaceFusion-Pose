# utils_img.py
import os
import numpy as np
import cv2

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def find_img(dirp: str, stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(dirp, stem + ext)
        if os.path.exists(p):
            return p
    return None

def imread_any(path: str, flag=cv2.IMREAD_UNCHANGED):
    # 支持中文路径
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def to_float01(img: np.ndarray) -> np.ndarray:
    # 兼容 8-bit / 16-bit
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    # 其他类型：按最大值归一化（防御）
    img = img.astype(np.float32)
    mx = float(np.max(img)) if np.max(img) > 0 else 1.0
    return img / mx

def crop_with_roi(img: np.ndarray, roi_xyxy, expand=1.30, jitter=0.10, rng=None):
    """
    img: HxW or HxWxC
    roi_xyxy: [x1,y1,x2,y2] (inclusive or exclusive都可，这里当作像素坐标边界)
    expand: 先按比例扩大
    jitter: 再做随机平移/缩放扰动（不改变姿态标签，属于视角裁剪扰动）
    """
    if rng is None:
        rng = np.random

    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in roi_xyxy]
    # 规范化：确保 x1<x2, y1<y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(2.0, (x2 - x1))
    bh = max(2.0, (y2 - y1))

    # expand
    bw *= expand
    bh *= expand

    # jitter (scale + shift)
    s = 1.0 + float(rng.uniform(-jitter, jitter))
    bw *= s
    bh *= s
    cx += float(rng.uniform(-jitter, jitter)) * bw
    cy += float(rng.uniform(-jitter, jitter)) * bh

    nx1 = int(max(0, cx - bw / 2.0))
    ny1 = int(max(0, cy - bh / 2.0))
    nx2 = int(min(W - 1, cx + bw / 2.0))
    ny2 = int(min(H - 1, cy + bh / 2.0))

    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        # 退化：返回整图
        return img

    return img[ny1:ny2, nx1:nx2]
