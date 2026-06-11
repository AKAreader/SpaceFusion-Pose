# build_saliency_maps.py
# 离线生成伪显著图（IR为主，可选RGB联合），输出到 ROOT/Saliency_Maps/*.png
import os
import argparse
import math
import numpy as np
import cv2
from tqdm import tqdm


def cv_imread(path: str, flag: int):
    """兼容中文路径的OpenCV读图；失败返回None"""
    if not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flag)
        return img
    except Exception:
        return None


def cv_imsave(path: str, img: np.ndarray):
    """兼容中文路径的OpenCV写图"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
        ext = ".png"
        path = os.path.splitext(path)[0] + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(path)


def to01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def entropy01(gray01: np.ndarray) -> float:
    """灰度熵归一化到[0,1]（256 bins最大熵=8）"""
    g = np.clip((gray01 * 255.0).astype(np.uint8), 0, 255)
    hist = np.bincount(g.flatten(), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 1e-12]
    ent = float(-(p * np.log2(p)).sum())
    return float(np.clip(ent / 8.0, 0.0, 1.0))


def build_ir_saliency(ir01: np.ndarray) -> np.ndarray:
    """
    IR伪显著图：阈值+形态学+最大连通域+距离变换软化
    返回[0,1] float32
    """
    ir_u8 = np.clip(ir01 * 255.0, 0, 255).astype(np.uint8)

    # Otsu阈值（如果很暗会偏小，后面有兜底）
    _, bw = cv2.threshold(ir_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 兜底：若Otsu分割太少，改用低阈值
    if bw.mean() < 2:
        _, bw = cv2.threshold(ir_u8, 5, 255, cv2.THRESH_BINARY)

    # 形态学去噪/连通
    k = np.ones((9, 9), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    # 最大连通域
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # 全黑：返回一个很弱的中心高斯，避免全0导致门控失效
        h, w = ir01.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        sigma = 0.25 * min(h, w)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return to01(g)

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    # 距离变换->软mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist01 = to01(dist)

    # 与原IR强度融合一点（增强热目标区域）
    sal = 0.7 * dist01 + 0.3 * ir01
    sal = cv2.GaussianBlur(sal, (0, 0), 3)
    return np.clip(sal, 0.0, 1.0).astype(np.float32)


def build_vis_saliency(rgb01: np.ndarray) -> np.ndarray:
    """
    VIS伪显著图：梯度幅值（纹理/边缘）+平滑
    返回[0,1]
    """
    gray = (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 2)
    return to01(mag).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\BaiduNetdiskDownload\All", help="数据根目录")
    ap.add_argument("--use_rgb_joint", action="store_true",
                    help="使用联合显著性：S_joint = λ*S_vis + (1-λ)*S_ir（夜间自动偏向IR）")
    ap.add_argument("--invert_ir", action="store_true",
                    help="黑热模式时建议开启：IR先做 1-IR 再生成显著图")
    ap.add_argument("--ext", default=".png", help="图像后缀：.png/.jpg 等")
    args = ap.parse_args()

    root = args.root
    ir_dir = os.path.join(root, "All_IR")
    rgb_dir = os.path.join(root, "All_RGB")
    out_dir = os.path.join(root, "Saliency_Maps")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(ir_dir):
        raise FileNotFoundError(f"Not found: {ir_dir}")

    names = [n for n in os.listdir(ir_dir) if n.lower().endswith(args.ext.lower())]
    names.sort()

    for name in tqdm(names):
        ir_path = os.path.join(ir_dir, name)
        ir = cv_imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            continue
        ir01 = to01(ir)
        if args.invert_ir:
            ir01 = 1.0 - ir01

        s_ir = build_ir_saliency(ir01)

        if args.use_rgb_joint:
            rgb_path = os.path.join(rgb_dir, name)
            rgb = cv_imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                s = s_ir
            else:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb01 = to01(rgb)
                s_vis = build_vis_saliency(rgb01)

                # λ：用可见光“可用性”估计（用熵近似），夜间趋向0 -> 更多用IR
                gray = (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)
                lam = entropy01(gray)  # [0,1]
                # 稍微压低，让夜间更偏IR
                lam = float(np.clip((lam - 0.15) / 0.85, 0.0, 1.0))

                # 报告推荐的联合形式
                # S_joint = λ*S_vis + (1-λ)*S_ir
                s = lam * s_vis + (1.0 - lam) * s_ir
        else:
            s = s_ir

        out = np.clip(s * 255.0, 0, 255).astype(np.uint8)
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".png")
        cv_imsave(out_path, out)

    print(f"[OK] Saliency maps saved to: {out_dir}")


if __name__ == "__main__":
    main()
