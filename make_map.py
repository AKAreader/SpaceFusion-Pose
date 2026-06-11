import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# ===================== 你只需要改这里 =====================
RGB_DIR = r"E:\ALL\RGB"
IR_DIR  = r"E:\ALL\IR"

# 各算法的“融合结果图”所在文件夹（fused_val 或你保存融合图的目录）
ALGOS = OrderedDict({
    "Baseline": r"E:\Other\seafusion",
    "TrainC":   r"E:\Other\SoPD-Net",
})

# 你之前挑的“5类卫星各一张”的记录（可选）。没有就设为 None，自动从 IR_DIR 里找5张
SELECT_JSON = r".\qualitative_report\selected_samples.json"  # 没有就改成 None

OUT_DIR = r".\residual_report"
USE_HEATMAP = True  # True: 彩色热力图；False: 灰度图
# =========================================================


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def safe_imread(path, flag):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)

def to_gray(img):
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def norm_0_255(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    return (y * 255.0).clip(0, 255).astype(np.uint8)

def colorize(x_gray_0_255):
    if not USE_HEATMAP:
        return x_gray_0_255
    return cv2.applyColorMap(x_gray_0_255, cv2.COLORMAP_JET)

def pick_samples():
    # 1) 优先用你已经挑好的 5 类样本
    if SELECT_JSON and os.path.exists(SELECT_JSON):
        with open(SELECT_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        # d["selected"] 是 sat -> filename
        return list(d["selected"].values())

    # 2) 没有 json：从 IR_DIR 自动找 5 张（按文件名排序取前5）
    files = [f for f in os.listdir(IR_DIR) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files[:5]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = pick_samples()
    print("[INFO] samples =", samples)

    for fname in samples:
        p_ir  = os.path.join(IR_DIR, fname)
        p_rgb = os.path.join(RGB_DIR, fname)

        ir  = safe_imread(p_ir,  cv2.IMREAD_GRAYSCALE)
        vis = safe_imread(p_rgb, cv2.IMREAD_COLOR)
        if ir is None or vis is None:
            print("[WARN] read fail:", fname)
            continue

        vis_g = to_gray(vis)

        # 每张样本输出一个“对比大图”：每行算法；每行展示 Fused + |F-IR| + |F-VIS|
        n_rows = len(ALGOS)
        fig, axes = plt.subplots(n_rows, 3, figsize=(10, 2.8*n_rows), dpi=150)

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, (algo, fused_dir) in enumerate(ALGOS.items()):
            p_f = os.path.join(fused_dir, fname)
            if not os.path.exists(p_f):
                axes[i, 0].axis("off"); axes[i, 1].axis("off"); axes[i, 2].axis("off")
                axes[i, 0].text(0.5, 0.5, f"{algo}\nMISSING", ha="center", va="center")
                continue

            fused = safe_imread(p_f, cv2.IMREAD_UNCHANGED)
            if fused is None:
                axes[i, 0].axis("off"); axes[i, 1].axis("off"); axes[i, 2].axis("off")
                axes[i, 0].text(0.5, 0.5, f"{algo}\nREAD_FAIL", ha="center", va="center")
                continue

            # 对齐尺寸：以 fused 的尺寸为准
            if fused.ndim == 2:
                H, W = fused.shape
            else:
                H, W = fused.shape[:2]
            ir_r  = cv2.resize(ir, (W, H), interpolation=cv2.INTER_LINEAR)
            vis_r = cv2.resize(vis_g, (W, H), interpolation=cv2.INTER_LINEAR)

            fused_g = to_gray(fused)
            if fused_g is None:
                fused_g = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)

            d_ir  = cv2.absdiff(fused_g, ir_r)
            d_vis = cv2.absdiff(fused_g, vis_r)

            d_ir_n  = norm_0_255(d_ir)
            d_vis_n = norm_0_255(d_vis)

            d_ir_show  = colorize(d_ir_n)
            d_vis_show = colorize(d_vis_n)

            # 保存单独的残差图（可用于论文插图）
            out_single_dir = os.path.join(OUT_DIR, "single_maps", algo)
            os.makedirs(out_single_dir, exist_ok=True)
            cv2.imencode(".png", d_ir_show)[1].tofile(os.path.join(out_single_dir, fname.replace(".png", "_absF_minus_IR.png")))
            cv2.imencode(".png", d_vis_show)[1].tofile(os.path.join(out_single_dir, fname.replace(".png", "_absF_minus_VIS.png")))

            # 画行：Fused / |F-IR| / |F-VIS|
            axes[i, 0].imshow(cv2.cvtColor(fused if fused.ndim==3 else cv2.cvtColor(fused, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f"{algo} - Fused", fontsize=11)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cv2.cvtColor(d_ir_show, cv2.COLOR_BGR2RGB) if USE_HEATMAP else d_ir_show, cmap=None if USE_HEATMAP else "gray")
            axes[i, 1].set_title("|F - IR|  (dark=similar)", fontsize=11)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(cv2.cvtColor(d_vis_show, cv2.COLOR_BGR2RGB) if USE_HEATMAP else d_vis_show, cmap=None if USE_HEATMAP else "gray")
            axes[i, 2].set_title("|F - VIS| (dark=similar)", fontsize=11)
            axes[i, 2].axis("off")

        plt.tight_layout()
        out_big = os.path.join(OUT_DIR, fname.replace(".png", "_residual_grid.png"))
        plt.savefig(out_big, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("[OK] saved:", out_big)

if __name__ == "__main__":
    main()
