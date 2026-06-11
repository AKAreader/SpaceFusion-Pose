# make_qualitative_grid.py
import os
import re
import json
from collections import OrderedDict, defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===================== 配置区域 =====================
ALGOS = OrderedDict({
    "DenseFuse": r"E:\Other\dense",
    "Pixel": r"E:\Other\pixel",
    "Pyramid": r"E:\Other\pyramid",
    "RFN": r"E:\Other\rfn",
    "SeAFusion": r"E:\Other\seafusion",
    "SoPD-Net": r"E:\Other\SoPD-Net",
})

REF_ALGO_NAME = "SoPD-Net"
OUT_DIR = r".\qualitative_report"
OUT_PNG = "qualitative_grid_tight.png"
OUT_JSON = "selected_samples.json"

PICK_STRATEGY = "first"
RECURSIVE = True
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ===================================================

def list_images(folder: str, recursive: bool = True):
    files = []
    if recursive:
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                if f.lower().endswith(IMG_EXTS):
                    files.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.join(folder, f))
    files.sort()
    return files


def safe_imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return img


def to_rgb(img):
    if img is None: return None
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_satellite_name(filename: str) -> str:
    """
    从文件名解析“卫星类别”
    优化逻辑：去除 '_45度' 或中文后缀，只保留纯净的卫星名
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]

    # 1. 先去掉 "__" 及后面的编号 (e.g. "Aqua_45度__001" -> "Aqua_45度")
    if "__" in stem:
        stem = stem.split("__")[0]

    # 2. 如果包含 "度" (中文)，强制去掉 "度" 及其前面的数字/下划线
    # 策略：按 "_" 分割，通常卫星名在第一段
    # 例如 "Clementine_10度" -> "Clementine"
    # "NOAA 20_45度" -> "NOAA 20"
    if "_" in stem:
        parts = stem.split("_")
        # 如果分割后的第一部分看起来像卫星名，就直接返回第一部分
        return parts[0]

    # 3. 兜底正则 (保留原逻辑作为备用)
    m = re.match(r"^([A-Za-z0-9\s]+)", stem)
    if m:
        return m.group(1).strip()

    return stem


def pick_one(lst, strategy="first"):
    if not lst: return None
    if strategy == "random": return lst[np.random.randint(0, len(lst))]
    if strategy == "middle": return lst[len(lst) // 2]
    return lst[0]


def build_stem_index(img_paths):
    idx = {}
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        idx[stem] = p
    return idx


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 准备数据
    if REF_ALGO_NAME not in ALGOS: raise ValueError(f"REF not found: {REF_ALGO_NAME}")
    ref_imgs = list_images(ALGOS[REF_ALGO_NAME], RECURSIVE)
    if not ref_imgs: raise RuntimeError("Reference folder empty")

    groups = defaultdict(list)
    for p in ref_imgs:
        groups[parse_satellite_name(p)].append(p)

    sats_top5 = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)[:5]

    selected = OrderedDict()
    # 临时变量用于计算尺寸
    sample_img_path = None

    for sat in sats_top5:
        path = pick_one(groups[sat], PICK_STRATEGY)
        selected[sat] = os.path.basename(path)
        if sample_img_path is None:
            sample_img_path = path

    with open(os.path.join(OUT_DIR, OUT_JSON), "w", encoding="utf-8") as f:
        json.dump({"selected": selected}, f, ensure_ascii=False, indent=2)

    algo_stem_index = {}
    for k, v in ALGOS.items():
        algo_stem_index[k] = build_stem_index(list_images(v, RECURSIVE))

    # ================= 关键修改：动态计算画布尺寸 =================

    # 1. 读取一张样本图，获取宽高比 (Height / Width)
    img_h, img_w = 480, 640  # 默认值防止读取失败
    if sample_img_path and os.path.exists(sample_img_path):
        tmp = safe_imread_unicode(sample_img_path)
        if tmp is not None:
            img_h, img_w = tmp.shape[:2]

    img_aspect_ratio = img_h / img_w  # 高宽比，例如 0.75

    # 2. 设定布局参数
    algo_names = list(ALGOS.keys())
    sat_names = list(selected.keys())
    n_rows = len(algo_names)
    n_cols_imgs = len(sat_names)

    # 定义“文字栏”相对宽度。假设文字栏宽度相当于 0.4 张图片宽
    TEXT_COL_RATIO = 0.4

    # 3. 设定一个固定的总宽度（单位：英寸），比如 16 英寸宽
    TOTAL_FIG_WIDTH = 16.0

    # 计算每一份（即一张图片）在画布上的实际宽度
    # 总宽度 = (文字栏宽度) + (N张图片宽度)
    # Total_W = unit_w * TEXT_COL_RATIO + unit_w * N
    unit_w = TOTAL_FIG_WIDTH / (TEXT_COL_RATIO + n_cols_imgs)

    # 计算对应的图片高度，保持比例
    unit_h = unit_w * img_aspect_ratio

    # 计算总高度 = 行数 * 每行高度 + 顶部标题的一点余量
    # 顶部标题大概预留 0.3 英寸
    TITLE_PAD = 0.3
    TOTAL_FIG_HEIGHT = (n_rows * unit_h) + TITLE_PAD

    print(f"检测到图片尺寸: {img_w}x{img_h} (Ratio: {img_aspect_ratio:.2f})")
    print(f"计算画布尺寸: {TOTAL_FIG_WIDTH:.1f} x {TOTAL_FIG_HEIGHT:.1f} 英寸")

    # 4. 创建画布
    fig = plt.figure(figsize=(TOTAL_FIG_WIDTH, TOTAL_FIG_HEIGHT), dpi=150)

    # GridSpec 布局
    # top参数用于给卫星名字留出空间
    # hspace=0, wspace=0 彻底消除间隙
    gs = gridspec.GridSpec(n_rows, n_cols_imgs + 1,
                           width_ratios=[TEXT_COL_RATIO] + [1.0] * n_cols_imgs,
                           wspace=0.0,
                           hspace=0.0,
                           top=1.0 - (TITLE_PAD / TOTAL_FIG_HEIGHT),
                           bottom=0, left=0, right=1)

    for i, algo in enumerate(algo_names):
        # --- 第一列：算法名 ---
        ax_txt = fig.add_subplot(gs[i, 0])
        ax_txt.axis("off")
        ax_txt.text(0.5, 0.5, algo, fontsize=12, fontweight="bold", ha="center", va="center")

        # --- 后续列：图片 ---
        idx_map = algo_stem_index[algo]

        for j, sat in enumerate(sat_names):
            ax_img = fig.add_subplot(gs[i, j + 1])

            # 只有第一行显示标题
            if i == 0:
                ax_img.set_title(sat, fontsize=12, pad=8)

            ax_img.axis("off")  # 关掉坐标轴，再次减少干扰

            # 找图
            fname = selected[sat]
            stem = os.path.splitext(fname)[0]
            p = idx_map.get(stem, None)
            if not p:
                cand = os.path.join(ALGOS[algo], fname)
                if os.path.exists(cand): p = cand

            if p and os.path.exists(p):
                im = safe_imread_unicode(p)
                rgb = to_rgb(im)
                if rgb is not None:
                    # aspect='auto' 会强制拉伸填满格子，防止微小空隙，
                    # 但如果图片尺寸本身一致，aspect='equal' (默认) 更好。
                    # 这里为了严丝合缝，我们信任前面的尺寸计算，保持默认即可。
                    ax_img.imshow(rgb)
                else:
                    ax_img.text(0.5, 0.5, "Error", ha="center")
            else:
                ax_img.text(0.5, 0.5, "Missing", ha="center")

    out_path = os.path.join(OUT_DIR, OUT_PNG)
    # 保存时不加 bbox_inches='tight'，因为我们已经精确计算了 coordinates
    # 如果加了 tight，可能会破坏我们精心计算的宽高比导致又出现缝隙
    # 但为了保险去掉周围白边，可以用 pad_inches=0
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()