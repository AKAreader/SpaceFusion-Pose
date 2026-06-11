import os
import re
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# ===================== 0) 配置区：你只需要改这里 =====================
RGB_DIR = r"E:\ALL\RGB"
IR_DIR  = r"E:\ALL\IR"

# 你的融合结果目录（要求：同名 png，例如 Aqua_45度__000059.png）
# 你可以把 Baseline/TrainA/TrainB/TrainC 的 fused_val 或你独立推理输出目录填进来
FUSED_DIRS = {
    "Baseline": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\baseline\20251228-234201\fused_val",
    "TrainA":   r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeA_v2\20260106-145605\fused_val",
    "TrainB":   r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeB_gate\20260106-170140\fused_val",
    "TrainC":   r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeC_v2_ssim\20260108-183933\fused_val",
}

# 输出根目录（脚本会自动创建子目录）
OUT_ROOT = r"./paper_figs_qualitative"

# 每个“叠放图”挑几张（建议 5~8）
N_STACK = 6

# 网格图：挑几个卫星类别（你说 5 类就写 5）
N_GROUPS = 5

# 图像统一尺寸（论文展示建议统一；你也可用 None 不缩放）
RESIZE_HW = (480, 640)  # (H, W) or None

# 随机种子（保证可复现）
SEED = 7

# ===================== 1) IO：支持中文路径的安全读取 =====================
def imread_unicode(path: str, flags: int):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img

def list_png(dir_path: str) -> List[str]:
    if not dir_path or not os.path.exists(dir_path):
        return []
    return [f for f in os.listdir(dir_path) if f.lower().endswith(".png")]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ===================== 2) 文件名匹配与分组（按卫星类别） =====================
def satellite_group(name: str) -> str:
    """
    你的文件名形如 Aqua_45度__000059.png / Clementine_10度__000208.png
    默认取第一个 '_' 之前作为卫星类别：Aqua / Clementine ...
    """
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    if "_" in stem:
        return stem.split("_")[0]
    # 兜底：取字母前缀
    m = re.match(r"([A-Za-z]+)", stem)
    return m.group(1) if m else "Unknown"

def intersect_common_names(rgb_dir, ir_dir, fused_dirs: Dict[str, str]) -> List[str]:
    rgb = set(list_png(rgb_dir))
    ir  = set(list_png(ir_dir))
    common = rgb & ir
    # 要求至少存在一个 fused 方法也有这张图（否则没法做对比）
    any_fused = set()
    for _, d in fused_dirs.items():
        any_fused |= set(list_png(d))
    common = sorted(list(common & any_fused))
    return common

def pick_one_per_group(names: List[str], k_groups: int, seed=0) -> List[str]:
    random.seed(seed)
    groups: Dict[str, List[str]] = {}
    for n in names:
        g = satellite_group(n)
        groups.setdefault(g, []).append(n)
    # 选 k 个 group
    all_g = sorted(groups.keys())
    if len(all_g) <= k_groups:
        chosen_g = all_g
    else:
        chosen_g = random.sample(all_g, k_groups)
        chosen_g = sorted(chosen_g)
    # 每个组选一张（可改成 random.choice / 固定第一张）
    picked = []
    for g in chosen_g:
        lst = sorted(groups[g])
        picked.append(lst[len(lst)//2])  # 取中间那张更稳定
    return picked

def pick_random(names: List[str], k: int, seed=0) -> List[str]:
    random.seed(seed)
    if len(names) <= k:
        return sorted(names)
    return sorted(random.sample(names, k))

# ===================== 3) 画图核心：叠放 stack / 网格 grid =====================
def to_rgb(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    if img_bgr_or_gray is None:
        return None
    if img_bgr_or_gray.ndim == 2:
        return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    # BGR -> RGB
    return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)

def resize_if(img: np.ndarray, resize_hw: Optional[Tuple[int, int]]):
    if img is None or resize_hw is None:
        return img
    h, w = resize_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def draw_label(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """在左上角画标签（RGB 格式）"""
    out = img_rgb.copy()
    # 转 BGR 画字再转回来，避免颜色错
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (8, 8), (8 + 12 * len(text) + 16, 42), (0, 0, 0), -1)
    cv2.putText(bgr, text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def save_png_unicode(path: str, img_rgb: np.ndarray):
    # RGB -> BGR
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(path)

def make_stack_figure(img_paths: List[Tuple[str, str]], out_path: str,
                      resize_hw=(480, 640), offset_px=18, alpha=0.72, title=None):
    """
    img_paths: [(label, path), ...] 同一模态的一组图
    叠放效果：每张图向右下偏移 offset_px，透明混合
    """
    # 读第一张确定尺寸
    imgs = []
    for lab, p in img_paths:
        if not os.path.exists(p):
            continue
        if p.lower().endswith(".png"):
            raw = imread_unicode(p, cv2.IMREAD_UNCHANGED)
        else:
            raw = imread_unicode(p, cv2.IMREAD_COLOR)
        # 兼容 4 通道
        if raw is None:
            continue
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]
        rgb = to_rgb(raw if raw.ndim == 2 else raw)
        rgb = resize_if(rgb, resize_hw)
        rgb = draw_label(rgb, lab)
        imgs.append(rgb)

    if len(imgs) == 0:
        print(f"[WARN] no images for stack: {out_path}")
        return

    H, W = imgs[0].shape[:2]
    n = len(imgs)
    canvas_h = H + offset_px * (n - 1)
    canvas_w = W + offset_px * (n - 1)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    # 从后往前叠（最后一张在最上层）
    for i, img in enumerate(imgs):
        dy = offset_px * i
        dx = offset_px * i
        patch = canvas[dy:dy+H, dx:dx+W, :]
        patch[:] = patch * (1 - alpha) + img.astype(np.float32) * alpha
        canvas[dy:dy+H, dx:dx+W, :] = patch

    out = np.clip(canvas, 0, 255).astype(np.uint8)

    # 可选：加标题条
    if title:
        bar_h = 52
        bar = np.zeros((bar_h, out.shape[1], 3), dtype=np.uint8)
        bgr = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, title, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        bar = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = np.vstack([bar, out])

    ensure_dir(os.path.dirname(out_path))
    save_png_unicode(out_path, out)
    print(f"[OK] stack saved: {out_path}")

def hstack_images(imgs: List[np.ndarray], gap=10) -> np.ndarray:
    """水平拼接（RGB），自动补齐高度"""
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    h = max(im.shape[0] for im in imgs)
    padded = []
    for im in imgs:
        hh, ww = im.shape[:2]
        if hh < h:
            pad = np.zeros((h - hh, ww, 3), dtype=im.dtype)
            im = np.vstack([im, pad])
        padded.append(im)
    gap_img = np.ones((h, gap, 3), dtype=np.uint8) * 255
    out = padded[0]
    for im in padded[1:]:
        out = np.hstack([out, gap_img, im])
    return out

def vstack_images(imgs: List[np.ndarray], gap=10) -> np.ndarray:
    """垂直拼接（RGB），自动补齐宽度"""
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return None
    w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        hh, ww = im.shape[:2]
        if ww < w:
            pad = np.zeros((hh, w - ww, 3), dtype=im.dtype)
            im = np.hstack([im, pad])
        padded.append(im)
    gap_img = np.ones((gap, w, 3), dtype=np.uint8) * 255
    out = padded[0]
    for im in padded[1:]:
        out = np.vstack([out, gap_img, im])
    return out

def load_rgb_ir_fused(name: str, fused_dirs: Dict[str, str], resize_hw) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    p_rgb = os.path.join(RGB_DIR, name)
    p_ir  = os.path.join(IR_DIR, name)
    rgb_raw = imread_unicode(p_rgb, cv2.IMREAD_COLOR)
    ir_raw  = imread_unicode(p_ir, cv2.IMREAD_GRAYSCALE)

    rgb = to_rgb(resize_if(rgb_raw, resize_hw))
    ir  = to_rgb(resize_if(ir_raw, resize_hw))
    if rgb is not None: rgb = draw_label(rgb, "RGB")
    if ir  is not None: ir  = draw_label(ir, "IR")

    fused = {}
    for m, d in fused_dirs.items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        f_raw = imread_unicode(p, cv2.IMREAD_UNCHANGED)
        if f_raw is None:
            continue
        if f_raw.ndim == 3 and f_raw.shape[2] == 4:
            f_raw = f_raw[:, :, :3]
        f = to_rgb(resize_if(f_raw, resize_hw))
        fused[m] = draw_label(f, m)
    return rgb, ir, fused

def make_grid_figure(picked_names: List[str], fused_dirs: Dict[str, str], out_path: str, resize_hw):
    """
    行：不同卫星类别挑的样本（picked_names）
    列：RGB / IR / 各方法 fused
    """
    rows = []
    for name in picked_names:
        rgb, ir, fused = load_rgb_ir_fused(name, fused_dirs, resize_hw)
        cols = [rgb, ir]
        # 按你定义的顺序输出
        for m in fused_dirs.keys():
            cols.append(fused.get(m, None))
        row = hstack_images(cols, gap=12)
        if row is None:
            continue

        # 每行左侧加一个文件名条（可删）
        bar_h = 40
        bar = np.zeros((bar_h, row.shape[1], 3), dtype=np.uint8)
        bgr = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, name, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        bar = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        row = np.vstack([bar, row])

        rows.append(row)

    grid = vstack_images(rows, gap=18)
    if grid is None:
        print(f"[WARN] grid empty: {out_path}")
        return
    ensure_dir(os.path.dirname(out_path))
    save_png_unicode(out_path, grid)
    print(f"[OK] grid saved: {out_path}")

# ===================== 4) 主流程 =====================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    out_stack = os.path.join(OUT_ROOT, "A_stacks")
    out_grid  = os.path.join(OUT_ROOT, "B_grids")
    ensure_dir(out_stack)
    ensure_dir(out_grid)

    common = intersect_common_names(RGB_DIR, IR_DIR, FUSED_DIRS)
    print(f"[INFO] common samples = {len(common)}")

    if len(common) == 0:
        print("[ERR] 没有找到 RGB/IR 与 fused 结果同名的样本。请确认 fused_val 里文件名与原图一致。")
        return

    # 1) 网格图：按卫星类别挑 5 张
    picked_by_group = pick_one_per_group(common, k_groups=N_GROUPS, seed=SEED)
    make_grid_figure(
        picked_names=picked_by_group,
        fused_dirs=FUSED_DIRS,
        out_path=os.path.join(out_grid, f"grid_{N_GROUPS}groups.png"),
        resize_hw=RESIZE_HW
    )

    # 2) 叠放图：RGB / IR 各随机挑 N_STACK 张
    picked_stack = pick_random(common, k=N_STACK, seed=SEED + 1)

    rgb_stack_list = [(os.path.splitext(n)[0], os.path.join(RGB_DIR, n)) for n in picked_stack]
    ir_stack_list  = [(os.path.splitext(n)[0], os.path.join(IR_DIR,  n)) for n in picked_stack]

    make_stack_figure(rgb_stack_list, os.path.join(out_stack, f"stack_RGB_{N_STACK}.png"),
                      resize_hw=RESIZE_HW, offset_px=18, alpha=0.72, title=f"RGB stack (N={N_STACK})")

    make_stack_figure(ir_stack_list, os.path.join(out_stack, f"stack_IR_{N_STACK}.png"),
                      resize_hw=RESIZE_HW, offset_px=18, alpha=0.72, title=f"IR stack (N={N_STACK})")

    # 3) 叠放图：每个方法各做一张 fused stack（同一批 picked_stack，保证可比）
    for m, d in FUSED_DIRS.items():
        fused_stack_list = [(os.path.splitext(n)[0], os.path.join(d, n)) for n in picked_stack]
        make_stack_figure(fused_stack_list, os.path.join(out_stack, f"stack_Fused_{m}_{N_STACK}.png"),
                          resize_hw=RESIZE_HW, offset_px=18, alpha=0.72, title=f"{m} fused stack (N={N_STACK})")

    print("\n[DONE] 输出目录：", os.path.abspath(OUT_ROOT))
    print("  - A_stacks/  叠放图（RGB/IR/各方法 fused）")
    print("  - B_grids/   同一组样本网格对比图（RGB/IR/各方法）")

if __name__ == "__main__":
    main()
