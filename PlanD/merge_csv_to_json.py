import os
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm


CFG = {
    "data_root": r"D:\BaiduNetdiskDownload\All",
    "saliency_dir_name": "Saliency_Maps",
    "splits": ["train", "val", "test"],

    # 只有当 json ROI 很大（接近整幅图）时才用 saliency 精修
    "roi_big_ratio": 0.60,

    # 关键：把 saliency 的框做得“保守”，宁可大一点也不要截断
    "pad_ratio": 0.45,

    # 防截断：如果目标 mask 贴边，就自动扩大框
    "ensure_full_object": True,
    "border_margin": 10,      # mask bbox 距离 crop 边界 < 10px 就认为可能被截断
    "max_fix_iters": 4,
    "fix_expand_step": 1.25,  # 每次扩大 25%
    "max_box_area_ratio": 0.80,  # 避免扩到几乎全图

    # IR mask 参数（用于检测是否截断）
    "mask_quantile": 0.92,
    "mask_blur": 2.0,

    # saliency 阈值（更松一点，避免只取到“卫星亮边”）
    "sal_seed_q": 0.90,
    "sal_grow_list": (0.82, 0.78, 0.74, 0.70),
}


def cv_imread(path: str, flag: int):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)


def to01(img_u8: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def clip_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 2, x1)))
    y1 = int(max(0, min(H - 2, y1)))
    x2 = int(max(x1 + 2, min(W, x2)))
    y2 = int(max(y1 + 2, min(H, y2)))
    return x1, y1, x2, y2


def box_area(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def pad_box(box: Tuple[int, int, int, int], W: int, H: int, pad_ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = int(pad_ratio * max(bw, bh))
    return clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)


def expand_box(box: Tuple[int, int, int, int], W: int, H: int, factor: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw *= factor
    bh *= factor
    return clip_box(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0, W, H)


def ir_mask_bbox(ir_crop_u8: np.ndarray, q: float, blur_sigma: float) -> Optional[Tuple[int, int, int, int]]:
    if ir_crop_u8 is None or ir_crop_u8.size == 0:
        return None
    ir = ir_crop_u8.astype(np.float32)
    if blur_sigma > 0:
        ir = cv2.GaussianBlur(ir, (0, 0), blur_sigma)

    thr = float(np.quantile(ir.reshape(-1), q))
    m = (ir >= thr).astype(np.uint8)

    ksz = max(5, int(min(m.shape[0], m.shape[1]) * 0.03))
    if ksz % 2 == 0:
        ksz += 1
    k = np.ones((ksz, ksz), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)

    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if (x2 - x1) * (y2 - y1) < 40:
        return None
    return x1, y1, x2, y2


def touches_border(b: Tuple[int, int, int, int], W: int, H: int, margin: int) -> bool:
    x1, y1, x2, y2 = b
    return (x1 <= margin) or (y1 <= margin) or (W - x2 <= margin) or (H - y2 <= margin)


def roi_from_saliency(sal01: np.ndarray, seed_q: float, grow_list) -> Optional[Tuple[int, int, int, int]]:
    H, W = sal01.shape[:2]
    if sal01.max() <= 1e-6:
        return None

    sal_blur = cv2.GaussianBlur(sal01.astype(np.float32), (0, 0), 2.0)

    t_seed = float(np.quantile(sal_blur.reshape(-1), seed_q))
    seed_mask = (sal_blur >= t_seed).astype(np.uint8)

    ksz = max(7, int(min(H, W) * 0.02))
    if ksz % 2 == 0:
        ksz += 1
    k = np.ones((ksz, ksz), np.uint8)

    candidates = []
    for qg in grow_list:
        tg = float(np.quantile(sal_blur.reshape(-1), qg))
        low = (sal_blur >= tg).astype(np.uint8)
        low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, k, iterations=2)
        low = cv2.dilate(low, k, iterations=2)

        num, lab = cv2.connectedComponents(low, connectivity=8)
        if num <= 1:
            continue

        for cid in range(1, num):
            comp = (lab == cid)
            if (comp & (seed_mask > 0)).sum() == 0:
                continue
            ys, xs = np.where(comp)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            area = (x2 - x1) * (y2 - y1)
            if area < 60:
                continue

            energy = float((sal_blur[y1:y2, x1:x2] * comp[y1:y2, x1:x2]).sum())
            candidates.append((energy, (x1, y1, x2, y2)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def make_safe_box(ir_u8: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    H, W = ir_u8.shape[:2]
    full_area = H * W

    cur = box
    for _ in range(int(CFG["max_fix_iters"])):
        x1, y1, x2, y2 = cur
        ir_crop = ir_u8[y1:y2, x1:x2]
        bb = ir_mask_bbox(ir_crop, float(CFG["mask_quantile"]), float(CFG["mask_blur"]))
        if bb is None:
            break

        bx1, by1, bx2, by2 = bb
        ch, cw = ir_crop.shape[:2]
        if touches_border((bx1, by1, bx2, by2), cw, ch, int(CFG["border_margin"])):
            cur = expand_box(cur, W, H, float(CFG["fix_expand_step"]))
            if box_area(cur) / float(full_area + 1e-6) > float(CFG["max_box_area_ratio"]):
                break
        else:
            break
    return cur


def main():
    root = CFG["data_root"]
    sal_dir = os.path.join(root, CFG["saliency_dir_name"])
    assert os.path.exists(root), f"data_root not found: {root}"
    assert os.path.exists(sal_dir), f"saliency dir not found: {sal_dir}"

    for split in CFG["splits"]:
        json_path = os.path.join(root, f"{split}_pose.json")
        if not os.path.exists(json_path):
            print(f"[SKIP] {json_path} not found")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        out_cache_path = os.path.join(root, f"roi_cache_{split}.json")
        cache: Dict[str, Any] = {}

        big_cnt = 0
        sal_used = 0
        fixed_cnt = 0

        for it in tqdm(items, desc=f"build roi cache [{split}]"):
            sid = str(int(it["id"]))

            ir_path = os.path.join(root, it["ir_path"])
            ir_u8 = cv_imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir_u8 is None:
                continue
            H, W = ir_u8.shape[:2]
            full_area = H * W

            x1, y1, x2, y2 = [float(v) for v in it["roi_xyxy"]]
            json_box = clip_box(x1, y1, x2, y2, W, H)
            json_ratio = box_area(json_box) / float(full_area + 1e-6)

            base_box = json_box
            source = "json"

            if json_ratio > float(CFG["roi_big_ratio"]):
                big_cnt += 1
                sal_path = os.path.join(sal_dir, os.path.basename(ir_path))
                sal_u8 = cv_imread(sal_path, cv2.IMREAD_GRAYSCALE)

                if sal_u8 is not None:
                    est = roi_from_saliency(to01(sal_u8), float(CFG["sal_seed_q"]), CFG["sal_grow_list"])
                else:
                    est = None

                if est is not None:
                    est = pad_box(est, W, H, float(CFG["pad_ratio"]))
                    base_box = est
                    source = "saliency"
                    sal_used += 1

            if bool(CFG["ensure_full_object"]):
                safe = make_safe_box(ir_u8, base_box)
                if safe != base_box:
                    fixed_cnt += 1
                base_box = safe

            cache[sid] = {
                "base_box": [int(base_box[0]), int(base_box[1]), int(base_box[2]), int(base_box[3])],
                "source": source,
                "W": int(W),
                "H": int(H),
            }

        with open(out_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"[OK] wrote: {out_cache_path}")
        print(f"[STAT] big_json_roi={big_cnt}/{len(items)}  sal_used={sal_used}  fixed_by_ir={fixed_cnt}")


if __name__ == "__main__":
    main()
