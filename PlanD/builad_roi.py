# build_roi_cache.py
# 作用：一次性用 Saliency_Maps 估计 ROI，并写入 roi_cache_{train/val/test}.json
# 用法：PyCharm 直接 Run（不需要命令行）

import os
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm


CFG = {
    "data_root": r"D:\BaiduNetdiskDownload\All",
    "saliency_dir_name": "Saliency_Maps",
    "splits": ["train", "val", "test"],  # 会读取 train_pose.json / val_pose.json / test_pose.json
    "roi_big_ratio": 0.60,               # JSON ROI 过大才用 saliency ROI
    "center_fallback_size": 480,         # saliency 不可用时中心框边长
    "pad_ratio": 0.18,                   # 给 saliency ROI 加 padding，防止只框到“头部”
}


def cv_imread(path: str, flag: int):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)


def to01_u8(img_u8: np.ndarray) -> np.ndarray:
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


def roi_from_saliency(
    sal01: np.ndarray,
    q_seed: float = 0.93,
    q_grow_list=(0.85, 0.80, 0.75, 0.70, 0.65),
    max_area_ratio: float = 0.55,
    min_area_ratio: float = 0.00025,
    min_side_ratio: float = 0.006,
    blur_sigma: float = 2.0,
    center_bias: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    H, W = sal01.shape[:2]
    full_area = H * W
    max_area = int(max_area_ratio * full_area)
    min_area = int(max(120, min_area_ratio * full_area))
    min_side = int(max(8, min_side_ratio * min(H, W)))

    if sal01.max() <= 1e-6:
        return None

    sal_blur = cv2.GaussianBlur(sal01.astype(np.float32), (0, 0), blur_sigma)

    t_seed = float(np.quantile(sal_blur.reshape(-1), q_seed))
    seed_mask = (sal_blur >= t_seed).astype(np.uint8)

    ksz = max(7, int(min(H, W) * 0.015))
    if ksz % 2 == 0:
        ksz += 1
    k = np.ones((ksz, ksz), np.uint8)

    cx0, cy0 = W / 2.0, H / 2.0
    diag = (W * W + H * H) ** 0.5

    candidates = []
    for q_grow in q_grow_list:
        t_grow = float(np.quantile(sal_blur.reshape(-1), q_grow))
        low = (sal_blur >= t_grow).astype(np.uint8)

        low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, k, iterations=2)
        low = cv2.dilate(low, k, iterations=1)

        num, lab = cv2.connectedComponents(low, connectivity=8)
        if num <= 1:
            continue

        for comp_id in range(1, num):
            comp = (lab == comp_id)
            if (comp & (seed_mask > 0)).sum() == 0:
                continue

            ys, xs = np.where(comp)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            bw, bh = x2 - x1, y2 - y1
            area = bw * bh

            if area < min_area or area > max_area:
                continue
            if min(bw, bh) < min_side:
                continue

            patch = sal_blur[y1:y2, x1:x2]
            comp_patch = comp[y1:y2, x1:x2]
            energy = float((patch * comp_patch).sum())

            bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = (((bx - cx0) ** 2 + (by - cy0) ** 2) ** 0.5) / (diag + 1e-6)
            center_factor = float(np.exp(-(dist ** 2) / (2 * (0.35 ** 2))))
            score = energy * (area ** 0.15) * ((1.0 - center_bias) + center_bias * center_factor)

            candidates.append((score, area, (x1, y1, x2, y2)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_area, best_box = candidates[0]

    # 防“亮星点”极小框：太小则在 top-k 中选面积最大的
    area_ratio = best_area / float(full_area + 1e-6)
    if area_ratio < 0.0015 and len(candidates) > 1:
        topk = candidates[:min(8, len(candidates))]
        best_box = max(topk, key=lambda x: x[1])[2]

    return best_box


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
        fallback_cnt = 0

        for it in tqdm(items, desc=f"build roi cache [{split}]"):
            sid = str(int(it["id"]))

            # JSON roi
            x1, y1, x2, y2 = [float(v) for v in it["roi_xyxy"]]

            # 读 IR 只为拿 H/W（避免依赖固定分辨率）
            ir_path = os.path.join(root, it["ir_path"])
            ir = cv_imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is None:
                continue
            H, W = ir.shape[:2]

            json_box = clip_box(x1, y1, x2, y2, W, H)
            full_area = H * W
            json_ratio = box_area(json_box) / float(full_area + 1e-6)

            base_box = json_box
            source = "json"

            if json_ratio > float(CFG["roi_big_ratio"]):
                big_cnt += 1
                # 用 saliency 估计 ROI
                sal_path = os.path.join(sal_dir, os.path.basename(ir_path))
                sal_u8 = cv_imread(sal_path, cv2.IMREAD_GRAYSCALE)
                if sal_u8 is not None:
                    sal01 = to01_u8(sal_u8)
                    est = roi_from_saliency(sal01)
                else:
                    est = None

                if est is not None:
                    est = pad_box(est, W, H, float(CFG["pad_ratio"]))
                    # 若仍很小，回退中心框
                    if box_area(est) / float(full_area + 1e-6) < 0.001:
                        cx, cy = W / 2.0, H / 2.0
                        s = float(CFG["center_fallback_size"]) / 2.0
                        est = clip_box(cx - s, cy - s, cx + s, cy + s, W, H)
                        fallback_cnt += 1
                        source = "center_fallback"
                    else:
                        source = "saliency"
                        sal_used += 1
                    base_box = est
                else:
                    cx, cy = W / 2.0, H / 2.0
                    s = float(CFG["center_fallback_size"]) / 2.0
                    base_box = clip_box(cx - s, cy - s, cx + s, cy + s, W, H)
                    fallback_cnt += 1
                    source = "center_fallback"

            cache[sid] = {
                "base_box": [int(base_box[0]), int(base_box[1]), int(base_box[2]), int(base_box[3])],
                "source": source,
                "W": int(W),
                "H": int(H),
            }

        with open(out_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"[OK] wrote: {out_cache_path}")
        print(f"[STAT] big_json_roi={big_cnt}/{len(items)}  sal_used={sal_used}  fallback={fallback_cnt}")


if __name__ == "__main__":
    main()
