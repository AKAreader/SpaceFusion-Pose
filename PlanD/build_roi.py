# build_roi_cache.py
import os
import json
import numpy as np
import cv2
from tqdm import tqdm

from config_pose import DATA_ROOT, TRAIN_JSON, VAL_JSON, TEST_JSON
from roi_tools import (
    cv_imread_cn, build_mask_from_saliency, build_mask_from_ir,
    mask_to_bbox, clip_square_box_keep_size, ensure_full_object
)

def build_one(split_json_path, out_path):
    with open(split_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    sal_dir = os.path.join(DATA_ROOT, "Saliency_Maps")
    cache = {}

    for s in tqdm(samples, desc=os.path.basename(out_path)):
        sid = int(s["id"])
        ir_path = os.path.join(DATA_ROOT, s["ir_path"])
        fname = os.path.basename(s["ir_path"])
        sal_path = os.path.join(sal_dir, fname)

        ir_u8 = cv_imread_cn(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir_u8 is None:
            continue
        H, W = ir_u8.shape[:2]
        ir = (ir_u8.astype(np.float32) / 255.0)

        obj_mask = None
        if os.path.exists(sal_path):
            sal_u8 = cv_imread_cn(sal_path, cv2.IMREAD_GRAYSCALE)
            if sal_u8 is not None:
                if sal_u8.shape[:2] != (H, W):
                    sal_u8 = cv2.resize(sal_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                sal = sal_u8.astype(np.float32) / 255.0
                obj_mask = build_mask_from_saliency(sal, q=0.85)

        if obj_mask is None:
            obj_mask = build_mask_from_ir(ir, thr=0.02)

        bbox = mask_to_bbox(obj_mask, min_area=200)
        if bbox is None:
            # fallback：用 json roi，但方形化
            x1, y1, x2, y2 = s["roi_xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = max(x2 - x1, y2 - y1) * 1.35
            box = list(clip_square_box_keep_size(cx, cy, side, W, H))
            cache[sid] = box
            continue

        # 方形化 + 外扩（保守）
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * 1.35
        box = list(clip_square_box_keep_size(cx, cy, side, W, H))

        # 防截断兜底：触边就自动放大
        box = ensure_full_object(box, obj_mask, W, H, max_iter=4, grow=1.18, border_tol=2)

        cache[sid] = box

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"[OK] saved: {out_path}  (n={len(cache)})")

def main():
    build_one(TRAIN_JSON, os.path.join(DATA_ROOT, "roi_cache_train.json"))
    build_one(VAL_JSON,   os.path.join(DATA_ROOT, "roi_cache_val.json"))
    build_one(TEST_JSON,  os.path.join(DATA_ROOT, "roi_cache_test.json"))

if __name__ == "__main__":
    main()
