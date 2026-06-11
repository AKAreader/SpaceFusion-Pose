# doctor_roi.py
import os
import json
import random
import numpy as np
import cv2

from config_pose import DATA_ROOT, TRAIN_JSON, OUT_SIZE, ROI_EXPAND_FIXED, ROI_JITTER_TRAIN
from roi_tools import cv_imread_cn, load_roi_cache, sample_square_from_roi

def draw_box(img_bgr, box, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "debug_rois")
    os.makedirs(out_dir, exist_ok=True)

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    cache = load_roi_cache(os.path.join(DATA_ROOT, "roi_cache_train.json"))
    rng = np.random.RandomState(42)

    pick = random.sample(samples, k=min(60, len(samples)))
    for s in pick:
        sid = int(s["id"])
        ir_path = os.path.join(DATA_ROOT, s["ir_path"])
        rgb_path = os.path.join(DATA_ROOT, s["rgb_path"])

        ir = cv_imread_cn(ir_path, cv2.IMREAD_GRAYSCALE)
        rgb = cv_imread_cn(rgb_path, cv2.IMREAD_COLOR)

        if ir is None or rgb is None:
            continue
        H, W = ir.shape[:2]

        base_json = s["roi_xyxy"]
        base_cache = cache.get(sid, None)

        # 训练采样框：用 cache 优先，否则 json
        base = base_cache if base_cache is not None else base_json
        sample_box = sample_square_from_roi(base, H, W, expand=ROI_EXPAND_FIXED, jitter=ROI_JITTER_TRAIN, rng=rng)

        # 画在 RGB 上
        vis = rgb.copy()
        draw_box(vis, base_json, (0, 255, 255), 2)  # 黄
        if base_cache is not None:
            draw_box(vis, base_cache, (0, 255, 0), 2)  # 绿
        draw_box(vis, sample_box, (0, 0, 255), 2)  # 红

        # 同时输出 crop 预览（模型看到的内容）
        x1, y1, x2, y2 = sample_box
        crop = rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

        fn = f"id{sid:04d}_{os.path.basename(s['ir_path'])}"
        cv2.imencode(".png", vis)[1].tofile(os.path.join(out_dir, "full_" + fn))
        cv2.imencode(".png", crop)[1].tofile(os.path.join(out_dir, "crop_" + fn))

    print(f"[OK] saved to: {out_dir}")

if __name__ == "__main__":
    main()
