import os
import json
import random
import numpy as np
import cv2

# ====== 你只需要改这里 ======
root = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
ann_path = os.path.join(root, "annotations_pose.json")
out_dir = os.path.join(root, "debug_roi_ir")
# ===========================

os.makedirs(out_dir, exist_ok=True)

# 读取标注
with open(ann_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

# 随机抽10条（不足10就全抽）
k = min(10, len(samples))
picked = random.sample(samples, k)

def imread_gray(path):
    # 支持中文路径的读取方式
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    return img

def imwrite(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode: {path}")
    buf.tofile(path)

for item in picked:
    ir_rel = item["ir_path"]
    x1, y1, x2, y2 = item["roi_xyxy"]
    sid = item["id"]

    ir_path = os.path.join(root, ir_rel)
    img = imread_gray(ir_path)

    # 转成BGR以便画彩色框
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 防御性裁剪（避免越界）
    H, W = img.shape
    x1 = max(0, min(W-1, int(x1)))
    x2 = max(0, min(W-1, int(x2)))
    y1 = max(0, min(H-1, int(y1)))
    y2 = max(0, min(H-1, int(y2)))

    # 画框与文字
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"id={sid}", (x1, max(0, y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out_path = os.path.join(out_dir, f"{sid}_ir_roi.png")
    imwrite(out_path, vis)

print(f"Done. Saved {k} images to: {out_dir}")
