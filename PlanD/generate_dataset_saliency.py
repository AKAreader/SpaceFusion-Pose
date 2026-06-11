# generate_saliency_all.py
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from saliency_gbmr import get_saliency_gbmr  # 确保这个文件还在

# === 修改路径 ===
IR_DIR = r"D:\BaiduNetdiskDownload\All\All_IR"
OUT_DIR = r"D:\BaiduNetdiskDownload\All\Saliency_Maps"


# ===============

# 辅助读取函数 (防止中文路径报错)
def cv_imread(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, -1)
        return img
    except:
        return None


def cv_imwrite(file_path, img):
    cv2.imencode('.png', img)[1].tofile(file_path)


os.makedirs(OUT_DIR, exist_ok=True)
img_list = glob(os.path.join(IR_DIR, "*.png"))

print(f"Generating Saliency Maps for {len(img_list)} images...")

for img_path in tqdm(img_list):
    img = cv_imread(img_path)
    if img is None: continue

    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 统一 resize 到 320 计算，加快速度
    img_resized = cv2.resize(img, (320, 320))

    try:
        saliency = get_saliency_gbmr(img_resized, n_segments=200)

        basename = os.path.basename(img_path)
        save_path = os.path.join(OUT_DIR, basename)
        cv_imwrite(save_path, saliency)
    except Exception as e:
        print(f"Error {img_path}: {e}")

print("Step 2 Done! Saliency Maps generated.")