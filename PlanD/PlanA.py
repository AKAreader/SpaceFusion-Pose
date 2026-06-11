# prepare_pose_dataset.py
import os, re, json, math
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# -----------------------------
# 0) 工具：找图像文件（支持多扩展名）
# -----------------------------
IMG_EXTS = [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]

def find_img(dirp, stem):
    for ext in IMG_EXTS:
        p = os.path.join(dirp, stem + ext)
        if os.path.exists(p):
            return p, ext
    return None, None

def check_rgb_ir_pairs(root):
    rgb_dir = os.path.join(root, "RGB")
    ir_dir  = os.path.join(root, "IR")
    stems = [f"{i:04d}" for i in range(1, 251)]
    miss = []
    for s in stems:
        r,_ = find_img(rgb_dir, s)
        i,_ = find_img(ir_dir,  s)
        if r is None or i is None:
            miss.append(s)
    print(f"[PAIR] expected 250, missing={len(miss)}")
    if miss:
        print("[PAIR] missing examples:", miss[:10])
    return len(miss)==0

# -----------------------------
# 1) 清洗CSV：选出与 1..250 对应的连续轨迹
# -----------------------------
def smooth_score(seq):
    pos = seq[["卫星位置_x(m)","卫星位置_y(m)","卫星位置_z(m)"]].to_numpy()
    q   = seq[["卫星四元数_w","卫星四元数_x","卫星四元数_y","卫星四元数_z"]].to_numpy()

    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()

    def quat_angle(q1, q2):
        q1 = np.array(q1, dtype=float); q2 = np.array(q2, dtype=float)
        q1 = q1 / np.linalg.norm(q1);   q2 = q2 / np.linalg.norm(q2)
        d  = abs(float(np.dot(q1, q2)))
        d  = min(1.0, max(-1.0, d))
        return 2.0 * math.acos(d)

    dq = sum(quat_angle(q[i], q[i+1]) for i in range(len(q)-1))
    return dp + 10.0 * dq

def clean_pose_csv(root, csv_name="Aqua_轨道倾斜60.0度.csv", tol=0.2):
    csv_path = os.path.join(root, csv_name)
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["时间戳(ISO)"], format="mixed")

    # 估计 t0, dt
    t0 = df[df["帧索引"]==1]["ts"].min()
    min_ts = df.groupby("帧索引")["ts"].min().sort_index()
    dt = float(np.median(min_ts.diff().dropna().dt.total_seconds().to_numpy()))
    print("[CSV] t0=", t0, "dt=", dt)

    expected = df["帧索引"].apply(lambda k: t0 + pd.to_timedelta((k-1)*dt, unit="s"))
    delta = (df["ts"] - expected).dt.total_seconds().abs()
    df_f = df[delta < tol].copy()

    print("[CSV] after time-filter rows:", len(df_f), "unique frames:", df_f["帧索引"].nunique())

    # 分支A/B：按 x 取最小/最大
    seqA=[]; seqB=[]
    for k,g in df_f.groupby("帧索引"):
        g2 = g.sort_values("卫星位置_x(m)")
        seqA.append(g2.iloc[0])
        seqB.append(g2.iloc[-1])
    seqA = pd.DataFrame(seqA).sort_values("帧索引")
    seqB = pd.DataFrame(seqB).sort_values("帧索引")

    scoreA = smooth_score(seqA)
    scoreB = smooth_score(seqB)
    print("[CSV] scoreA:", scoreA, "scoreB:", scoreB)

    pose_clean = seqA if scoreA <= scoreB else seqB
    pose_clean = pose_clean[pose_clean["帧索引"].between(1,250)].sort_values("帧索引").copy()

    out_path = os.path.join(root, "pose_clean.csv")
    pose_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[CSV] saved:", out_path, "rows:", len(pose_clean))
    return out_path

# -----------------------------
# 2) 生成ROI框：从IR自动找最大连通域外接矩形
# -----------------------------
def read_gray(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"read failed: {path}")
    return img

def find_roi_bbox(ir_gray, margin=1.3):
    H, W = ir_gray.shape
    img = cv2.GaussianBlur(ir_gray, (5,5), 0)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return (0,0,W-1,H-1)

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])

    cx = x + w/2; cy = y + h/2
    w2 = w * margin; h2 = h * margin
    x1 = int(max(0, cx - w2/2)); y1 = int(max(0, cy - h2/2))
    x2 = int(min(W-1, cx + w2/2)); y2 = int(min(H-1, cy + h2/2))
    return (x1,y1,x2,y2)

def make_roi_boxes(root, margin=1.3):
    ir_dir = os.path.join(root, "IR")
    records=[]
    for i in range(1, 251):
        stem = f"{i:04d}"
        p, _ = find_img(ir_dir, stem)
        if p is None:
            raise FileNotFoundError(stem)
        ir = read_gray(p)
        x1,y1,x2,y2 = find_roi_bbox(ir, margin=margin)
        records.append({"frame": i, "roi_x1": x1, "roi_y1": y1, "roi_x2": x2, "roi_y2": y2})

    out = os.path.join(root, "roi_boxes.csv")
    pd.DataFrame(records).to_csv(out, index=False, encoding="utf-8-sig")
    print("[ROI] saved:", out)
    return out

# -----------------------------
# 3) 生成最终 Pose-only JSON
# -----------------------------
def make_annotations_pose_json(root):
    pose_path = os.path.join(root, "pose_clean.csv")
    roi_path  = os.path.join(root, "roi_boxes.csv")
    pose = pd.read_csv(pose_path).set_index("帧索引")
    roi  = pd.read_csv(roi_path).set_index("frame")

    # 读尺寸 + 扩展名
    rgb_dir = os.path.join(root, "RGB")
    p0, ext = find_img(rgb_dir, "0001")
    W, H = Image.open(p0).size

    # 相机内参
    f_mm = float(pose.loc[1, "摄像机焦距(mm)"])
    sw   = float(pose.loc[1, "摄像机传感器宽度(mm)"])
    sh   = float(pose.loc[1, "摄像机传感器高度(mm)"])
    fx = f_mm / sw * W
    fy = f_mm / sh * H
    cx = W / 2
    cy = H / 2

    samples=[]
    for i in range(1, 251):
        stem = f"{i:04d}"
        _, ext = find_img(os.path.join(root,"RGB"), stem)
        r = pose.loc[i]
        b = roi.loc[i]
        samples.append({
            "id": stem,
            "rgb_path": f"RGB/{stem}{ext}",
            "ir_path":  f"IR/{stem}{ext}",
            "width": W, "height": H,
            "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "roi_xyxy": [int(b["roi_x1"]), int(b["roi_y1"]), int(b["roi_x2"]), int(b["roi_y2"])],
            "pose_quat_wxyz": [float(r["卫星四元数_w"]), float(r["卫星四元数_x"]), float(r["卫星四元数_y"]), float(r["卫星四元数_z"])],
            "position_xyz_m": [float(r["卫星位置_x(m)"]), float(r["卫星位置_y(m)"]), float(r["卫星位置_z(m)"])]
        })

    out_json = os.path.join(root, "annotations_pose.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print("[JSON] saved:", out_json, "num:", len(samples))
    return out_json

def main():
    root = r"D:\BaiduNetdiskDownload\data\Aqua_60度"

    assert check_rgb_ir_pairs(root), "RGB/IR not fully paired!"

    # 生成 pose_clean.csv（如果已经有且你确认没问题，可以注释掉）
    clean_pose_csv(root, csv_name="Aqua_轨道倾斜60.0度.csv", tol=0.2)

    # 生成 roi_boxes.csv
    make_roi_boxes(root, margin=1.3)

    # 生成 annotations_pose.json
    make_annotations_pose_json(root)

if __name__ == "__main__":
    main()
