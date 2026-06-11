import os, json
import numpy as np

ROOT = r"D:\BaiduNetdiskDownload\All"

def main():
    path = os.path.join(ROOT, "train_pose.json")
    data = json.load(open(path, "r", encoding="utf-8"))

    qs = np.array([d["pose_quat_wxyz"] for d in data], np.float32)
    qn = np.linalg.norm(qs, axis=1)
    print("[quat] norm: mean/min/max =", qn.mean(), qn.min(), qn.max())

    t = np.array([d["pose_t"] for d in data], np.float32)
    print("[t] z: min/max =", t[:,2].min(), t[:,2].max(), " neg% =", float((t[:,2] <= 0).mean()))

    roi = np.array([d["roi_xyxy"] for d in data], np.float32)
    w = roi[:,2] - roi[:,0]
    h = roi[:,3] - roi[:,1]
    print("[roi] w/h min/median/max =", w.min(), np.median(w), w.max(), "/", h.min(), np.median(h), h.max())

if __name__ == "__main__":
    main()
