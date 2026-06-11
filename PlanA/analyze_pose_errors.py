# analyze_pose_errors.py
import os
import csv
import json
import math
import numpy as np
import cv2
import torch

from models import PoseResNet50
from metrics import quat_angle_deg
from utils_img import imread_any, to_float01, crop_with_roi


# ====== 固定配置（不需要每次写参数） ======
DATA_ROOT = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
TEST_JSON = os.path.join(DATA_ROOT, "test_pose.json")

MODE = "ir"  # "ir" or "rgbir"
OUT_SIZE = 320
ROI_EXPAND = 1.30
ROI_JITTER = 0.0

CKPT = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\runs_pose_fixed\best_ir.pt"

TOPK = 10
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def imwrite(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(path)


def resize_any(img: np.ndarray, out_size: int) -> np.ndarray:
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _norm_rgb_imagenet(rgb01: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb01 - mean) / std


def _norm_ir_default(ir01: np.ndarray) -> np.ndarray:
    mean = 0.5
    std = 0.25
    return (ir01 - mean) / std


def load_state_dict_from_ckpt(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def build_input(item: dict) -> torch.Tensor:
    roi = item["roi_xyxy"]

    # IR
    ir_path = os.path.join(DATA_ROOT, item["ir_path"])
    ir = imread_any(ir_path, flag=cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir01 = to_float01(ir)

    ir_roi = crop_with_roi(ir01, roi, expand=ROI_EXPAND, jitter=ROI_JITTER, rng=np.random)
    ir_roi = resize_any(ir_roi, OUT_SIZE)
    ir_roi_n = _norm_ir_default(ir_roi)

    if MODE == "ir":
        x = ir_roi_n[None, :, :]  # 1xHxW
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

    # RGB
    rgb_path = os.path.join(DATA_ROOT, item["rgb_path"])
    rgb = imread_any(rgb_path, flag=cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb01 = to_float01(rgb)

    rgb_roi = crop_with_roi(rgb01, roi, expand=ROI_EXPAND, jitter=ROI_JITTER, rng=np.random)
    rgb_roi = resize_any(rgb_roi, OUT_SIZE)
    rgb_roi_n = _norm_rgb_imagenet(rgb_roi)

    x4 = np.concatenate([rgb_roi_n.transpose(2, 0, 1), ir_roi_n[None, :, :]], axis=0)
    return torch.from_numpy(x4.astype(np.float32)).unsqueeze(0)


def visualize_one(item: dict, angle_deg: float, out_dir: str):
    sid = item["id"]
    roi = item["roi_xyxy"]

    ir_path = os.path.join(DATA_ROOT, item["ir_path"])
    ir = imread_any(ir_path, flag=cv2.IMREAD_UNCHANGED)
    if ir.ndim == 3:
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    ir01 = to_float01(ir)

    full = (ir01 * 255.0).clip(0, 255).astype(np.uint8)
    full_bgr = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)

    H, W = full.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))

    cv2.rectangle(full_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(full_bgr, f"id={sid} err={angle_deg:.3f}deg ex={ROI_EXPAND:.2f}",
                (max(0, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    imwrite(os.path.join(out_dir, f"{sid}_full_{angle_deg:.3f}deg.png"), full_bgr)

    crop = crop_with_roi(ir01, roi, expand=ROI_EXPAND, jitter=0.0, rng=np.random)
    crop = resize_any(crop, OUT_SIZE)
    crop_u8 = (crop * 255.0).clip(0, 255).astype(np.uint8)
    crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2BGR)
    cv2.putText(crop_bgr, f"id={sid} crop ex={ROI_EXPAND:.2f}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    imwrite(os.path.join(out_dir, f"{sid}_crop_{angle_deg:.3f}deg.png"), crop_bgr)


@torch.no_grad()
def main():
    assert os.path.isdir(DATA_ROOT), f"DATA_ROOT not found: {DATA_ROOT}"
    assert os.path.exists(TEST_JSON), f"TEST_JSON not found: {TEST_JSON}"
    assert os.path.exists(CKPT), f"CKPT not found: {CKPT}"

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    in_ch = 1 if MODE == "ir" else 4
    model = PoseResNet50(in_ch=in_ch, pretrained=False, dropout=0.0).to(DEVICE)
    model.load_state_dict(load_state_dict_from_ckpt(CKPT), strict=True)
    model.eval()

    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"analysis_{MODE}_test_fixed")
    best_dir = os.path.join(out_root, "analysis_best")
    worst_dir = os.path.join(out_root, "analysis_worst")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    records = []
    angles_all = []

    for start in range(0, len(samples), BATCH_SIZE):
        batch_items = samples[start:start+BATCH_SIZE]

        xs = []
        qgts = []
        ids = []

        for it in batch_items:
            x = build_input(it)
            qgt = torch.tensor(it["pose_quat_wxyz"], dtype=torch.float32).unsqueeze(0)
            xs.append(x)
            qgts.append(qgt)
            ids.append(it["id"])

        x = torch.cat(xs, dim=0).to(DEVICE, non_blocking=True)
        q_gt = torch.cat(qgts, dim=0).to(DEVICE, non_blocking=True)

        q_pred = model(x)
        ang = quat_angle_deg(q_pred, q_gt).detach().cpu().numpy()

        for i, it in enumerate(batch_items):
            a = float(ang[i])
            angles_all.append(a)
            records.append({
                "id": it["id"],
                "angle_deg": a,
                "ir_path": it["ir_path"],
                "rgb_path": it.get("rgb_path", ""),
                "roi_xyxy": it["roi_xyxy"],
            })

    angles_all = np.array(angles_all, dtype=np.float32)
    mean = float(angles_all.mean())
    med = float(np.median(angles_all))
    p90 = float(np.percentile(angles_all, 90))

    records_sorted = sorted(records, key=lambda r: r["angle_deg"])
    best = records_sorted[:min(TOPK, len(records_sorted))]
    worst = records_sorted[-min(TOPK, len(records_sorted)):][::-1]

    csv_path = os.path.join(out_root, "errors_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "angle_deg", "ir_path", "rgb_path", "roi_xyxy"])
        for r in records_sorted:
            w.writerow([r["id"], f"{r['angle_deg']:.6f}", r["ir_path"], r["rgb_path"], json.dumps(r["roi_xyxy"])])

    sum_path = os.path.join(out_root, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(f"MODE: {MODE}\n")
        f.write(f"CKPT: {CKPT}\n")
        f.write(f"TEST_JSON: {TEST_JSON}\n")
        f.write(f"ROI_EXPAND: {ROI_EXPAND}\n")
        f.write(f"OUT_SIZE: {OUT_SIZE}\n\n")
        f.write(f"ang(mean/median/p90): {mean:.3f} / {med:.3f} / {p90:.3f}\n\n")

        f.write(f"BEST {len(best)} samples:\n")
        for r in best:
            f.write(f"  {r['id']}  {r['angle_deg']:.3f} deg\n")

        f.write(f"\nWORST {len(worst)} samples:\n")
        for r in worst:
            f.write(f"  {r['id']}  {r['angle_deg']:.3f} deg\n")

    id2item = {it["id"]: it for it in samples}
    for r in best:
        visualize_one(id2item[r["id"]], r["angle_deg"], best_dir)
    for r in worst:
        visualize_one(id2item[r["id"]], r["angle_deg"], worst_dir)

    print(f"[OK] Saved: {out_root}")
    print(f"[OK] errors_test.csv: {csv_path}")
    print(f"[OK] summary.txt: {sum_path}")
    print(f"[OK] ang(mean/median/p90): {mean:.3f}/{med:.3f}/{p90:.3f}")


if __name__ == "__main__":
    main()
