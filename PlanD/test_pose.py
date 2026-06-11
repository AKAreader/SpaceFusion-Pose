# test_pose.py
# Python 3.8+ compatible

import os
import csv
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from pose_dataset import PoseDataset
from models import PoseResNet50
from models_dual import PhysioSaliencyNet
from metrics import quat_loss, quat_angle_deg


def quat_to_euler_wxyz(q):
    w, x, y, z = q
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * x + y * z)
    t4 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t3, t4)
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def find_ckpt(save_dir: str, mode: str) -> str:
    p = os.path.join(save_dir, mode, f"best_{mode}.pt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"ckpt not found: {p}")
    return p


@torch.no_grad()
def run_test(model, loader, device, out_csv: str, mode: str):
    model.eval()
    rows = []
    losses = []
    angles = []

    for batch in loader:
        q_gt = batch["q_gt"].to(device)
        ids = batch["id"]

        if mode == "dual":
            q_pred = model(batch["rgb"].to(device), batch["ir"].to(device), batch["saliency"].to(device))
        else:
            q_pred = model(batch["x"].to(device))

        losses.append(float(quat_loss(q_pred, q_gt).cpu()))
        ang = quat_angle_deg(q_pred, q_gt).cpu().numpy()
        angles.append(ang)

        q_pred_np = q_pred.cpu().numpy()
        q_gt_np = q_gt.cpu().numpy()

        for i in range(len(ids)):
            yp, pp, rp = quat_to_euler_wxyz(q_pred_np[i])
            yg, pg, rg = quat_to_euler_wxyz(q_gt_np[i])
            rows.append({
                "id": int(ids[i]),
                "angle_deg": float(ang[i]),
                "q_pred_w": float(q_pred_np[i, 0]),
                "q_pred_x": float(q_pred_np[i, 1]),
                "q_pred_y": float(q_pred_np[i, 2]),
                "q_pred_z": float(q_pred_np[i, 3]),
                "q_gt_w": float(q_gt_np[i, 0]),
                "q_gt_x": float(q_gt_np[i, 1]),
                "q_gt_y": float(q_gt_np[i, 2]),
                "q_gt_z": float(q_gt_np[i, 3]),
                "pred_yaw": float(yp),
                "pred_pitch": float(pp),
                "pred_roll": float(rp),
                "gt_yaw": float(yg),
                "gt_pitch": float(pg),
                "gt_roll": float(rg),
                "mode": mode,
            })

    ang_all = np.concatenate(angles, axis=0) if angles else np.array([0.0], dtype=np.float32)
    loss_mean = float(np.mean(losses)) if losses else 0.0
    mean = float(np.mean(ang_all))
    med = float(np.median(ang_all))
    p90 = float(np.percentile(ang_all, 90))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = os.path.join(os.path.dirname(out_csv), "summary_test.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"mode={mode}\n")
        f.write(f"loss_mean={loss_mean:.6f}\n")
        f.write(f"angle_mean={mean:.6f}\n")
        f.write(f"angle_median={med:.6f}\n")
        f.write(f"angle_p90={p90:.6f}\n")

    print(f"[OK] mean/median/p90 = {mean:.3f}/{med:.3f}/{p90:.3f}")
    print(f"[OK] csv={out_csv}")
    print(f"[OK] summary={summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"D:\BaiduNetdiskDownload\All")
    ap.add_argument("--mode", default="dual", choices=["ir", "rgbir", "dual"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--save_dir", default="runs_pose")
    ap.add_argument("--out_dir", default="test_outputs_pose")
    ap.add_argument("--out_size", type=int, default=320)
    ap.add_argument("--roi_expand", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_json = os.path.join(args.data_root, "test_pose.json")
    ds = PoseDataset(
        data_root=args.data_root, json_path=test_json,
        mode=args.mode, out_size=args.out_size,
        train=False, roi_expand=args.roi_expand, roi_jitter=0.0, seed=args.seed
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    ckpt_path = find_ckpt(args.save_dir, args.mode)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if args.mode == "dual":
        model = PhysioSaliencyNet(dropout=0.1)
    else:
        in_ch = 1 if args.mode == "ir" else 4
        model = PoseResNet50(in_ch=in_ch, pretrained=False, dropout=0.1)

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    out_csv = os.path.join(args.out_dir, args.mode, "pred_test_full.csv")
    run_test(model, loader, device, out_csv, args.mode)


if __name__ == "__main__":
    main()
