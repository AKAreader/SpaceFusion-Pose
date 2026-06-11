# test_pose.py (PlanA_RGB)
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from pose_dataset import PoseDataset
from models import PoseResNet50
from metrics import quat_loss, quat_angle_deg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度")
    parser.add_argument("--json_path", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度\test_pose.json")
    parser.add_argument("--ckpt", type=str, default=r"runs_pose_rgb\best_rgb.pt")
    parser.add_argument("--mode", type=str, choices=["ir", "rgb", "rgbir"], default="rgb")
    parser.add_argument("--out_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--export_dir", type=str, default=r"test_exports_rgb")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.export_dir, exist_ok=True)

    ds = PoseDataset(
        data_root=args.data_root,
        json_path=args.json_path,
        mode=args.mode,
        out_size=args.out_size,
        train=False,
        roi_expand=1.30,
        roi_jitter=0.0,
        roi_expand_mode="fixed",
        share_crop=True,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    in_ch = 1 if args.mode == "ir" else (3 if args.mode == "rgb" else 4)
    model = PoseResNet50(in_ch=in_ch, pretrained=False, dropout=0.0).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    all_ids = []
    all_ang = []
    all_loss = []
    all_qpred = []
    all_qgt = []

    for x, q_gt, ids in loader:
        x = x.to(device, non_blocking=True)
        q_gt = q_gt.to(device, non_blocking=True)
        q_pred = model(x)

        loss = quat_loss(q_pred, q_gt)
        ang = quat_angle_deg(q_pred, q_gt)

        all_ids += list(ids)
        all_loss.append(loss.item())
        all_ang.append(ang.detach().cpu().numpy())
        all_qpred.append(q_pred.detach().cpu().numpy())
        all_qgt.append(q_gt.detach().cpu().numpy())

    ang = np.concatenate(all_ang, axis=0)
    mean = float(np.mean(ang))
    med  = float(np.median(ang))
    p90  = float(np.percentile(ang, 90))

    print(f"[TEST] mode={args.mode} ckpt={args.ckpt}")
    print(f"[TEST] loss {float(np.mean(all_loss)):.5f} | ang(mean/med/p90) {mean:.3f}/{med:.3f}/{p90:.3f}")

    qpred = np.concatenate(all_qpred, axis=0)
    qgt = np.concatenate(all_qgt, axis=0)

    # export csv
    out_csv = os.path.join(args.export_dir, f"pred_test_full_{args.mode}_out{args.out_size}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("id,ang_deg,q_pred_w,q_pred_x,q_pred_y,q_pred_z,q_gt_w,q_gt_x,q_gt_y,q_gt_z\n")
        for i, sid in enumerate(all_ids):
            f.write(f"{sid},{ang[i]:.6f},"
                    f"{qpred[i,0]:.6f},{qpred[i,1]:.6f},{qpred[i,2]:.6f},{qpred[i,3]:.6f},"
                    f"{qgt[i,0]:.6f},{qgt[i,1]:.6f},{qgt[i,2]:.6f},{qgt[i,3]:.6f}\n")

    summary = {
        "mode": args.mode,
        "ckpt": args.ckpt,
        "out_size": args.out_size,
        "mean_deg": mean,
        "median_deg": med,
        "p90_deg": p90,
        "csv": out_csv
    }
    out_json = os.path.join(args.export_dir, f"summary_{args.mode}_out{args.out_size}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] csv:", out_csv)
    print("[OK] summary:", out_json)


if __name__ == "__main__":
    main()
