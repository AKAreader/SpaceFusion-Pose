# train_pose.py
import os
import time
import math
import json
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from pose_dataset import PoseDataset
from models import PoseResNet50
from metrics import quat_loss, quat_angle_deg

from geometry_loss import GeometricReprojLoss, make_default_sat_points


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, use_camera: bool = False):
    model.eval()
    losses = []
    angles = []
    for batch in loader:
        if not use_camera:
            x, q_gt, _ids = batch
        else:
            x, q_gt, t_gt, K, box, _ids = batch  # noqa: F841  (evaluate不使用t_gt,K)

        x = x.to(device, non_blocking=True)
        q_gt = q_gt.to(device, non_blocking=True)

        q_pred = model(x)
        loss = quat_loss(q_pred, q_gt)
        ang = quat_angle_deg(q_pred, q_gt)

        losses.append(loss.item())
        angles.append(ang.detach().cpu())

    angles = torch.cat(angles, dim=0).numpy()
    mean = float(np.mean(angles))
    med  = float(np.median(angles))
    p90  = float(np.percentile(angles, 90))
    return float(np.mean(losses)), mean, med, p90


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度")
    parser.add_argument("--train_json", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度\train_pose.json")
    parser.add_argument("--val_json", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度\val_pose.json")

    parser.add_argument("--mode", type=str, choices=["ir", "rgbir"], default="rgbir")
    parser.add_argument("--out_size", type=int, default=320)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--roi_expand", type=float, default=1.30)
    parser.add_argument("--roi_jitter", type=float, default=0.10)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", type=str, default="runs_pose")

    # ===== geometry regularization (optional) =====
    parser.add_argument("--geo_w", type=float, default=0.1,
                        help=">0 to enable reprojection regularization")
    parser.add_argument("--geo_huber", type=float, default=10.0,
                        help="Huber delta in pixels on out_size image")
    parser.add_argument("--geo_pts_scale", type=float, default=1.0,
                        help="Scale virtual 3D points (meters)")

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    use_camera = args.geo_w > 0.0

    # Dataset / Loader
    train_ds = PoseDataset(
        data_root=args.data_root,
        json_path=args.train_json,
        mode=args.mode,
        out_size=args.out_size,
        train=True,
        roi_expand=args.roi_expand,
        roi_jitter=args.roi_jitter,
        roi_expand_mode="random",
        roi_expand_min=1.2,
        roi_expand_max=1.7,
        share_crop=True,
        return_camera=use_camera,
    )
    val_ds = PoseDataset(
        data_root=args.data_root,
        json_path=args.val_json,
        mode=args.mode,
        out_size=args.out_size,
        train=False,
        roi_expand=args.roi_expand,
        roi_jitter=0.0,
        roi_expand_mode="fixed",
        share_crop=True,
        return_camera=use_camera,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True
    )

    # Model
    in_ch = 1 if args.mode == "ir" else 4
    model = PoseResNet50(in_ch=in_ch, pretrained=args.pretrained, dropout=args.dropout).to(device)
    if args.compile:
        model = torch.compile(model)

    # Geometry loss (optional)
    geo_loss_fn = None
    if use_camera:
        pts = make_default_sat_points(device=device, scale=args.geo_pts_scale)
        geo_loss_fn = GeometricReprojLoss(pts_body=pts, huber_delta=args.geo_huber).to(device)

    # Optimizer / Scheduler（warmup + cosine）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        t = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=args.amp)

    best_med = float("inf")
    log_path = os.path.join(args.save_dir, f"log_{args.mode}.txt")

    # 记录配置
    with open(os.path.join(args.save_dir, f"config_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0

        for batch in train_loader:
            if not use_camera:
                x, q_gt, _ids = batch
                t_gt = K = None
            else:
                x, q_gt, t_gt, K, box, _ids = batch  # noqa: F841

            x = x.to(device, non_blocking=True)
            q_gt = q_gt.to(device, non_blocking=True)

            if use_camera:
                t_gt = t_gt.to(device, non_blocking=True)
                K = K.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                q_pred = model(x)
                loss = quat_loss(q_pred, q_gt)

                if use_camera:
                    loss_geo = geo_loss_fn(q_pred, q_gt, t_gt, K)
                    loss = loss + float(args.geo_w) * loss_geo

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        scheduler.step()
        train_loss = running / max(1, len(train_loader))

        val_loss, mean_deg, med_deg, p90_deg = evaluate(model, val_loader, device, use_camera=use_camera)

        lr_now = scheduler.get_last_lr()[0]
        msg = (f"Epoch {epoch+1:03d}/{args.epochs} | "
               f"lr {lr_now:.3e} | "
               f"train_loss {train_loss:.5f} | "
               f"val_loss {val_loss:.5f} | "
               f"ang(mean/med/p90) {mean_deg:.3f}/{med_deg:.3f}/{p90_deg:.3f} | "
               f"geo_w {args.geo_w:.3g} | "
               f"time {time.time()-t0:.1f}s")
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        # 保存最好模型（按 median 角误差）
        if med_deg < best_med:
            best_med = med_deg
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_med": best_med,
                "args": vars(args)
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"best_{args.mode}.pt"))

    print("Training finished. Best median(deg):", best_med)


if __name__ == "__main__":
    main()
