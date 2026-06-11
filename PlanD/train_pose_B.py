# train_pose_fixed.py  (Baseline B: fixed ROI scale, OUT_SIZE=320)
import os
import time
import math
import json
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, get_worker_info
from torch.amp import autocast, GradScaler

from pose_dataset import PoseDataset
from models import PoseResNet50
from metrics import quat_loss, quat_angle_deg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    info = get_worker_info()
    ds = info.dataset
    base = ds.rng.randint(0, 10_000_000)
    ds.rng = np.random.RandomState(base + worker_id * 1000)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_n = 0
    angles = []

    for x, q_gt, _ids in loader:
        x = x.to(device, non_blocking=True)
        q_gt = q_gt.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=amp):
            q_pred = model(x)
            loss = quat_loss(q_pred, q_gt)
            ang = quat_angle_deg(q_pred, q_gt)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        angles.append(ang.detach().cpu())

    angles = torch.cat(angles, dim=0).numpy()
    mean = float(np.mean(angles))
    med  = float(np.median(angles))
    p90  = float(np.percentile(angles, 90))
    return total_loss / max(1, total_n), mean, med, p90


def main():
    parser = argparse.ArgumentParser()

    # 默认路径（你不想每次写参数）
    parser.add_argument("--data_root", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度")
    parser.add_argument("--train_json", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度\train_pose.json")
    parser.add_argument("--val_json", type=str, default=r"D:\BaiduNetdiskDownload\data\Aqua_60度\val_pose.json")

    parser.add_argument("--mode", type=str, choices=["ir", "rgbir"], default="ir")
    parser.add_argument("--out_size", type=int, default=320)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument("--roi_expand", type=float, default=1.30)  # 训练/验证都固定
    parser.add_argument("--roi_jitter", type=float, default=0.10)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # 单独保存到不同目录，防止覆盖你的 scaleaug 实验
    parser.add_argument("--save_dir", type=str, default="runs_pose_fixed")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ===== Dataset / Loader =====
    # Baseline：训练也固定 expand（不做随机尺度增强）
    train_ds = PoseDataset(
        data_root=args.data_root,
        json_path=args.train_json,
        mode=args.mode,
        out_size=args.out_size,
        train=True,
        roi_expand=args.roi_expand,
        roi_jitter=args.roi_jitter,
        roi_expand_mode="fixed",
        seed=args.seed,
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
        seed=args.seed,
    )

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )

    # ===== Model =====
    in_ch = 1 if args.mode == "ir" else 4
    model = PoseResNet50(in_ch=in_ch, pretrained=args.pretrained, dropout=args.dropout).to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        t = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler("cuda", enabled=args.amp)

    best_med = float("inf")
    log_path = os.path.join(args.save_dir, f"log_{args.mode}.txt")

    cfg = vars(args).copy()
    cfg.update({
        "roi_expand_mode_train": "fixed",
        "roi_expand_mode_val": "fixed",
    })
    with open(os.path.join(args.save_dir, f"config_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        total_loss = 0.0
        total_n = 0

        for x, q_gt, _ids in train_loader:
            x = x.to(device, non_blocking=True)
            q_gt = q_gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=args.amp):
                q_pred = model(x)
                loss = quat_loss(q_pred, q_gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

        scheduler.step()
        train_loss = total_loss / max(1, total_n)

        val_loss, mean_deg, med_deg, p90_deg = evaluate(model, val_loader, device, amp=args.amp)

        lr_now = scheduler.get_last_lr()[0]
        msg = (f"Epoch {epoch+1:03d}/{args.epochs} | "
               f"lr {lr_now:.3e} | "
               f"train_loss {train_loss:.5f} | "
               f"val_loss {val_loss:.5f} | "
               f"ang(mean/med/p90) {mean_deg:.3f}/{med_deg:.3f}/{p90_deg:.3f} | "
               f"time {time.time()-t0:.1f}s")
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if med_deg < best_med:
            best_med = med_deg
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_med": best_med,
                "args": cfg
            }
            torch.save(ckpt, os.path.join(args.save_dir, f"best_{args.mode}.pt"))

    print("Training finished. Best median(deg):", best_med)


if __name__ == "__main__":
    main()
