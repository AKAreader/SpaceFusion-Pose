# train_pose.py
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config_pose import (
    DATA_ROOT, TRAIN_JSON, VAL_JSON, MODE, OUT_SIZE,
    USE_ROI_CACHE, ROI_EXPAND_FIXED, ROI_JITTER_TRAIN, ROI_EXPAND_MODE, ROI_EXPAND_MIN, ROI_EXPAND_MAX,
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, GEO_WEIGHT,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, PREFETCH_FACTOR,
    RUNS_DIR, OVERFIT_DEBUG, OVERFIT_N,
    USE_SOFT_MASK, SOFT_MASK_SOURCE
)

from pose_dataset import PoseDataset
from models import PoseResNet50
from models_dual import PhysioSaliencyNet
from metrics import quat_loss, quat_angle_deg
from loss_geo import GeometricLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def amp_autocast(device_type="cuda", enabled=True):
    # 兼容 torch.amp / torch.cuda.amp
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    from torch.cuda.amp import autocast
    return autocast(enabled=enabled)

def amp_scaler(enabled=True):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    from torch.cuda.amp import GradScaler
    return GradScaler(enabled=enabled)

@torch.no_grad()
def evaluate(model, loader, device, mode):
    model.eval()
    losses, angles = [], []
    for batch in loader:
        q_gt = batch["q_gt"].to(device, non_blocking=True)
        if mode == "dual":
            q_pred = model(batch["rgb"].to(device), batch["ir"].to(device), batch["saliency"].to(device))
        else:
            q_pred = model(batch["x"].to(device))
        losses.append(quat_loss(q_pred, q_gt).item())
        angles.append(quat_angle_deg(q_pred, q_gt).detach().cpu())
    if not angles:
        return 0.0, 0.0, 0.0, 0.0
    ang = torch.cat(angles, dim=0).numpy()
    return float(np.mean(losses)), float(np.mean(ang)), float(np.median(ang)), float(np.percentile(ang, 90))

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    save_dir = os.path.join(RUNS_DIR, MODE)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    roi_cache_path = None
    if USE_ROI_CACHE:
        roi_cache_path = os.path.join(DATA_ROOT, f"roi_cache_{'train'}.json")

    train_ds = PoseDataset(
        DATA_ROOT, TRAIN_JSON, mode=MODE, out_size=OUT_SIZE, train=True,
        roi_expand=ROI_EXPAND_FIXED, roi_jitter=ROI_JITTER_TRAIN,
        roi_expand_mode=ROI_EXPAND_MODE, roi_expand_min=ROI_EXPAND_MIN, roi_expand_max=ROI_EXPAND_MAX,
        roi_cache_path=roi_cache_path,
        use_soft_mask=USE_SOFT_MASK,
        soft_mask_source=SOFT_MASK_SOURCE
    )
    val_cache = os.path.join(DATA_ROOT, "roi_cache_val.json") if USE_ROI_CACHE else None
    val_ds = PoseDataset(
        DATA_ROOT, VAL_JSON, mode=MODE, out_size=OUT_SIZE, train=False,
        roi_expand=ROI_EXPAND_FIXED, roi_jitter=0.0,
        roi_expand_mode="fixed",
        roi_cache_path=val_cache,
        use_soft_mask=USE_SOFT_MASK,
        soft_mask_source=SOFT_MASK_SOURCE
    )

    if OVERFIT_DEBUG:
        print(f"[DEBUG] overfit mode enabled: n={OVERFIT_N} (val=train subset)")
        idx = list(range(min(OVERFIT_N, len(train_ds))))
        train_ds.samples = [train_ds.samples[i] for i in idx]
        val_ds.samples = [val_ds.samples[i] for i in idx]

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

    print(f"[INFO] device={device} amp={use_amp}")
    print(f"[INFO] mode={MODE} save_dir={save_dir}")
    print(f"[INFO] train={len(train_ds)} val={len(val_ds)} batch={BATCH_SIZE} out_size={OUT_SIZE}")
    print(f"[INFO] roi_cache={'ON' if USE_ROI_CACHE else 'OFF'} soft_mask={'ON' if USE_SOFT_MASK else 'OFF'}")
    print(f"[INFO] roi_expand={ROI_EXPAND_FIXED} jitter={ROI_JITTER_TRAIN} expand_mode={ROI_EXPAND_MODE}")
    print(f"[INFO] workers={NUM_WORKERS} pin_memory={PIN_MEMORY} persistent={PERSISTENT_WORKERS} prefetch={PREFETCH_FACTOR}")

    if MODE == "dual":
        model = PhysioSaliencyNet(dropout=0.1).to(device)
    else:
        in_ch = 4 if MODE == "rgbir" else 1
        model = PoseResNet50(in_ch=in_ch, pretrained=True, dropout=0.1).to(device)

    geo_loss_fn = GeometricLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )
    scaler = amp_scaler(enabled=use_amp)

    best_med = float("inf")

    for ep in range(EPOCHS):
        model.train()
        loss_pos_list, loss_geo_list = [], []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            q_gt = batch["q_gt"].to(device, non_blocking=True)

            with amp_autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                if MODE == "dual":
                    q_pred = model(batch["rgb"].to(device), batch["ir"].to(device), batch["saliency"].to(device))
                else:
                    q_pred = model(batch["x"].to(device))

                l_pos = quat_loss(q_pred, q_gt)
                l_geo = geo_loss_fn(q_pred, q_gt, batch["t_gt"].to(device), batch["K"].to(device))
                loss = l_pos + GEO_WEIGHT * l_geo

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_pos_list.append(float(l_pos.item()))
            loss_geo_list.append(float(l_geo.item()))

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                print(f"  ep={ep+1:03d} step={step:04d}/{len(train_loader)} loss={loss.item():.4f} pos={l_pos.item():.4f} geo={l_geo.item():.4f}")

        val_l, mean, med, p90 = evaluate(model, val_loader, device, MODE)
        msg = (
            f"Ep {ep+1:03d}/{EPOCHS} | "
            f"Train pos={np.mean(loss_pos_list):.4f} geo={np.mean(loss_geo_list):.4f} | "
            f"Val loss={val_l:.4f} | "
            f"Angle mean/med/p90={mean:.2f}/{med:.2f}/{p90:.2f} | "
            f"T={time.time()-t0:.0f}s"
        )
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if med < best_med:
            best_med = med
            ckpt = {
                "model": model.state_dict(),
                "best_med": best_med,
                "mode": MODE,
                "out_size": OUT_SIZE
            }
            save_path = os.path.join(save_dir, f"best_{MODE}.pt")
            torch.save(ckpt, save_path)
            print(f"[SAVE] {save_path}  best_med={best_med:.3f}")

if __name__ == "__main__":
    main()
