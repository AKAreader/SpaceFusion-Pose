#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Compare 2~3 models:
- export fused images on VAL
- compute fusion quality metrics (EN/AG/SF/SSIM/MI)
- downstream pose eval: freeze fusion, train a unified pose head, compare angle errors
- generate intuitive plots (metrics + pose angle)

运行：
- 可直接 python again.py
- 或用 launcher.py 点按钮运行
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
from utils import RGB2YCrCb, YCbCr2RGB, save_img_single


# -------------------------
# 0) 配置（可用环境变量覆盖，方便 launcher）
# -------------------------
@dataclass
class CFG:
    RGB_DIR: str = os.environ.get("RGB_DIR", r"E:\Test\RGB")
    IR_DIR: str  = os.environ.get("IR_DIR",  r"E:\Test\IR")
    XLSX: str    = os.environ.get("XLSX",    r"E:\Test\Aqua_with_unified_id.xlsx")

    GPU_ID: int = int(os.environ.get("GPU_ID", "0"))
    BATCH_SIZE_VAL: int = int(os.environ.get("BATCH_SIZE_VAL", "8"))
    NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "4"))
    RESIZE_HW: Tuple[int, int] = (480, 640)

    # ====== 这里填你要对比的 ckpt（默认用环境变量传入；不传就手改下面）======
    CKPT_BASELINE: str = os.environ.get("CKPT_BASELINE", r"")
    CKPT_ROUTEA: str   = os.environ.get("CKPT_ROUTEA",   r"")
    CKPT_GATE: str     = os.environ.get("CKPT_GATE",     r"")

    # 下游 pose eval
    POSE_EPOCHS: int = int(os.environ.get("POSE_EPOCHS", "12"))
    LR_POSE: float = float(os.environ.get("LR_POSE", "5e-4"))
    POSE_INPUT_RGB: bool = os.environ.get("POSE_INPUT_RGB", "1") == "1"  # fused RGB 输入 pose head

    OUT_ROOT: str = os.environ.get("COMPARE_ROOT", r"./compare_runs")


# -------------------------
# 1) Logger & dirs
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def setup_logger(out_dir: str) -> logging.Logger:
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "compare.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("compare")

def safe_torch_load_weights(path: str, device: torch.device):
    # 兼容 torch.load weights_only 参数
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# -------------------------
# 2) Pose head + utils
# -------------------------
class PoseRegressor(nn.Module):
    def __init__(self, out_dim: int = 4, in_ch: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

def quat_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).clamp(-1.0, 1.0)
    return torch.mean(1.0 - dot * dot)

@torch.no_grad()
def quat_angle_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot) * (180.0 / np.pi)
    return ang


# -------------------------
# 3) Excel pose map（同 train 脚本那套 key 变体）
# -------------------------
def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.endswith(".0") and s.replace(".", "", 1).isdigit():
        s2 = s[:-2]
        if s2.isdigit():
            return s2
    return s

def _stem(name: str) -> str:
    base = os.path.basename(_safe_str(name))
    return os.path.splitext(base)[0]

def _key_variants(s: str) -> List[str]:
    s = _safe_str(s)
    if not s:
        return []
    base = os.path.basename(s)
    st = os.path.splitext(base)[0]
    out = []
    for v in [s, base, st]:
        if v and v not in out:
            out.append(v)
    if st.isdigit():
        n = int(st)
        out.append(str(n))
        for w in [3, 4, 5, 6, 7, 8]:
            out.append(str(n).zfill(w))
            out.append(str(n).zfill(w) + ".png")
        out.append(str(n) + ".png")
    return list(dict.fromkeys(out))

def load_pose_map_from_excel(xlsx_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = None
    for c in ["pair_id", "Unified_ID", "帧索引"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    need = ["卫星四元数_w", "卫星四元数_x", "卫星四元数_y", "卫星四元数_z"]
    alt = ["qw", "qx", "qy", "qz"]
    if all(c in df.columns for c in need):
        qcols = need
    elif all(c in df.columns for c in alt):
        qcols = alt
    else:
        raise RuntimeError("Quaternion columns not found in Excel.")

    for c in qcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[id_col] + qcols)

    q = df[qcols].to_numpy(dtype=np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q = q / n
    df.loc[:, qcols] = q

    pose_map: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        rid = _safe_str(r[id_col])
        quat = r[qcols].to_numpy(dtype=np.float32)
        for k in _key_variants(rid):
            pose_map[k] = quat
    return pose_map

def lookup_pose_batch(names, pose_map: Dict[str, np.ndarray], device: torch.device):
    names_list = list(names)
    B = len(names_list)
    pose = np.zeros((B, 4), dtype=np.float32)
    mask = np.zeros((B, 1), dtype=np.float32)
    for i, nm in enumerate(names_list):
        s = _safe_str(nm)
        found = None
        for k in _key_variants(s):
            if k in pose_map:
                found = pose_map[k]
                break
        if found is None:
            st = _stem(s)
            for k in _key_variants(st):
                if k in pose_map:
                    found = pose_map[k]
                    break
        if found is not None:
            pose[i] = found
            mask[i, 0] = 1.0
    hit = float(mask.mean())
    return (
        torch.from_numpy(pose).to(device=device, dtype=torch.float32),
        torch.from_numpy(mask).to(device=device, dtype=torch.float32),
        hit
    )


# -------------------------
# 4) Fusion metrics（EN/AG/SF/SSIM/MI）—— 修复 MI 实现
# -------------------------
def to_uint8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)

def entropy_en(img01: np.ndarray) -> float:
    u = to_uint8(img01).ravel()
    hist = np.bincount(u, minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def avg_gradient_ag(img01: np.ndarray) -> float:
    u = img01.astype(np.float32)
    gx = u[:, 1:] - u[:, :-1]
    gy = u[1:, :] - u[:-1, :]
    return float(np.mean(np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2) + 1e-12))

def spatial_frequency_sf(img01: np.ndarray) -> float:
    u = img01.astype(np.float32)
    rf = np.sqrt(np.mean((u[:, 1:] - u[:, :-1]) ** 2) + 1e-12)
    cf = np.sqrt(np.mean((u[1:, :] - u[:-1, :]) ** 2) + 1e-12)
    return float(np.sqrt(rf * rf + cf * cf))

def ssim_simple(a01: np.ndarray, b01: np.ndarray) -> float:
    # 简化 SSIM（窗口版很重；对比趋势足够用）
    a = a01.astype(np.float64)
    b = b01.astype(np.float64)
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    return float(((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2) + 1e-12))

def mutual_information(a01: np.ndarray, b01: np.ndarray, bins: int = 64) -> float:
    # 标准 MI：sum pxy * log(pxy/(px*py))
    a = to_uint8(a01).ravel()
    b = to_uint8(b01).ravel()
    hist2d, _, _ = np.histogram2d(a, b, bins=bins, range=[[0, 255], [0, 255]])
    pxy = hist2d / (hist2d.sum() + 1e-12)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = (pxy[nz] * (np.log2(pxy[nz]) - np.log2(px[nz.any(axis=1)])[:, 0] + 0.0 - np.log2(py[:, nz.any(axis=0)][0]))).sum()
    # 上面为了避免复杂广播，换成更稳的写法：
    # 直接用外积：
    px2 = px.reshape(-1, 1)
    py2 = py.reshape(1, -1)
    denom = px2 * py2 + 1e-12
    mi = float((pxy[nz] * np.log2((pxy[nz] + 1e-12) / (denom[nz] + 1e-12))).sum())
    return mi

def metric_pack(fused01: np.ndarray, y01: np.ndarray, ir01: np.ndarray) -> Dict[str, float]:
    EN = entropy_en(fused01)
    AG = avg_gradient_ag(fused01)
    SF = spatial_frequency_sf(fused01)
    SSIM = 0.5 * (ssim_simple(fused01, y01) + ssim_simple(fused01, ir01))
    MI = mutual_information(fused01, y01) + mutual_information(fused01, ir01)
    return {"EN": EN, "AG": AG, "SF": SF, "SSIM": SSIM, "MI": MI}


# -------------------------
# 5) Export + metrics on VAL
# -------------------------
@torch.no_grad()
def export_and_metrics(model: nn.Module, val_loader: DataLoader, device: torch.device, out_dir: str, tag: str) -> pd.DataFrame:
    fused_dir = os.path.join(out_dir, f"fused_{tag}")
    ensure_dir(fused_dir)

    rows = []
    pbar = tqdm(val_loader, desc=f"Export+Metrics on VAL [{tag}]")
    model.eval()

    for (vis, ir, _pose, _mask, names) in pbar:
        vis = vis.to(device)
        ir  = ir.to(device)

        Y, Cb, Cr = RGB2YCrCb(vis)
        fused_Y = model(Y, ir)
        fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)

        # numpy for metrics (Y/IR/fused in [0,1])
        Y_np = Y.detach().cpu().numpy()[:, 0]
        IR_np = ir.detach().cpu().numpy()[:, 0]
        F_np = fused_Y.detach().cpu().numpy()[:, 0]

        for k, nm in enumerate(names):
            # save fused RGB
            save_img_single(fused_rgb[k], os.path.join(fused_dir, nm))

            m = metric_pack(F_np[k], Y_np[k], IR_np[k])
            m["name"] = nm
            rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"metrics_{tag}_per_image.csv"), index=False, encoding="utf-8-sig")
    return df


# -------------------------
# 6) Downstream pose eval（冻结 fusion，训练统一 pose head）
# -------------------------
def train_pose_head(fusion_model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    pose_map: Dict[str, np.ndarray], device: torch.device,
                    pose_epochs: int, lr_pose: float, pose_input_rgb: bool,
                    out_dir: str, tag: str) -> pd.DataFrame:

    pose_net = PoseRegressor(out_dim=4, in_ch=(3 if pose_input_rgb else 1)).to(device)
    opt = torch.optim.Adam(pose_net.parameters(), lr=lr_pose)

    records = []
    fusion_model.eval()

    for ep in range(1, pose_epochs + 1):
        pose_net.train()
        tr_losses = []
        tr_hits = []

        pbar = tqdm(train_loader, desc=f"{tag} PoseTrain ep{ep}/{pose_epochs}")
        for (vis, ir, _pose, _mask, names) in pbar:
            vis = vis.to(device)
            ir  = ir.to(device)

            pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
            valid = (pose_mask.view(-1) > 0.5)
            if int(valid.sum().item()) == 0:
                continue

            with torch.no_grad():
                Y, Cb, Cr = RGB2YCrCb(vis)
                fused_Y = fusion_model(Y, ir)
                fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if pose_input_rgb else fused_Y

            pred = pose_net(fused_in)
            loss = quat_loss(pred[valid], pose_gt[valid])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_losses.append(float(loss.detach().cpu()))
            tr_hits.append(hit)
            pbar.set_postfix_str(f"loss={np.mean(tr_losses):.4f} hit={np.mean(tr_hits):.3f}")

        # val
        pose_net.eval()
        all_ang = []
        val_hits = []
        with torch.no_grad():
            for (vis, ir, _pose, _mask, names) in val_loader:
                vis = vis.to(device)
                ir  = ir.to(device)

                pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
                valid = (pose_mask.view(-1) > 0.5)
                if int(valid.sum().item()) == 0:
                    continue

                Y, Cb, Cr = RGB2YCrCb(vis)
                fused_Y = fusion_model(Y, ir)
                fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if pose_input_rgb else fused_Y

                pred = pose_net(fused_in)
                ang = quat_angle_deg(pred[valid], pose_gt[valid])
                all_ang.append(ang.detach().cpu().numpy())
                val_hits.append(hit)

        angle_mean = angle_med = angle_p95 = float("nan")
        if all_ang:
            a = np.concatenate(all_ang, axis=0)
            angle_mean = float(np.mean(a))
            angle_med  = float(np.median(a))
            angle_p95  = float(np.percentile(a, 95))

        rec = {
            "model": tag,
            "epoch": ep,
            "train_loss": float(np.mean(tr_losses)) if tr_losses else float("nan"),
            "train_hit": float(np.mean(tr_hits)) if tr_hits else float("nan"),
            "val_angle_mean": angle_mean,
            "val_angle_med": angle_med,
            "val_angle_p95": angle_p95,
            "val_hit": float(np.mean(val_hits)) if val_hits else float("nan"),
        }
        records.append(rec)
        logging.getLogger("compare").info(
            f"[{tag}] ep{ep}: train_loss={rec['train_loss']:.4f} train_hit={rec['train_hit']:.3f} | "
            f"VAL angle mean/med/p95 = {angle_mean:.2f}/{angle_med:.2f}/{angle_p95:.2f} deg | val_hit={rec['val_hit']:.3f}"
        )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, f"pose_train_curve_{tag}.csv"), index=False, encoding="utf-8-sig")
    return df


# -------------------------
# 7) Plotting（直观图）
# -------------------------
def plot_metrics_summary(summary_csv: str, out_png: str):
    df = pd.read_csv(summary_csv)
    # df columns: model, EN_mean, AG_mean, ...
    metrics = ["EN", "AG", "SF", "SSIM", "MI"]

    plt.figure()
    x = np.arange(len(metrics))
    width = 0.25
    models = df["model"].tolist()

    for i, m in enumerate(models):
        vals = [df.loc[i, f"{k}_mean"] for k in metrics]
        plt.bar(x + (i - (len(models)-1)/2) * width, vals, width=width, label=m)

    plt.xticks(x, metrics)
    plt.title("Fusion Quality Metrics (VAL mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_pose_curve(curves: List[Tuple[str, str]], out_png: str):
    # curves: [(tag, csv_path), ...]
    plt.figure()
    for tag, csv_path in curves:
        df = pd.read_csv(csv_path)
        plt.plot(df["epoch"].values, df["val_angle_mean"].values, marker="o", label=f"{tag} mean")
    plt.xlabel("Pose head epoch")
    plt.ylabel("Val angle error (deg)")
    plt.title("Downstream Pose Angle Error (mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------------
# 8) Main
# -------------------------
def main():
    cfg = CFG()

    out_dir = os.path.abspath(os.path.join(cfg.OUT_ROOT, f"compare_{time.strftime('%Y%m%d_%H%M%S')}"))
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # dataset split must match training behavior
    train_ds = Fusion_dataset(split="train", ir_path=cfg.IR_DIR, vi_path=cfg.RGB_DIR, xlsx_path=cfg.XLSX, resize_hw=cfg.RESIZE_HW)
    val_ds   = Fusion_dataset(split="val",   ir_path=cfg.IR_DIR, vi_path=cfg.RGB_DIR, xlsx_path=cfg.XLSX, resize_hw=cfg.RESIZE_HW,
                              split_seed=train_ds.split_seed, train_ratio=train_ds.train_ratio)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE_VAL, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=(cfg.NUM_WORKERS > 0))
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE_VAL, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
                              persistent_workers=(cfg.NUM_WORKERS > 0))

    # pose map
    pose_map = load_pose_map_from_excel(cfg.XLSX)
    logger.info(f"[POSE] pose_map_keys={len(pose_map)}")

    # models list
    models = []
    if cfg.CKPT_BASELINE:
        models.append(("baseline", cfg.CKPT_BASELINE))
    if cfg.CKPT_ROUTEA:
        models.append(("routeA_v2", cfg.CKPT_ROUTEA))
    if cfg.CKPT_GATE:
        models.append(("routeB_gate", cfg.CKPT_GATE))

    if len(models) < 2:
        raise RuntimeError("Need at least 2 ckpts. Set CKPT_BASELINE/CKPT_ROUTEA/CKPT_GATE.")

    logger.info("CKPTs:")
    for tag, p in models:
        logger.info(f"  {tag}: {p}")

    # load & export+metrics
    metric_dfs = {}
    for tag, ckpt_path in models:
        m = FusionNet(output=1).to(device)
        sd = safe_torch_load_weights(ckpt_path, device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        metric_dfs[tag] = export_and_metrics(m, val_loader, device, out_dir, tag)

    # merge per-image metrics
    base_tag = models[0][0]
    df_merge = metric_dfs[base_tag].copy()
    df_merge = df_merge.rename(columns={c: f"{c}_{base_tag}" for c in ["EN", "AG", "SF", "SSIM", "MI"]})

    for tag, _ in models[1:]:
        df2 = metric_dfs[tag].rename(columns={c: f"{c}_{tag}" for c in ["EN", "AG", "SF", "SSIM", "MI"]})
        df_merge = df_merge.merge(df2[["name"] + [f"{c}_{tag}" for c in ["EN","AG","SF","SSIM","MI"]]], on="name", how="inner")

    df_merge.to_csv(os.path.join(out_dir, "metrics_per_image.csv"), index=False, encoding="utf-8-sig")

    # summary
    sum_rows = []
    for tag, _ in models:
        df = metric_dfs[tag]
        row = {"model": tag}
        for c in ["EN", "AG", "SF", "SSIM", "MI"]:
            row[f"{c}_mean"] = float(df[c].mean())
            row[f"{c}_std"]  = float(df[c].std())
        sum_rows.append(row)

    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")
    logger.info("[OK] fusion compare saved: metrics_per_image.csv + metrics_summary.csv + fused_*")

    # downstream pose eval
    logger.info("Downstream pose eval: ON (freeze fusion, train unified pose head).")
    pose_dir = os.path.join(out_dir, "downstream_pose")
    ensure_dir(pose_dir)

    pose_curves = []
    for tag, ckpt_path in models:
        fusion = FusionNet(output=1).to(device)
        sd = safe_torch_load_weights(ckpt_path, device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        fusion.load_state_dict(sd, strict=False)
        curve = train_pose_head(
            fusion, train_loader, val_loader, pose_map, device,
            pose_epochs=cfg.POSE_EPOCHS, lr_pose=cfg.LR_POSE, pose_input_rgb=cfg.POSE_INPUT_RGB,
            out_dir=pose_dir, tag=tag
        )
        curve_path = os.path.join(pose_dir, f"pose_train_curve_{tag}.csv")
        pose_curves.append((tag, curve_path))

    # 合并 pose summary
    all_curve = []
    for tag, path in pose_curves:
        df = pd.read_csv(path)
        df["model"] = tag
        all_curve.append(df)
    df_pose = pd.concat(all_curve, ignore_index=True)
    df_pose.to_csv(os.path.join(pose_dir, "pose_train_curve_all.csv"), index=False, encoding="utf-8-sig")

    # plots
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    plot_metrics_summary(os.path.join(out_dir, "metrics_summary.csv"),
                         os.path.join(plots_dir, "metrics_summary.png"))
    plot_pose_curve(pose_curves, os.path.join(plots_dir, "pose_angle_mean.png"))

    logger.info(f"[OK] plots saved: {plots_dir}")
    logger.info(f"DONE. All outputs at: {out_dir}")
    print("\n[OK] DONE. Output folder:\n" + out_dir)


if __name__ == "__main__":
    main()
