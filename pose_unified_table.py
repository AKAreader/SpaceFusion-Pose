#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doctor import (
    PairFolderDataset,
    PoseRegressorPlain,
    _load_fusion_model,
    joint_saliency,
    load_pose_map_from_excel,
    lookup_pose_one,
    quat_angle_deg,
)
from utils import RGB2YCrCb, YCbCr2RGB


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _pick_latest(patterns: List[str], fallback: str) -> str:
    hits: List[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat))
    hits = [os.path.abspath(p) for p in hits if os.path.isfile(p)]
    if not hits:
        return fallback
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


DEFAULT_SEAFUSION_CKPT = _pick_latest(
    [
        os.path.join(REPO_ROOT, "runs_fusion", "baseline", "*", "ckpt", "fusion_best_total.pth"),
        os.path.join(REPO_ROOT, "runs_fusion", "baseline", "*", "ckpt", "fusion_best.pth"),
    ],
    os.path.join(REPO_ROOT, "model", "Fusion", "fusionmodel_final.pth"),
)

DEFAULT_SOPD_CKPT = os.path.join(
    REPO_ROOT,
    "runs_fusion",
    "routeC_v2_ssim_safe_bs1_acc8",
    "20260416-143257",
    "ckpt",
    "fusion_best_angle.pth",
)

DEFAULT_SOPD_NOPOSE_CKPT = os.path.join(
    REPO_ROOT,
    "runs_fusion",
    "routeC_v2_ssim_nopose_safe_bs1_acc8",
    "20260416-151011",
    "ckpt",
    "fusion_best_angle.pth",
)


@dataclass
class CFG:
    rgb_dir: str = os.environ.get("RGB_DIR", r"E:\ALL\RGB")
    ir_dir: str = os.environ.get("IR_DIR", r"E:\ALL\IR")
    xlsx: str = os.environ.get("XLSX", r"E:\ALL\All_Poses_Merged.xlsx")

    train_ratio: float = float(os.environ.get("TRAIN_RATIO", "0.9"))
    split_seed: int = int(os.environ.get("SPLIT_SEED", "3407"))
    resize_h: int = int(os.environ.get("RESIZE_H", "480"))
    resize_w: int = int(os.environ.get("RESIZE_W", "640"))

    gpu_id: int = int(os.environ.get("GPU_ID", "0"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "8"))
    num_workers: int = int(os.environ.get("NUM_WORKERS", "0"))

    pose_epochs: int = int(os.environ.get("POSE_EPOCHS", "12"))
    lr_pose: float = float(os.environ.get("LR_POSE", "5e-4"))
    seed: int = int(os.environ.get("SEED", "3407"))

    seafusion_ckpt: str = os.environ.get("SEAFUSION_CKPT", DEFAULT_SEAFUSION_CKPT).strip()
    sopd_ckpt: str = os.environ.get("SOPD_CKPT", DEFAULT_SOPD_CKPT).strip()
    sopd_nopose_ckpt: str = os.environ.get("SOPD_NOPOSE_CKPT", DEFAULT_SOPD_NOPOSE_CKPT).strip()
    out_root: str = os.environ.get("OUT_ROOT", r"./runs_pose_unified")
    tag: str = os.environ.get("TAG", "paper_pose_table")
    methods: str = os.environ.get("METHODS", "visible,infrared,seafusion,sopd,sopd_nopose")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_root(out_root: str, tag: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(os.path.join(out_root, tag, ts))


def setup_logger(run_root: str) -> logging.Logger:
    ensure_dir(run_root)
    logger = logging.getLogger("pose_unified")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(run_root, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> CFG:
    cfg = CFG()
    ap = argparse.ArgumentParser(description="Unified downstream pose evaluation table")
    ap.add_argument("--rgb-dir", default=cfg.rgb_dir)
    ap.add_argument("--ir-dir", default=cfg.ir_dir)
    ap.add_argument("--xlsx", default=cfg.xlsx)
    ap.add_argument("--train-ratio", type=float, default=cfg.train_ratio)
    ap.add_argument("--split-seed", type=int, default=cfg.split_seed)
    ap.add_argument("--resize-h", type=int, default=cfg.resize_h)
    ap.add_argument("--resize-w", type=int, default=cfg.resize_w)
    ap.add_argument("--gpu-id", type=int, default=cfg.gpu_id)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--num-workers", type=int, default=cfg.num_workers)
    ap.add_argument("--pose-epochs", type=int, default=cfg.pose_epochs)
    ap.add_argument("--lr-pose", type=float, default=cfg.lr_pose)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--seafusion-ckpt", default=cfg.seafusion_ckpt)
    ap.add_argument("--sopd-ckpt", default=cfg.sopd_ckpt)
    ap.add_argument("--sopd-nopose-ckpt", default=cfg.sopd_nopose_ckpt)
    ap.add_argument("--out-root", default=cfg.out_root)
    ap.add_argument("--tag", default=cfg.tag)
    ap.add_argument(
        "--methods",
        default=cfg.methods,
        help="comma-separated from: visible,infrared,seafusion,sopd,sopd_nopose",
    )
    ns = ap.parse_args()
    return CFG(**vars(ns))


def lookup_pose_batch(names, pose_map: Dict[str, np.ndarray], device: torch.device):
    names_list = list(names)
    batch = len(names_list)
    pose = np.zeros((batch, 4), dtype=np.float32)
    mask = np.zeros((batch, 1), dtype=np.float32)
    for i, name in enumerate(names_list):
        gt = lookup_pose_one(name, pose_map)
        if gt is not None:
            pose[i] = gt
            mask[i, 0] = 1.0
    hit = float(mask.mean()) if batch > 0 else 0.0
    pose_t = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
    return pose_t, mask_t, hit


def quat_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).clamp(-1.0, 1.0)
    return torch.mean(1.0 - dot * dot)


def build_method_specs(cfg: CFG) -> List[dict]:
    specs = {
        "visible": {
            "id": "visible",
            "name": "Visible only",
            "kind": "visible",
        },
        "infrared": {
            "id": "infrared",
            "name": "Infrared only",
            "kind": "infrared",
        },
        "seafusion": {
            "id": "seafusion",
            "name": "SeAFusion",
            "kind": "fusion",
            "type": "fusionnet",
            "fusion_ckpt": cfg.seafusion_ckpt,
        },
        "sopd": {
            "id": "sopd",
            "name": "SoPD-Net",
            "kind": "fusion",
            "type": "gate",
            "fusion_ckpt": cfg.sopd_ckpt,
            "use_rectify": False,
            "rectify_alpha": 0.35,
            "rectify_detach_sal": True,
        },
        "sopd_nopose": {
            "id": "sopd_nopose",
            "name": "SoPD-Net w/o pose loss",
            "kind": "fusion",
            "type": "gate",
            "fusion_ckpt": cfg.sopd_nopose_ckpt,
            "use_rectify": False,
            "rectify_alpha": 0.35,
            "rectify_detach_sal": True,
        },
    }

    wanted = [x.strip().lower() for x in cfg.methods.split(",") if x.strip()]
    methods = []
    for key in wanted:
        if key not in specs:
            raise ValueError(f"Unknown method id: {key}")
        methods.append(specs[key])
    if not methods:
        raise RuntimeError("No methods selected.")
    return methods


def build_method_input(method: dict, model, vis: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    kind = method["kind"]
    if kind == "visible":
        return vis
    if kind == "infrared":
        return ir.repeat(1, 3, 1, 1)

    y, cb, cr = RGB2YCrCb(vis)
    if method["type"] == "fusionnet":
        fused_y = model(y, ir)
    elif method["type"] == "gate":
        sal = joint_saliency(y, ir, expand_k=9, thresh=0.55)
        fused_y, _, _ = model(y, ir, sal=sal)
    else:
        raise ValueError(f"Unknown method type: {method.get('type')}")
    return YCbCr2RGB(fused_y, cb, cr)


@torch.no_grad()
def evaluate_epoch(
    method: dict,
    model,
    pose_net: torch.nn.Module,
    loader: DataLoader,
    pose_map: Dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    pose_net.eval()
    all_ang = []
    val_hits = []
    val_loss_sum = 0.0
    val_loss_count = 0
    for vis, ir, names in loader:
        vis = vis.to(device, non_blocking=True)
        ir = ir.to(device, non_blocking=True)
        pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
        valid = pose_mask.view(-1) > 0.5
        if int(valid.sum().item()) == 0:
            continue
        fused_in = build_method_input(method, model, vis, ir)
        pred = pose_net(fused_in)
        vloss = quat_loss(pred[valid], pose_gt[valid])
        ang = quat_angle_deg(pred[valid], pose_gt[valid])
        all_ang.append(ang.detach().cpu().numpy())
        val_hits.append(hit)
        n_valid = int(valid.sum().item())
        val_loss_sum += float(vloss.detach().cpu()) * n_valid
        val_loss_count += n_valid

    angle_mean = angle_med = angle_p95 = float("nan")
    if all_ang:
        arr = np.concatenate(all_ang, axis=0)
        angle_mean = float(np.mean(arr))
        angle_med = float(np.median(arr))
        angle_p95 = float(np.percentile(arr, 95))
    return {
        "val_angle_mean_deg": angle_mean,
        "val_angle_med_deg": angle_med,
        "val_angle_p95_deg": angle_p95,
        "val_pose_loss": (val_loss_sum / val_loss_count) if val_loss_count > 0 else float("nan"),
        "val_hit": float(np.mean(val_hits)) if val_hits else float("nan"),
    }


def train_pose_head_for_method(
    method: dict,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pose_map: Dict[str, np.ndarray],
    device: torch.device,
    cfg: CFG,
    out_dir: str,
    logger: logging.Logger,
) -> dict:
    pose_net = PoseRegressorPlain(out_dim=4, in_ch=3).to(device)
    opt = torch.optim.Adam(pose_net.parameters(), lr=cfg.lr_pose)

    rows = []
    best_mean_row: Optional[dict] = None
    best_loss_row: Optional[dict] = None
    model_name = method["name"]
    best_mean_ckpt = os.path.join(out_dir, f"{method['id']}_pose_best_mean.pth")
    best_loss_ckpt = os.path.join(out_dir, f"{method['id']}_pose_best_loss.pth")

    for epoch in range(1, cfg.pose_epochs + 1):
        pose_net.train()
        train_losses = []
        train_hits = []

        pbar = tqdm(train_loader, desc=f"{model_name} PoseTrain ep{epoch}/{cfg.pose_epochs}")
        for vis, ir, names in pbar:
            vis = vis.to(device, non_blocking=True)
            ir = ir.to(device, non_blocking=True)
            pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
            valid = pose_mask.view(-1) > 0.5
            if int(valid.sum().item()) == 0:
                continue

            with torch.no_grad():
                fused_in = build_method_input(method, model, vis, ir)

            pred = pose_net(fused_in)
            loss = quat_loss(pred[valid], pose_gt[valid])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            train_hits.append(hit)
            pbar.set_postfix_str(
                f"loss={np.mean(train_losses):.4f} hit={np.mean(train_hits):.3f}"
            )

        rec = {
            "method": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "train_hit": float(np.mean(train_hits)) if train_hits else float("nan"),
        }
        rec.update(evaluate_epoch(method, model, pose_net, val_loader, pose_map, device))
        rows.append(rec)

        logger.info(
            f"[{model_name}] ep{epoch}: "
            f"train_loss={rec['train_loss']:.4f} train_hit={rec['train_hit']:.3f} | "
            f"val mean/med/p95={rec['val_angle_mean_deg']:.2f}/{rec['val_angle_med_deg']:.2f}/{rec['val_angle_p95_deg']:.2f} deg | "
            f"val_pose_loss={rec['val_pose_loss']:.6f} | "
            f"val_hit={rec['val_hit']:.3f}"
        )

        if not np.isnan(rec["val_angle_mean_deg"]):
            if best_mean_row is None or rec["val_angle_mean_deg"] < best_mean_row["val_angle_mean_deg"]:
                best_mean_row = dict(rec)
                torch.save({"pose_net": pose_net.state_dict(), "epoch": epoch, "record": dict(rec)}, best_mean_ckpt)
                logger.info(f"[CKPT] {model_name} best_mean pose judge saved: {best_mean_ckpt}")

        if not np.isnan(rec["val_pose_loss"]):
            if best_loss_row is None or rec["val_pose_loss"] < best_loss_row["val_pose_loss"]:
                best_loss_row = dict(rec)
                torch.save({"pose_net": pose_net.state_dict(), "epoch": epoch, "record": dict(rec)}, best_loss_ckpt)
                logger.info(f"[CKPT] {model_name} best_loss pose judge saved: {best_loss_ckpt}")

    curve_csv = os.path.join(out_dir, f"pose_curve_{method['id']}.csv")
    with open(curve_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if best_mean_row is None:
        raise RuntimeError(f"No valid pose metrics for method: {model_name}")
    if best_loss_row is None:
        best_loss_row = dict(best_mean_row)
        torch.save(
            {"pose_net": pose_net.state_dict(), "epoch": int(best_mean_row["epoch"]), "record": dict(best_mean_row)},
            best_loss_ckpt,
        )
        logger.info(f"[CKPT] {model_name} best_loss fallback pose judge saved: {best_loss_ckpt}")

    return {
        "best_mean": best_mean_row,
        "best_loss": best_loss_row,
        "best_mean_ckpt": best_mean_ckpt,
        "best_loss_ckpt": best_loss_ckpt,
    }


def export_markdown_table(rows: List[dict], out_path: str) -> None:
    metric_keys = ["Mean Error", "Median Error", "P95 Error"]
    mins = {
        "Mean Error": min(r["Mean Error"] for r in rows),
        "Median Error": min(r["Median Error"] for r in rows),
        "P95 Error": min(r["P95 Error"] for r in rows),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| Method | Mean Error ↓ | Median Error ↓ | P95 Error ↓ |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in rows:
            vals = []
            for k in metric_keys:
                v = r[k]
                txt = f"{v:.4f}"
                if abs(v - mins[k]) < 1e-12:
                    txt = f"**{txt}**"
                vals.append(txt)
            f.write(f"| {r['Method']} | {vals[0]} | {vals[1]} | {vals[2]} |\n")


def main():
    cfg = parse_args()
    run_root = make_run_root(cfg.out_root, cfg.tag)
    ensure_dir(run_root)
    logger = setup_logger(run_root)
    set_seed(cfg.seed)

    logger.info(f"RUN_ROOT: {run_root}")
    logger.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    logger.info(f"[CKPT] SeAFusion ckpt = {cfg.seafusion_ckpt}")
    logger.info(f"[CKPT] SoPD ckpt      = {cfg.sopd_ckpt}")
    logger.info(f"[CKPT] SoPD(no-pose)  = {cfg.sopd_nopose_ckpt}")

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_ds = PairFolderDataset(
        cfg.rgb_dir,
        cfg.ir_dir,
        "train",
        (cfg.resize_h, cfg.resize_w),
        cfg.split_seed,
        cfg.train_ratio,
    )
    val_ds = PairFolderDataset(
        cfg.rgb_dir,
        cfg.ir_dir,
        "val",
        (cfg.resize_h, cfg.resize_w),
        cfg.split_seed,
        cfg.train_ratio,
    )

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples  : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    pose_map = load_pose_map_from_excel(cfg.xlsx, logger)
    logger.info(f"[POSE] pose_map_keys={len(pose_map)}")

    methods = build_method_specs(cfg)
    for method in methods:
        if method["kind"] == "fusion":
            logger.info(f"[METHOD] {method['name']} fusion_ckpt = {method['fusion_ckpt']}")

    best_rows = []
    judge_manifest_rows = []
    for method in methods:
        logger.info(f"[METHOD] {method['name']}")
        model = None
        if method["kind"] == "fusion":
            model = _load_fusion_model(method, device, logger)

        out_dir = os.path.join(run_root, method["id"])
        ensure_dir(out_dir)

        best = train_pose_head_for_method(
            method=method,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            pose_map=pose_map,
            device=device,
            cfg=cfg,
            out_dir=out_dir,
            logger=logger,
        )

        best_mean = best["best_mean"]
        best_loss = best["best_loss"]
        best_rows.append(
            {
                "Method": method["name"],
                "Mean Error": float(best_mean["val_angle_mean_deg"]),
                "Median Error": float(best_mean["val_angle_med_deg"]),
                "P95 Error": float(best_mean["val_angle_p95_deg"]),
            }
        )
        judge_manifest_rows.append(
            {
                "method_id": method["id"],
                "method_name": method["name"],
                "pose_judge_ckpt_best_mean": os.path.abspath(best["best_mean_ckpt"]),
                "pose_judge_ckpt_best_loss": os.path.abspath(best["best_loss_ckpt"]),
                "fusion_ckpt": method.get("fusion_ckpt", None),
                "best_epoch_best_mean": int(best_mean["epoch"]),
                "best_epoch_best_loss": int(best_loss["epoch"]),
                "run_config": asdict(cfg),
            }
        )

    table_csv = os.path.join(run_root, "paper_pose_table.csv")
    with open(table_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "Mean Error", "Median Error", "P95 Error"],
        )
        writer.writeheader()
        writer.writerows(best_rows)

    export_markdown_table(best_rows, os.path.join(run_root, "paper_pose_table.md"))
    logger.info(f"[DONE] paper table csv: {table_csv}")
    logger.info(f"[DONE] paper table md : {os.path.join(run_root, 'paper_pose_table.md')}")
    manifest_path = os.path.join(run_root, "pose_judge_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"run_root": run_root, "methods": judge_manifest_rows}, f, ensure_ascii=False, indent=2)
    logger.info(f"[DONE] pose judge manifest: {manifest_path}")


if __name__ == "__main__":
    main()
