#!/usr/bin/env python3
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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from doctor import PoseRegressorPlain, _load_fusion_model, joint_saliency, quat_angle_deg
from utils import RGB2YCrCb, YCbCr2RGB


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIRED_MANIFEST_FIELDS = [
    "satellite_name",
    "frame_idx",
    "visible_path",
    "infrared_path",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "difficulty_reason_visible",
    "difficulty_reason_infrared",
    "subset_name",
]

SUBSET_BINS: List[Tuple[str, int, int]] = [
    ("subset_1", 1, 50),
    ("subset_2", 51, 100),
    ("subset_3", 101, 150),
    ("subset_4", 151, 200),
    ("subset_5", 201, 250),
]

MAIN_GROUP_BINS: List[Tuple[str, int, int]] = [
    ("illumination_hard", 1, 150),
    ("hard_all", 1, 200),
    ("normal_ref", 201, 250),
]


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
    train_csv: str = os.environ.get("FIRE_TRAIN_CSV", os.path.join(REPO_ROOT, "fire_train.csv"))
    val_csv: str = os.environ.get("FIRE_VAL_CSV", os.path.join(REPO_ROOT, "fire_val.csv"))
    test_csv: str = os.environ.get("FIRE_TEST_HARD_CSV", os.path.join(REPO_ROOT, "fire_test_hard.csv"))

    resize_h: int = int(os.environ.get("RESIZE_H", "480"))
    resize_w: int = int(os.environ.get("RESIZE_W", "640"))
    gpu_id: int = int(os.environ.get("GPU_ID", "0"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "8"))
    num_workers: int = int(os.environ.get("NUM_WORKERS", "0"))
    safe_runtime: bool = os.environ.get("SAFE_RUNTIME", "0") == "1"
    pose_epochs: int = int(os.environ.get("POSE_EPOCHS", "12"))
    lr_pose: float = float(os.environ.get("LR_POSE", "5e-4"))
    seed: int = int(os.environ.get("SEED", "3407"))

    seafusion_ckpt: str = os.environ.get("SEAFUSION_CKPT", DEFAULT_SEAFUSION_CKPT).strip()
    sopd_ckpt: str = os.environ.get("SOPD_CKPT", DEFAULT_SOPD_CKPT).strip()
    sopd_nopose_ckpt: str = os.environ.get("SOPD_NOPOSE_CKPT", DEFAULT_SOPD_NOPOSE_CKPT).strip()
    methods: str = os.environ.get("METHODS", "visible,infrared,seafusion,sopd,sopd_nopose")
    pose_judge_select: str = os.environ.get("POSE_JUDGE_SELECT", "best_mean")

    out_root: str = os.environ.get("OUT_ROOT", r"./runs_pose_unified")
    tag: str = os.environ.get("TAG", "fire_domain_pose_train_eval")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_root(out_root: str, tag: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(os.path.join(out_root, tag, ts))


def setup_logger(run_root: str) -> logging.Logger:
    ensure_dir(run_root)
    logger = logging.getLogger("fire_domain_pose")
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


def set_seed(seed: int, safe_runtime: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if safe_runtime:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> CFG:
    cfg = CFG()
    ap = argparse.ArgumentParser(description="Same-domain fire pose judge train+test-only evaluation")
    ap.add_argument("--train-csv", default=cfg.train_csv)
    ap.add_argument("--val-csv", default=cfg.val_csv)
    ap.add_argument("--test-csv", default=cfg.test_csv)
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
    ap.add_argument(
        "--methods",
        default=cfg.methods,
        help="comma-separated from: visible,infrared,seafusion,sopd,sopd_nopose",
    )
    ap.add_argument(
        "--pose-judge-select",
        default=cfg.pose_judge_select,
        help="one of: best_mean,best_loss; choose which fixed judge to use for test-only eval",
    )
    ap.add_argument("--out-root", default=cfg.out_root)
    ap.add_argument("--tag", default=cfg.tag)
    ns = ap.parse_args()
    return CFG(**vars(ns))


def _safe_int(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _subset_name_from_idx(frame_idx: int) -> str:
    for name, lo, hi in SUBSET_BINS:
        if lo <= frame_idx <= hi:
            return name
    return ""


def _resolve_path(path_text: str, csv_path: str) -> str:
    p = str(path_text).strip()
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(csv_path)), p))


def load_manifest_rows(csv_path: str) -> List[dict]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        got = reader.fieldnames or []
        missing = [k for k in REQUIRED_MANIFEST_FIELDS if k not in got]
        if missing:
            raise RuntimeError(f"Manifest missing required fields: {missing}. file={csv_path}")
        for r in reader:
            frame_idx = _safe_int(r.get("frame_idx"))
            if frame_idx is None:
                continue
            q = [
                _safe_float(r.get("quat_w")),
                _safe_float(r.get("quat_x")),
                _safe_float(r.get("quat_y")),
                _safe_float(r.get("quat_z")),
            ]
            if any(v is None for v in q):
                continue
            quat = np.asarray(q, dtype=np.float32)
            norm = float(np.linalg.norm(quat))
            if norm <= 1e-12:
                continue
            quat = quat / norm

            visible_path = _resolve_path(str(r.get("visible_path", "")), csv_path)
            infrared_path = _resolve_path(str(r.get("infrared_path", "")), csv_path)
            if not os.path.isfile(visible_path):
                raise FileNotFoundError(f"visible_path not found: {visible_path} (from {csv_path})")
            if not os.path.isfile(infrared_path):
                raise FileNotFoundError(f"infrared_path not found: {infrared_path} (from {csv_path})")

            subset_name = str(r.get("subset_name", "")).strip()
            if not subset_name:
                subset_name = _subset_name_from_idx(frame_idx)

            rows.append(
                {
                    "satellite_name": str(r.get("satellite_name", "")).strip(),
                    "frame_idx": frame_idx,
                    "visible_path": visible_path,
                    "infrared_path": infrared_path,
                    "quat": quat,
                    "subset_name": subset_name,
                }
            )
    if not rows:
        raise RuntimeError(f"No valid rows loaded from manifest: {csv_path}")
    return rows


class FireManifestPoseDataset(Dataset):
    def __init__(self, rows: Sequence[dict], resize_hw: Tuple[int, int]):
        super().__init__()
        self.rows = list(rows)
        self.resize_hw = resize_hw

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _read_rgb(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise RuntimeError(f"Failed to read file bytes: {path}")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imdecode failed: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = resize_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    @staticmethod
    def _read_ir(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise RuntimeError(f"Failed to read file bytes: {path}")
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cv2.imdecode failed: {path}")
        h, w = resize_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        arr = (img.astype(np.float32) / 255.0)[None, ...]
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        vis = self._read_rgb(row["visible_path"], self.resize_hw)
        ir = self._read_ir(row["infrared_path"], self.resize_hw)
        quat = torch.from_numpy(np.asarray(row["quat"], dtype=np.float32))
        return vis, ir, quat, int(row["frame_idx"]), str(row.get("subset_name", ""))


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


def quat_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).clamp(-1.0, 1.0)
    return torch.mean(1.0 - dot * dot)


@torch.no_grad()
def evaluate_loader(
    method: dict,
    model,
    pose_net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    safe_runtime: bool = False,
) -> dict:
    pose_net.eval()
    all_ang = []
    val_loss_sum = 0.0
    val_loss_count = 0
    for vis, ir, gt, _frame_idx, _subset_name in loader:
        vis = vis.to(device, non_blocking=(not safe_runtime))
        ir = ir.to(device, non_blocking=(not safe_runtime))
        gt = gt.to(device, non_blocking=(not safe_runtime))
        x = build_method_input(method, model, vis, ir)
        pred = pose_net(x)
        vloss = quat_loss(pred, gt)
        ang = quat_angle_deg(pred, gt)
        all_ang.append(ang.detach().cpu().numpy())
        val_loss_sum += float(vloss.detach().cpu()) * int(gt.shape[0])
        val_loss_count += int(gt.shape[0])

    if not all_ang:
        return {
            "val_angle_mean_deg": float("nan"),
            "val_angle_med_deg": float("nan"),
            "val_angle_p95_deg": float("nan"),
            "val_pose_loss": float("nan"),
        }
    arr = np.concatenate(all_ang, axis=0)
    return {
        "val_angle_mean_deg": float(np.mean(arr)),
        "val_angle_med_deg": float(np.median(arr)),
        "val_angle_p95_deg": float(np.percentile(arr, 95)),
        "val_pose_loss": (val_loss_sum / val_loss_count) if val_loss_count > 0 else float("nan"),
    }


def train_pose_head_for_method(
    method: dict,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
        pbar = tqdm(train_loader, desc=f"{model_name} PoseTrain ep{epoch}/{cfg.pose_epochs}")
        for vis, ir, gt, _frame_idx, _subset_name in pbar:
            vis = vis.to(device, non_blocking=(not cfg.safe_runtime))
            ir = ir.to(device, non_blocking=(not cfg.safe_runtime))
            gt = gt.to(device, non_blocking=(not cfg.safe_runtime))

            with torch.no_grad():
                fused_in = build_method_input(method, model, vis, ir)
            pred = pose_net(fused_in)
            loss = quat_loss(pred, gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            pbar.set_postfix_str(f"loss={np.mean(train_losses):.4f}")

        rec = {
            "method": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
        }
        rec.update(evaluate_loader(method, model, pose_net, val_loader, device, safe_runtime=cfg.safe_runtime))
        rows.append(rec)

        logger.info(
            f"[{model_name}] ep{epoch}: "
            f"train_loss={rec['train_loss']:.4f} | "
            f"val mean/med/p95={rec['val_angle_mean_deg']:.2f}/{rec['val_angle_med_deg']:.2f}/{rec['val_angle_p95_deg']:.2f} deg | "
            f"val_pose_loss={rec['val_pose_loss']:.6f}"
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
        "curve_csv": curve_csv,
    }


def load_pose_judge_model(ckpt_path: str, device: torch.device) -> PoseRegressorPlain:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pose judge ckpt not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "pose_net" in payload:
        state = payload["pose_net"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise RuntimeError(f"Unsupported pose judge ckpt format: {ckpt_path}")
    net = PoseRegressorPlain(out_dim=4, in_ch=3).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


@torch.no_grad()
def collect_test_angles(
    method: dict,
    model,
    pose_net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    safe_runtime: bool = False,
) -> List[dict]:
    pose_net.eval()
    rows: List[dict] = []
    for vis, ir, gt, frame_idx, subset_name in tqdm(loader, desc=f"Test {method['name']}"):
        vis = vis.to(device, non_blocking=(not safe_runtime))
        ir = ir.to(device, non_blocking=(not safe_runtime))
        gt = gt.to(device, non_blocking=(not safe_runtime))

        x = build_method_input(method, model, vis, ir)
        pred = pose_net(x)
        ang = quat_angle_deg(pred, gt).detach().cpu().numpy()

        frame_np = frame_idx.detach().cpu().numpy()
        subset_list = list(subset_name)
        for i in range(len(ang)):
            idx = int(frame_np[i])
            sname = str(subset_list[i]) if subset_list[i] else _subset_name_from_idx(idx)
            rows.append(
                {
                    "frame_idx": idx,
                    "subset_name": sname,
                    "angle_deg": float(ang[i]),
                }
            )
    return rows


def _group_metrics(rows: Sequence[dict], group_name: str) -> dict:
    vals = np.asarray([float(r["angle_deg"]) for r in rows], dtype=np.float64)
    if vals.size == 0:
        return {
            "Group": group_name,
            "Mean Error": float("nan"),
            "Median Error": float("nan"),
            "P95 Error": float("nan"),
            "N": 0,
        }
    return {
        "Group": group_name,
        "Mean Error": float(np.mean(vals)),
        "Median Error": float(np.median(vals)),
        "P95 Error": float(np.percentile(vals, 95)),
        "N": int(vals.size),
    }


def build_main_group_rows(angle_rows: Sequence[dict], method_name: str) -> List[dict]:
    out: List[dict] = []
    for group_name, lo, hi in MAIN_GROUP_BINS:
        grp = [r for r in angle_rows if lo <= int(r["frame_idx"]) <= hi]
        m = _group_metrics(grp, group_name)
        m["Method"] = method_name
        out.append(m)
    return out


def build_breakdown_rows(angle_rows: Sequence[dict], method_name: str) -> List[dict]:
    out: List[dict] = []
    for subset_name, lo, hi in SUBSET_BINS:
        grp = [
            r
            for r in angle_rows
            if (str(r.get("subset_name", "")).strip() == subset_name) or (lo <= int(r["frame_idx"]) <= hi)
        ]
        m = _group_metrics(grp, subset_name)
        m["Method"] = method_name
        out.append(m)
    return out


def write_table_csv(rows: Sequence[dict], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Group", "Method", "Mean Error", "Median Error", "P95 Error", "N"])
        writer.writeheader()
        writer.writerows(rows)


def write_table_md(rows: Sequence[dict], out_md: str) -> None:
    ensure_dir(os.path.dirname(out_md))
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| Group | Method | Mean Error | Median Error | P95 Error | N |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['Group']} | {r['Method']} | "
                f"{float(r['Mean Error']):.4f} | {float(r['Median Error']):.4f} | {float(r['P95 Error']):.4f} | {int(r['N'])} |\n"
            )


def main() -> None:
    cfg = parse_args()
    if cfg.pose_judge_select not in {"best_mean", "best_loss"}:
        raise ValueError(f"pose_judge_select must be best_mean or best_loss, got: {cfg.pose_judge_select}")
    if cfg.safe_runtime:
        cfg.num_workers = 0

    run_root = make_run_root(cfg.out_root, cfg.tag)
    ensure_dir(run_root)
    logger = setup_logger(run_root)
    set_seed(cfg.seed, cfg.safe_runtime)

    logger.info(f"RUN_ROOT: {run_root}")
    logger.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_rows = load_manifest_rows(cfg.train_csv)
    val_rows = load_manifest_rows(cfg.val_csv)
    test_rows = load_manifest_rows(cfg.test_csv)
    logger.info(f"Train rows: {len(train_rows)}")
    logger.info(f"Val rows  : {len(val_rows)}")
    logger.info(f"Test rows : {len(test_rows)}")

    train_ds = FireManifestPoseDataset(train_rows, (cfg.resize_h, cfg.resize_w))
    val_ds = FireManifestPoseDataset(val_rows, (cfg.resize_h, cfg.resize_w))
    test_ds = FireManifestPoseDataset(test_rows, (cfg.resize_h, cfg.resize_w))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(not cfg.safe_runtime),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(not cfg.safe_runtime),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, cfg.batch_size // 2),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(not cfg.safe_runtime),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    methods = build_method_specs(cfg)
    fusion_models: Dict[str, Optional[torch.nn.Module]] = {}
    for method in methods:
        if method["kind"] == "fusion":
            logger.info(f"[METHOD] {method['name']} fusion_ckpt = {method['fusion_ckpt']}")
            fusion_models[method["id"]] = _load_fusion_model(method, device, logger)
        else:
            fusion_models[method["id"]] = None

    judge_manifest_rows = []
    trainval_rows = []
    judge_ckpt_selected: Dict[str, str] = {}

    for method in methods:
        logger.info(f"[TRAIN] {method['name']}")
        out_dir = os.path.join(run_root, method["id"])
        ensure_dir(out_dir)

        best = train_pose_head_for_method(
            method=method,
            model=fusion_models[method["id"]],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            out_dir=out_dir,
            logger=logger,
        )

        best_mean = best["best_mean"]
        best_loss = best["best_loss"]
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
        judge_ckpt_selected[method["id"]] = os.path.abspath(
            best["best_mean_ckpt"] if cfg.pose_judge_select == "best_mean" else best["best_loss_ckpt"]
        )
        trainval_rows.append(
            {
                "Method": method["name"],
                "Mean Error": float(best_mean["val_angle_mean_deg"]),
                "Median Error": float(best_mean["val_angle_med_deg"]),
                "P95 Error": float(best_mean["val_angle_p95_deg"]),
            }
        )

    trainval_csv = os.path.join(run_root, "fire_domain_trainval_pose_table.csv")
    with open(trainval_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Mean Error", "Median Error", "P95 Error"])
        writer.writeheader()
        writer.writerows(trainval_rows)
    logger.info(f"[DONE] train/val table: {trainval_csv}")

    manifest_path = os.path.join(run_root, "fire_domain_pose_judge_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_root": run_root,
                "judge_select_for_test": cfg.pose_judge_select,
                "train_csv": os.path.abspath(cfg.train_csv),
                "val_csv": os.path.abspath(cfg.val_csv),
                "test_csv": os.path.abspath(cfg.test_csv),
                "methods": judge_manifest_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[DONE] pose judge manifest: {manifest_path}")

    main_rows: List[dict] = []
    breakdown_rows: List[dict] = []
    for method in methods:
        logger.info(f"[TEST-ONLY] {method['name']} using {cfg.pose_judge_select}")
        judge = load_pose_judge_model(judge_ckpt_selected[method["id"]], device=device)
        angle_rows = collect_test_angles(
            method=method,
            model=fusion_models[method["id"]],
            pose_net=judge,
            loader=test_loader,
            device=device,
            safe_runtime=cfg.safe_runtime,
        )
        main_rows.extend(build_main_group_rows(angle_rows, method["name"]))
        breakdown_rows.extend(build_breakdown_rows(angle_rows, method["name"]))

    main_csv = os.path.join(run_root, "fire_domain_pose_main_table.csv")
    main_md = os.path.join(run_root, "fire_domain_pose_main_table.md")
    write_table_csv(main_rows, main_csv)
    write_table_md(main_rows, main_md)
    logger.info(f"[DONE] main table csv: {main_csv}")
    logger.info(f"[DONE] main table md : {main_md}")

    breakdown_csv = os.path.join(run_root, "fire_domain_pose_breakdown_table.csv")
    breakdown_md = os.path.join(run_root, "fire_domain_pose_breakdown_table.md")
    write_table_csv(breakdown_rows, breakdown_csv)
    write_table_md(breakdown_rows, breakdown_md)
    logger.info(f"[DONE] breakdown table csv: {breakdown_csv}")
    logger.info(f"[DONE] breakdown table md : {breakdown_md}")


if __name__ == "__main__":
    main()
