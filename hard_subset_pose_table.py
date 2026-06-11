#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from doctor import PoseRegressorPlain, _load_fusion_model, joint_saliency, quat_angle_deg
from utils import RGB2YCrCb, YCbCr2RGB


SUBSETS: List[Tuple[str, int, int]] = [
    ("subset_1", 1, 50),
    ("subset_2", 51, 100),
    ("subset_3", 101, 150),
    ("subset_4", 151, 200),
    ("subset_5", 201, 250),
]

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEAFUSION_CKPT = os.path.join(
    REPO_ROOT,
    "runs_fusion",
    "baseline",
    "20251228-234201",
    "ckpt",
    "fusion_best_total.pth",
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
    visible_root: str = os.environ.get("VISIBLE_ROOT", r"D:\BaiduNetdiskDownload\fire\Aqua_VisibleBatch250")
    infrared_root: str = os.environ.get("INFRARED_ROOT", r"D:\BaiduNetdiskDownload\fire\Aqua_IRBatch250")
    visible_csv: str = os.environ.get(
        "VISIBLE_CSV",
        r"D:\BaiduNetdiskDownload\fire\Aqua_VisibleBatch250\visible_batch_250.csv",
    )
    infrared_csv: str = os.environ.get(
        "INFRARED_CSV",
        r"D:\BaiduNetdiskDownload\fire\Aqua_IRBatch250\ir_batch_250.csv",
    )
    train_ratio: float = float(os.environ.get("TRAIN_RATIO", "0.8"))
    split_seed: int = int(os.environ.get("SPLIT_SEED", "3407"))
    resize_h: int = int(os.environ.get("RESIZE_H", "480"))
    resize_w: int = int(os.environ.get("RESIZE_W", "640"))
    gpu_id: int = int(os.environ.get("GPU_ID", "0"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "8"))
    num_workers: int = int(os.environ.get("NUM_WORKERS", "0"))
    pose_epochs: int = int(os.environ.get("POSE_EPOCHS", "12"))
    lr_pose: float = float(os.environ.get("LR_POSE", "5e-4"))
    seed: int = int(os.environ.get("SEED", "3407"))
    safe_runtime: bool = os.environ.get("SAFE_RUNTIME", "0") == "1"
    pose_judge_manifest: str = os.environ.get(
        "POSE_JUDGE_MANIFEST",
        r"D:\Redundancy\edgedownload\SeAFusion-main\runs_pose_unified\paper_pose_table\20260421-204807\pose_judge_manifest.json",
    )
    pose_judge_select: str = os.environ.get("POSE_JUDGE_SELECT", "best_mean")
    seafusion_ckpt: str = os.environ.get("SEAFUSION_CKPT", DEFAULT_SEAFUSION_CKPT).strip()
    sopd_ckpt: str = os.environ.get("SOPD_CKPT", DEFAULT_SOPD_CKPT).strip()
    sopd_nopose_ckpt: str = os.environ.get("SOPD_NOPOSE_CKPT", DEFAULT_SOPD_NOPOSE_CKPT).strip()
    methods: str = os.environ.get("METHODS", "visible,infrared,seafusion,sopd,sopd_nopose")
    out_root: str = os.environ.get("OUT_ROOT", r"./runs_pose_unified")
    tag: str = os.environ.get("TAG", "hard_subset_pose_table")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_root(out_root: str, tag: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(os.path.join(out_root, tag, ts))


def setup_logger(run_root: str) -> logging.Logger:
    ensure_dir(run_root)
    logger = logging.getLogger("hard_subset_pose")
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
    ap = argparse.ArgumentParser(description="Hard-subset downstream pose comparison (visible vs infrared)")
    ap.add_argument("--visible-root", default=cfg.visible_root)
    ap.add_argument("--infrared-root", default=cfg.infrared_root)
    ap.add_argument("--visible-csv", default=cfg.visible_csv)
    ap.add_argument("--infrared-csv", default=cfg.infrared_csv)
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
    ap.add_argument("--pose-judge-manifest", default=cfg.pose_judge_manifest)
    ap.add_argument(
        "--pose-judge-select",
        default=cfg.pose_judge_select,
        help="one of: best_mean,best_loss",
    )
    ap.add_argument("--seafusion-ckpt", default=cfg.seafusion_ckpt)
    ap.add_argument("--sopd-ckpt", default=cfg.sopd_ckpt)
    ap.add_argument("--sopd-nopose-ckpt", default=cfg.sopd_nopose_ckpt)
    ap.add_argument(
        "--methods",
        default=cfg.methods,
        help="comma-separated from: visible,infrared,seafusion,sopd,sopd_nopose",
    )
    ap.add_argument("--out-root", default=cfg.out_root)
    ap.add_argument("--tag", default=cfg.tag)
    ns = ap.parse_args()
    return CFG(**vars(ns))


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).strip().lower())


def _find_frame_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    norm_map = {_norm_col(c): c for c in cols}
    for key in ["frameidx", "frameindex", "idx", "frame", "frames", "index"]:
        if key in norm_map:
            return norm_map[key]
    for c in cols:
        n = _norm_col(c)
        if "frame" in n and "idx" in n:
            return c
    raise RuntimeError(f"frame_idx column not found. columns={cols}")


def _extract_int_from_text(x) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    m = re.search(r"(\d+)", s)
    if m is None:
        return None
    return int(m.group(1))


def _find_image_name_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        n = _norm_col(c)
        if any(k in n for k in ["image", "img", "filename", "file", "name", "path"]):
            return c
    return None


def _find_quat_cols(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    norm = {_norm_col(c): c for c in cols}

    direct_sets = [
        ["qw", "qx", "qy", "qz"],
        ["quatw", "quatx", "quaty", "quatz"],
        ["quaternionw", "quaternionx", "quaterniony", "quaternionz"],
    ]
    for ds in direct_sets:
        if all(k in norm for k in ds):
            return [norm[k] for k in ds]

    cand = []
    for c in cols:
        n = _norm_col(c)
        if any(tag in n for tag in ["quat", "quaternion"]):
            cand.append(c)
    if len(cand) >= 4:
        ordered = sorted(cand, key=lambda x: _norm_col(x))
        if len(ordered) >= 4:
            return ordered[:4]

    suffix_map: Dict[str, str] = {}
    for c in cols:
        n = _norm_col(c)
        if n.endswith("w"):
            suffix_map["w"] = c
        if n.endswith("x"):
            suffix_map["x"] = c
        if n.endswith("y"):
            suffix_map["y"] = c
        if n.endswith("z"):
            suffix_map["z"] = c
    if all(k in suffix_map for k in ["w", "x", "y", "z"]):
        return [suffix_map["w"], suffix_map["x"], suffix_map["y"], suffix_map["z"]]

    raise RuntimeError(f"Quaternion columns not found. columns={cols}")


def _resolve_image_path(root: str, frame_idx: int, row: pd.Series) -> str:
    base = os.path.join(root, "images")
    fallback = os.path.join(base, f"{frame_idx:04d}.png")
    if os.path.isfile(fallback):
        return fallback

    # Try CSV-provided filename/path when present.
    row_dict = row.to_dict()
    for key, val in row_dict.items():
        nk = _norm_col(key)
        if not any(k in nk for k in ["image", "img", "filename", "file", "name", "path"]):
            continue
        if val is None:
            continue
        s = str(val).strip()
        if s == "":
            continue
        p = s if os.path.isabs(s) else os.path.join(root, s)
        if os.path.isfile(p):
            return p
        p2 = os.path.join(base, os.path.basename(s))
        if os.path.isfile(p2):
            return p2
    return fallback


def _prepare_one_csv(csv_path: str, root: str, modality: str, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    frame_col = _find_frame_col(df)
    quat_cols = _find_quat_cols(df)
    name_col = _find_image_name_col(df)
    logger.info(
        f"[{modality}] frame_col={frame_col}, quat_cols={quat_cols}, image_col={name_col if name_col else '(none)'}"
    )

    local = df.copy()
    local["_frame_idx"] = local[frame_col].apply(_extract_int_from_text)
    local = local.dropna(subset=["_frame_idx"]).copy()
    local["_frame_idx"] = local["_frame_idx"].astype(int)
    local = local[(local["_frame_idx"] >= 1) & (local["_frame_idx"] <= 250)].copy()

    for c in quat_cols:
        local[c] = pd.to_numeric(local[c], errors="coerce")
    local = local.dropna(subset=quat_cols).copy()

    quat = local[quat_cols].to_numpy(dtype=np.float32)
    n = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8
    quat = quat / n
    local["_qw"] = quat[:, 0]
    local["_qx"] = quat[:, 1]
    local["_qy"] = quat[:, 2]
    local["_qz"] = quat[:, 3]

    # Keep one row per frame_idx.
    local = local.sort_values(by=["_frame_idx"]).drop_duplicates(subset=["_frame_idx"], keep="first")
    local["_image_path"] = local.apply(
        lambda r: _resolve_image_path(root, int(r["_frame_idx"]), r),
        axis=1,
    )
    return local


def _subset_name_from_idx(frame_idx: int) -> Optional[str]:
    for name, lo, hi in SUBSETS:
        if lo <= frame_idx <= hi:
            return name
    return None


def build_paired_manifest(cfg: CFG, logger: logging.Logger) -> List[dict]:
    vis_df = _prepare_one_csv(cfg.visible_csv, cfg.visible_root, "visible", logger)
    ir_df = _prepare_one_csv(cfg.infrared_csv, cfg.infrared_root, "infrared", logger)

    vis_map = {int(r["_frame_idx"]): r for _, r in vis_df.iterrows()}
    ir_map = {int(r["_frame_idx"]): r for _, r in ir_df.iterrows()}
    common_idx = sorted(set(vis_map.keys()).intersection(ir_map.keys()))
    if not common_idx:
        raise RuntimeError("No paired frame_idx overlap found between visible and infrared CSV files.")

    manifest: List[dict] = []
    for idx in common_idx:
        subset_name = _subset_name_from_idx(idx)
        if subset_name is None:
            continue
        vr = vis_map[idx]
        ir = ir_map[idx]
        row = {
            "frame_idx": idx,
            "subset": subset_name,
            "visible_path": str(vr["_image_path"]),
            "infrared_path": str(ir["_image_path"]),
            "visible_quat": np.array([vr["_qw"], vr["_qx"], vr["_qy"], vr["_qz"]], dtype=np.float32),
            "infrared_quat": np.array([ir["_qw"], ir["_qx"], ir["_qy"], ir["_qz"]], dtype=np.float32),
        }
        manifest.append(row)

    if not manifest:
        raise RuntimeError("Paired manifest is empty after subset filtering.")

    # Sanity: ensure all subset bins exist.
    by_subset = {}
    for r in manifest:
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
    for sname, lo, hi in SUBSETS:
        logger.info(f"[MANIFEST] {sname} ({lo}-{hi}): {by_subset.get(sname, 0)} pairs")
    logger.info(f"[MANIFEST] total pairs: {len(manifest)}")
    return manifest


class PairedSubsetPoseDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        method_id: str,
        resize_hw: Tuple[int, int],
    ):
        super().__init__()
        if method_id not in {"visible", "infrared", "seafusion", "sopd", "sopd_nopose"}:
            raise ValueError(f"Unsupported method: {method_id}")
        self.rows = list(rows)
        self.method_id = method_id
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
    def _read_ir_to_rgb(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise RuntimeError(f"Failed to read file bytes: {path}")
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cv2.imdecode failed: {path}")
        h, w = resize_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        arr = (img.astype(np.float32) / 255.0)[None, ...]
        arr = np.repeat(arr, 3, axis=0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        vis = self._read_rgb(row["visible_path"], self.resize_hw)
        ir = self._read_ir_to_rgb(row["infrared_path"], self.resize_hw)[:1, ...]
        if self.method_id == "infrared":
            q = row["infrared_quat"]
        else:
            q = row["visible_quat"]
        q_t = torch.from_numpy(np.asarray(q, dtype=np.float32))
        name = f"{int(row['frame_idx']):04d}.png"
        return vis, ir, q_t, name


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
def evaluate_epoch(
    method: dict,
    model,
    pose_net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    safe_runtime: bool = False,
) -> dict:
    pose_net.eval()
    all_ang = []
    for vis, ir, gt, _names in loader:
        vis = vis.to(device, non_blocking=(not safe_runtime))
        ir = ir.to(device, non_blocking=(not safe_runtime))
        gt = gt.to(device, non_blocking=(not safe_runtime))
        with torch.no_grad():
            x = build_method_input(method, model, vis, ir)
        pred = pose_net(x)
        ang = quat_angle_deg(pred, gt)
        all_ang.append(ang.detach().cpu().numpy())

    if not all_ang:
        return {
            "val_angle_mean_deg": float("nan"),
            "val_angle_med_deg": float("nan"),
            "val_angle_p95_deg": float("nan"),
        }
    arr = np.concatenate(all_ang, axis=0)
    return {
        "val_angle_mean_deg": float(np.mean(arr)),
        "val_angle_med_deg": float(np.median(arr)),
        "val_angle_p95_deg": float(np.percentile(arr, 95)),
    }


def train_pose_head_for_subset(
    method: dict,
    model,
    subset_name: str,
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    device: torch.device,
    cfg: CFG,
    out_dir: str,
    logger: logging.Logger,
) -> dict:
    train_ds = PairedSubsetPoseDataset(train_rows, method["id"], (cfg.resize_h, cfg.resize_w))
    val_ds = PairedSubsetPoseDataset(val_rows, method["id"], (cfg.resize_h, cfg.resize_w))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"{subset_name}/{method['id']} has empty train or val split.")

    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(not cfg.safe_runtime),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(max(1, cfg.batch_size // 2), max(1, len(val_ds))),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(not cfg.safe_runtime),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    pose_net = PoseRegressorPlain(out_dim=4, in_ch=3).to(device)
    opt = torch.optim.Adam(pose_net.parameters(), lr=cfg.lr_pose)
    rows = []
    best_row: Optional[dict] = None

    for epoch in range(1, cfg.pose_epochs + 1):
        pose_net.train()
        train_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"{subset_name}/{method['name']} PoseTrain ep{epoch}/{cfg.pose_epochs}",
        )
        for vis, ir, gt, _names in pbar:
            vis = vis.to(device, non_blocking=(not cfg.safe_runtime))
            ir = ir.to(device, non_blocking=(not cfg.safe_runtime))
            gt = gt.to(device, non_blocking=(not cfg.safe_runtime))

            with torch.no_grad():
                x = build_method_input(method, model, vis, ir)
            pred = pose_net(x)
            loss = quat_loss(pred, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_losses.append(float(loss.detach().cpu()))
            pbar.set_postfix_str(f"loss={np.mean(train_losses):.4f}")

        rec = {
            "subset": subset_name,
            "method": method["name"],
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
        }
        rec.update(
            evaluate_epoch(
                method,
                model,
                pose_net,
                val_loader,
                device,
                safe_runtime=cfg.safe_runtime,
            )
        )
        rows.append(rec)

        logger.info(
            f"[{subset_name}/{method['name']}] ep{epoch}: "
            f"train_loss={rec['train_loss']:.4f} | "
            f"val mean/med/p95={rec['val_angle_mean_deg']:.2f}/{rec['val_angle_med_deg']:.2f}/{rec['val_angle_p95_deg']:.2f} deg"
        )

        if not np.isnan(rec["val_angle_mean_deg"]):
            if best_row is None or rec["val_angle_mean_deg"] < best_row["val_angle_mean_deg"]:
                best_row = dict(rec)

    curve_csv = os.path.join(out_dir, f"pose_curve_{subset_name}_{method['id']}.csv")
    with open(curve_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if best_row is None:
        raise RuntimeError(f"No valid pose metrics for {subset_name}/{method['name']}")
    return best_row


def build_method_specs(cfg: CFG) -> List[dict]:
    specs = {
        "visible": {"id": "visible", "name": "Visible only", "kind": "visible"},
        "infrared": {"id": "infrared", "name": "Infrared only", "kind": "infrared"},
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


def split_rows(rows: Sequence[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    idx = np.arange(len(rows))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(len(rows) * float(train_ratio))
    n_train = min(max(1, n_train), len(rows) - 1)
    train_rows = [rows[i] for i in idx[:n_train].tolist()]
    val_rows = [rows[i] for i in idx[n_train:].tolist()]
    return train_rows, val_rows


def load_pose_judge_paths(manifest_path: str, select: str) -> Dict[str, str]:
    if select not in {"best_mean", "best_loss"}:
        raise ValueError(f"pose_judge_select must be best_mean or best_loss, got: {select}")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"pose_judge_manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    methods = data.get("methods", [])
    if not isinstance(methods, list):
        raise RuntimeError(f"Invalid manifest format: {manifest_path}")
    key = "pose_judge_ckpt_best_mean" if select == "best_mean" else "pose_judge_ckpt_best_loss"
    out: Dict[str, str] = {}
    for item in methods:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("method_id", "")).strip()
        ckpt = str(item.get(key, "")).strip()
        if not mid or not ckpt:
            continue
        if not os.path.isabs(ckpt):
            ckpt = os.path.abspath(os.path.join(os.path.dirname(manifest_path), ckpt))
        out[mid] = ckpt
    return out


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


def build_eval_groups(manifest: Sequence[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {name: [] for name, _, _ in SUBSETS}
    groups["hard_all"] = []
    groups["normal_ref"] = []
    for r in manifest:
        idx = int(r["frame_idx"])
        sname = r.get("subset")
        if sname in groups:
            groups[sname].append(r)
        if 1 <= idx <= 200:
            groups["hard_all"].append(r)
        if 201 <= idx <= 250:
            groups["normal_ref"].append(r)
    return groups


def export_markdown_table(rows: List[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| Subset | Method | Mean Error (deg) | Median Error (deg) | P95 Error (deg) |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['Subset']} | {r['Method']} | "
                f"{r['Mean Error']:.4f} | {r['Median Error']:.4f} | {r['P95 Error']:.4f} |\n"
            )


def main():
    cfg = parse_args()
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

    methods = build_method_specs(cfg)
    manifest = build_paired_manifest(cfg, logger)
    if cfg.pose_judge_select not in {"best_mean", "best_loss"}:
        raise ValueError(f"pose_judge_select must be best_mean or best_loss, got: {cfg.pose_judge_select}")
    fusion_models: Dict[str, Optional[torch.nn.Module]] = {}
    for method in methods:
        if method["kind"] == "fusion":
            logger.info(f"[METHOD] {method['name']} fusion_ckpt = {method['fusion_ckpt']}")
            fusion_models[method["id"]] = _load_fusion_model(method, device, logger)
        else:
            fusion_models[method["id"]] = None

    judge_paths = load_pose_judge_paths(cfg.pose_judge_manifest, cfg.pose_judge_select)
    pose_judges: Dict[str, PoseRegressorPlain] = {}
    for method in methods:
        mid = method["id"]
        if mid not in judge_paths:
            raise RuntimeError(f"Method {mid} not found in pose judge manifest: {cfg.pose_judge_manifest}")
        ckpt_path = judge_paths[mid]
        logger.info(f"[POSE_JUDGE] method={method['name']} select={cfg.pose_judge_select} ckpt={ckpt_path}")
        pose_judges[mid] = load_pose_judge_model(ckpt_path, device)

    group_rows = build_eval_groups(manifest)
    ordered_groups = ["hard_all", "normal_ref"] + [name for name, _, _ in SUBSETS]

    best_rows: List[dict] = []
    for subset_name in ordered_groups:
        rows = group_rows.get(subset_name, [])
        if len(rows) < 2:
            logger.warning(f"[SKIP] {subset_name} has insufficient rows: {len(rows)}")
            continue
        logger.info(f"[GROUP] {subset_name}: total={len(rows)} (test-only)")

        for method in methods:
            logger.info(f"[METHOD] {subset_name} / {method['name']}")
            eval_ds = PairedSubsetPoseDataset(rows, method["id"], (cfg.resize_h, cfg.resize_w))
            eval_loader = DataLoader(
                eval_ds,
                batch_size=min(max(1, cfg.batch_size // 2), max(1, len(eval_ds))),
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(not cfg.safe_runtime),
                drop_last=False,
                persistent_workers=(cfg.num_workers > 0),
            )
            metric = evaluate_epoch(
                method=method,
                model=fusion_models[method["id"]],
                pose_net=pose_judges[method["id"]],
                loader=eval_loader,
                device=device,
                safe_runtime=cfg.safe_runtime,
            )
            best_rows.append(
                {
                    "Subset": subset_name,
                    "Method": method["name"],
                    "Mean Error": float(metric["val_angle_mean_deg"]),
                    "Median Error": float(metric["val_angle_med_deg"]),
                    "P95 Error": float(metric["val_angle_p95_deg"]),
                }
            )

    table_csv = os.path.join(run_root, "hard_subset_pose_table.csv")
    with open(table_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Subset", "Method", "Mean Error", "Median Error", "P95 Error"],
        )
        writer.writeheader()
        writer.writerows(best_rows)

    table_md = os.path.join(run_root, "hard_subset_pose_table.md")
    export_markdown_table(best_rows, table_md)
    logger.info(f"[DONE] csv: {table_csv}")
    logger.info(f"[DONE] md : {table_md}")


if __name__ == "__main__":
    main()
