#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
train_B_gate.py

RouteB (Gate版): RouteA_v2(显著/ROI引导) + Gate(自适应门控) + Pose(任务驱动约束)

本版修复你当前的两个致命问题：
1) Windows DataLoader spawn + Dataset pickle 导致的递归崩溃（移除 __getattr__）
2) cv2.imread 在 Windows 上对中文路径/文件名常失败（改用 cv2.imdecode + np.fromfile）
3) 不再依赖 TaskFusion_dataset 内部“按 Excel 组装文件名”的逻辑；改为扫描目录取 RGB/IR 同名交集，确保不会再出现“构造了目录中不存在的名字”。

环境变量（可选）：
DATA_ROOT / RGB_DIR / IR_DIR / XLSX
EPOCHS, BATCH_SIZE, NUM_WORKERS, GPU_ID
INIT_FUSION_CKPT
G_MIN, GATE_WARMUP_EPOCHS, LAMBDA_AUX_BASE
POSE_WARMUP_EPOCHS, LAMBDA_POSE, USE_POSE_SUPERVISION, POSE_INPUT_RGB
TRAIN_RATIO, SPLIT_SEED
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import cv2  # 用 imdecode + np.fromfile 解决中文路径
from FusionNet import FusionNet
from utils import RGB2YCrCb, YCbCr2RGB, save_img_single


# -------------------------
# 0) 配置
# -------------------------
@dataclass
class CFG:
    DATA_ROOT: str = os.environ.get("DATA_ROOT", r"E:\ALL").strip()

    RGB_DIR: str = (os.environ.get("RGB_DIR", "").strip()
                    or os.path.join(os.environ.get("DATA_ROOT", r"E:\ALL"), "RGB"))
    IR_DIR: str = (os.environ.get("IR_DIR", "").strip()
                   or os.path.join(os.environ.get("DATA_ROOT", r"E:\ALL"), "IR"))
    XLSX: str = (os.environ.get("XLSX", "").strip()
                 or os.path.join(os.environ.get("DATA_ROOT", r"E:\ALL"), "All_Poses_Merged.xlsx"))

    # split
    TRAIN_RATIO: float = float(os.environ.get("TRAIN_RATIO", "0.9"))
    SPLIT_SEED: int = int(os.environ.get("SPLIT_SEED", "3407"))

    # training
    GPU_ID: int = int(os.environ.get("GPU_ID", "0"))
    EPOCHS: int = int(os.environ.get("EPOCHS", "10"))
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "8"))
    NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "4"))
    LR_FUSION: float = float(os.environ.get("LR_FUSION", "1e-3"))
    LR_GATE: float = float(os.environ.get("LR_GATE", "1e-3"))
    LR_POSE: float = float(os.environ.get("LR_POSE", "5e-4"))
    RESIZE_HW: Tuple[int, int] = (480, 640)  # (H, W)

    # RouteA_v2 ramp（alpha: 0 -> 1）
    RAMP_EPOCHS: int = int(os.environ.get("RAMP_EPOCHS", "4"))

    # baseline loss 权重
    W_INT: float = float(os.environ.get("W_INT", "1.0"))
    W_GRAD: float = float(os.environ.get("W_GRAD", "10.0"))

    # ROI/显著参数
    SAL_ROI_EXPAND_K: int = int(os.environ.get("SAL_ROI_EXPAND_K", "9"))
    SAL_THRESH: float = float(os.environ.get("SAL_THRESH", "0.55"))
    LAMBDA_FG: float = float(os.environ.get("LAMBDA_FG", "2.5"))
    LAMBDA_BG: float = float(os.environ.get("LAMBDA_BG", "0.8"))

    # Gate 防塌陷
    LAMBDA_TV: float = float(os.environ.get("LAMBDA_TV", "0.01"))
    LAMBDA_GMEAN: float = float(os.environ.get("LAMBDA_GMEAN", "0.00"))
    G_MIN: float = float(os.environ.get("G_MIN", "0.10"))
    GATE_WARMUP_EPOCHS: int = int(os.environ.get("GATE_WARMUP_EPOCHS", "2"))
    LAMBDA_AUX_BASE: float = float(os.environ.get("LAMBDA_AUX_BASE", "0.25"))

    # Pose
    USE_POSE_SUPERVISION: bool = os.environ.get("USE_POSE_SUPERVISION", "1") == "1"
    POSE_WARMUP_EPOCHS: int = int(os.environ.get("POSE_WARMUP_EPOCHS", "1"))
    LAMBDA_POSE: float = float(os.environ.get("LAMBDA_POSE", "0.2"))
    POSE_INPUT_RGB: bool = os.environ.get("POSE_INPUT_RGB", "1") == "1"

    # init ckpt
    INIT_FUSION_CKPT: str = os.environ.get(
        "INIT_FUSION_CKPT",
        r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeA_v2\20251229-002739\ckpt\fusion_best_angle.pth"
    ).strip()

    # out
    RUNS_ROOT: str = os.environ.get("RUNS_ROOT", r"./runs_fusion")
    TAG: str = "routeB_gate"


# -------------------------
# 1) 日志/目录
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def make_run_root(runs_root: str, tag: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(os.path.join(runs_root, tag, ts))

def setup_logger(run_root: str) -> logging.Logger:
    ensure_dir(run_root)
    log_path = os.path.join(run_root, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("train")

def torch_load_safe(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _try_load_state_dict_stripped(model: nn.Module, sd: dict, logger: logging.Logger) -> None:
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint is not a state_dict dict.")
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    keys = list(sd.keys())
    if not keys:
        raise RuntimeError("Empty state_dict.")

    def strip_prefix(d: dict, prefix: str) -> dict:
        out = {}
        for k, v in d.items():
            out[k[len(prefix):]] = v if k.startswith(prefix) else v
        return out

    if any(k.startswith("module.") for k in keys):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        keys = list(sd.keys())

    if any(k.startswith("fusion.") for k in keys):
        sd = {k.replace("fusion.", "", 1): v for k, v in sd.items()}
        keys = list(sd.keys())

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[INIT] load_state_dict strict=False. missing={len(missing)} unexpected={len(unexpected)}")

def init_fusion_from_ckpt(fusion: nn.Module, ckpt_path: str, device: torch.device, logger: logging.Logger) -> None:
    if not ckpt_path:
        logger.info("[INIT] fusion init skipped (INIT_FUSION_CKPT not set).")
        return
    if not os.path.isfile(ckpt_path):
        logger.info(f"[INIT] fusion init skipped (ckpt not found): {ckpt_path}")
        return
    sd = torch_load_safe(ckpt_path, map_location=device)
    _try_load_state_dict_stripped(fusion, sd, logger)
    logger.info(f"[INIT] fusion init OK from: {ckpt_path}")


# -------------------------
# 2) 可靠读图（支持中文路径）
# -------------------------
def imread_unicode(path: str, flags: int):
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img

def read_rgb_tensor(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
    img = imread_unicode(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"RGB读取失败：{path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = resize_hw
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    return torch.from_numpy(arr)

def read_ir_tensor(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"IR读取失败：{path}")
    H, W = resize_hw
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32) / 255.0
    arr = arr[None, ...]  # 1,H,W
    return torch.from_numpy(arr)


# -------------------------
# 3) 目录配对 Dataset（RGB/IR 同名交集）
# -------------------------
def _list_images_one_level(dir_path: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    out = []
    with os.scandir(dir_path) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(exts):
                out.append(e.name)
    out.sort()
    return out

class PairFolderDataset(Dataset):
    """
    返回 (vis, ir, pose_dummy, mask_dummy, names)
    names = 文件名（如 'Aqua_45度__000133.png' 或 '0001.png'）
    """
    def __init__(self,
                 rgb_dir: str,
                 ir_dir: str,
                 split: str,
                 resize_hw: Tuple[int, int],
                 split_seed: int,
                 train_ratio: float):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.resize_hw = resize_hw

        rgb_names = _list_images_one_level(rgb_dir)
        ir_names = set(_list_images_one_level(ir_dir))
        common = [n for n in rgb_names if n in ir_names]

        if len(common) == 0:
            raise RuntimeError(
                f"RGB/IR 同名交集为 0。\n"
                f"请检查：\n"
                f"  RGB_DIR={rgb_dir}\n"
                f"  IR_DIR ={ir_dir}\n"
                f"是否同名配对（含中文、前缀、零填充必须一致）。"
            )

        rng = np.random.RandomState(split_seed)
        idx = np.arange(len(common))
        rng.shuffle(idx)
        n_train = int(len(common) * float(train_ratio))
        if split == "train":
            sel = idx[:n_train]
        elif split == "val":
            sel = idx[n_train:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.names = [common[i] for i in sel.tolist()]

        # 对外暴露：保持你原日志字段
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.pose_type = "none"
        self.pose_dim = 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        rgb_path = os.path.join(self.rgb_dir, name)
        ir_path  = os.path.join(self.ir_dir, name)

        vis = read_rgb_tensor(rgb_path, self.resize_hw)
        ir  = read_ir_tensor(ir_path, self.resize_hw)

        pose_dummy = torch.zeros(1, dtype=torch.float32)
        mask_dummy = torch.zeros(1, dtype=torch.float32)
        return vis, ir, pose_dummy, mask_dummy, name


# -------------------------
# 4) Pose head + quat loss + 角误差
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
# 5) Excel pose map（key 变体：支持 'xxx__000164.png'）
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

def _tail_after_double_underscore(st: str) -> Optional[str]:
    if "__" in st:
        tail = st.split("__")[-1]
        return tail if tail else None
    return None

def _key_variants(s: str) -> List[str]:
    s = _safe_str(s)
    if not s:
        return []
    base = os.path.basename(s)
    st = os.path.splitext(base)[0]
    out: List[str] = []

    for v in [s, base, st]:
        if v and v not in out:
            out.append(v)

    tail = _tail_after_double_underscore(st)
    if tail:
        if tail not in out:
            out.append(tail)
        if (tail + ".png") not in out:
            out.append(tail + ".png")

    def add_digit_forms(d: str):
        if d.isdigit():
            n = int(d)
            out.append(str(n))
            for w in [3, 4, 5, 6, 7, 8]:
                out.append(str(n).zfill(w))
            for w in [3, 4, 5, 6, 7, 8]:
                out.append(str(n).zfill(w) + ".png")
            out.append(str(n) + ".png")

    add_digit_forms(st)
    if tail:
        add_digit_forms(tail)

    # 去重保序
    return list(dict.fromkeys(out))

def load_pose_map_from_excel(xlsx_path: str, logger: logging.Logger) -> Dict[str, np.ndarray]:
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = None
    for c in ["pair_id", "Unified_ID", "帧索引"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]
        logger.warning(f"[POSE] id column not found explicitly; fallback to first col: {id_col}")
    logger.info(f"[POSE] Using id column: {id_col}")

    need = ["卫星四元数_w", "卫星四元数_x", "卫星四元数_y", "卫星四元数_z"]
    alt = ["qw", "qx", "qy", "qz"]
    if all(c in df.columns for c in need):
        qcols = need
    elif all(c in df.columns for c in alt):
        qcols = alt
    else:
        raise RuntimeError(f"[POSE] Quaternion columns not found. cols={df.columns.tolist()}")

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
        if not rid:
            continue
        quat = r[qcols].to_numpy(dtype=np.float32)
        for k in _key_variants(rid):
            pose_map[k] = quat

    logger.info(f"[POSE] Excel loaded. rows={len(df)} pose_map_keys={len(pose_map)}")
    logger.info(f"[POSE] quat cols: {','.join(qcols)}")
    logger.info(f"[POSE] id_col: {id_col}")
    logger.info(f"[POSE] sample keys: {list(pose_map.keys())[:10]}")
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

    hit = float(mask.mean()) if B > 0 else 0.0
    pose_t = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
    return pose_t, mask_t, hit


# -------------------------
# 6) RouteA_v2 ROI/显著损失
# -------------------------
def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

def joint_saliency(Y: torch.Tensor, IR: torch.Tensor, expand_k: int, thresh: float) -> torch.Tensor:
    Y = Y.clamp(0, 1)
    IR = IR.clamp(0, 1)
    g1 = _sobel_mag(Y)
    g2 = _sobel_mag(IR)
    s = 0.5 * g1 + 0.5 * g2

    B = s.shape[0]
    s_flat = s.view(B, -1)
    s_min = s_flat.min(dim=1)[0].view(B, 1, 1, 1)
    s_max = s_flat.max(dim=1)[0].view(B, 1, 1, 1)
    s = (s - s_min) / (s_max - s_min + 1e-8)

    roi = (s > thresh).float()
    if expand_k and expand_k >= 3:
        roi = F.max_pool2d(roi, kernel_size=expand_k, stride=1, padding=expand_k // 2)

    sal = (0.7 * s + 0.3 * roi).clamp(0, 1)
    return sal

def loss_intensity(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor) -> torch.Tensor:
    target = torch.max(Y, IR)
    return torch.mean(torch.abs(fused - target))

def loss_grad(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor) -> torch.Tensor:
    gf = _sobel_mag(fused)
    gy = _sobel_mag(Y)
    gi = _sobel_mag(IR)
    target = torch.max(gy, gi)
    return torch.mean(torch.abs(gf - target))

def loss_intensity_roi(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor, sal: torch.Tensor, lam_fg: float, lam_bg: float) -> torch.Tensor:
    target = torch.max(Y, IR)
    w = lam_fg * sal + lam_bg * (1.0 - sal)
    return torch.mean(w * torch.abs(fused - target))

def loss_grad_roi(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor, sal: torch.Tensor, lam_fg: float, lam_bg: float) -> torch.Tensor:
    gf = _sobel_mag(fused)
    gy = _sobel_mag(Y)
    gi = _sobel_mag(IR)
    target = torch.max(gy, gi)
    w = lam_fg * sal + lam_bg * (1.0 - sal)
    return torch.mean(w * torch.abs(gf - target))


# -------------------------
# 7) Gate 模块
# -------------------------
class GateNet(nn.Module):
    def __init__(self, in_ch: int = 4, g_min: float = 0.10):
        super().__init__()
        self.g_min = float(g_min)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1,  3, 1, 1),
        )

    def forward(self, x):
        logits = self.net(x)
        g = torch.sigmoid(logits)
        g = self.g_min + (1.0 - self.g_min) * g
        return g

class FusionWithGate(nn.Module):
    def __init__(self, g_min: float = 0.10):
        super().__init__()
        self.fusion = FusionNet(output=1)
        self.gate = GateNet(in_ch=4, g_min=g_min)

    def forward(self, Y: torch.Tensor, IR: torch.Tensor):
        base = self.fusion(Y, IR)
        inp = torch.cat([Y, IR, base, torch.abs(Y - IR)], dim=1)
        g = self.gate(inp)
        fallback = torch.max(Y, IR)
        fused = g * base + (1.0 - g) * fallback
        return fused, g, base

def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (
        torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) +
        torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    )


# -------------------------
# 8) 导出 fused_val
# -------------------------
@torch.no_grad()
def export_fused(model: FusionWithGate, loader: DataLoader, device: torch.device, save_dir: str, force_base: bool = False):
    ensure_dir(save_dir)
    model.eval()
    pbar = tqdm(loader, desc=f"Export -> {save_dir}", leave=False)
    for (vis, ir, _pose, _mask, names) in pbar:
        vis = vis.to(device)
        ir = ir.to(device)
        Y, Cb, Cr = RGB2YCrCb(vis)
        fused_Y, g, base = model(Y, ir)
        if force_base:
            fused_Y = base
        fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)
        for k in range(len(names)):
            save_img_single(fused_rgb[k], os.path.join(save_dir, names[k]))


# -------------------------
# 9) 主训练
# -------------------------
def train():
    cfg = CFG()
    run_root = make_run_root(cfg.RUNS_ROOT, cfg.TAG)
    ckpt_dir = os.path.join(run_root, "ckpt")
    fused_root = os.path.join(run_root, "fused_val")
    ensure_dir(ckpt_dir)
    ensure_dir(fused_root)

    logger = setup_logger(run_root)

    logger.info(f"RUN_ROOT: {run_root}")
    logger.info("LOSS_MODE: routeB_gate (routeA_v2 + gate + pose + anti-collapse)")
    logger.info(f"DATA_ROOT: {cfg.DATA_ROOT}")
    logger.info(f"RGB_DIR: {cfg.RGB_DIR}")
    logger.info(f"IR_DIR : {cfg.IR_DIR}")
    logger.info(f"XLSX   : {cfg.XLSX}")

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # pose map
    pose_map: Dict[str, np.ndarray] = {}
    if cfg.USE_POSE_SUPERVISION:
        pose_map = load_pose_map_from_excel(cfg.XLSX, logger)
    use_pose = cfg.USE_POSE_SUPERVISION and (len(pose_map) > 0)
    logger.info(f"Pose supervision: {'ON' if use_pose else 'OFF'}")

    # dataset (来自目录同名交集)
    train_ds = PairFolderDataset(cfg.RGB_DIR, cfg.IR_DIR, "train", cfg.RESIZE_HW, cfg.SPLIT_SEED, cfg.TRAIN_RATIO)
    val_ds   = PairFolderDataset(cfg.RGB_DIR, cfg.IR_DIR, "val",   cfg.RESIZE_HW, cfg.SPLIT_SEED, cfg.TRAIN_RATIO)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    logger.info(f"[Dataset raw] Pose type: {train_ds.pose_type} | Pose dim: {train_ds.pose_dim}")
    logger.info(f"[OUT] ckpt_dir: {os.path.abspath(ckpt_dir)}")
    logger.info(f"[OUT] fused_root: {os.path.abspath(fused_root)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.BATCH_SIZE // 2), shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=True, drop_last=False, persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    # models
    model = FusionWithGate(g_min=cfg.G_MIN).to(device)
    init_fusion_from_ckpt(model.fusion, cfg.INIT_FUSION_CKPT, device, logger)

    pose_net = PoseRegressor(out_dim=4, in_ch=(3 if cfg.POSE_INPUT_RGB else 1)).to(device) if use_pose else None

    params = [
        {"params": model.fusion.parameters(), "lr": cfg.LR_FUSION},
        {"params": model.gate.parameters(),  "lr": cfg.LR_GATE},
    ]
    optim = torch.optim.Adam(params)
    pose_optim = torch.optim.Adam(pose_net.parameters(), lr=cfg.LR_POSE) if pose_net else None

    best_total = float("inf")
    best_pose  = float("inf")
    best_angle = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        if pose_net:
            pose_net.train()

        if cfg.RAMP_EPOCHS <= 1:
            alpha = 1.0
        else:
            alpha = (epoch - 1) / float(cfg.RAMP_EPOCHS - 1)
            alpha = float(np.clip(alpha, 0.0, 1.0))

        st = time.time()
        meters = {"tot": [], "fus": [], "Lin": [], "Lgrad": [], "hit": [], "pose": [], "gmean": [], "gstd": [], "bypass": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} (alpha={alpha:.2f})")
        for (vis, ir, _pose_dummy, _mask_dummy, names) in pbar:
            vis = vis.to(device, non_blocking=True)
            ir  = ir.to(device, non_blocking=True)

            Y, Cb, Cr = RGB2YCrCb(vis)
            sal = joint_saliency(Y, ir, cfg.SAL_ROI_EXPAND_K, cfg.SAL_THRESH)

            fused_Y, g, base = model(Y, ir)

            if epoch <= cfg.GATE_WARMUP_EPOCHS:
                fused_Y = base
                g = torch.ones_like(base)

            Lin_b = loss_intensity(fused_Y, Y, ir)
            Lg_b  = loss_grad(fused_Y, Y, ir)
            fus_b = cfg.W_INT * Lin_b + cfg.W_GRAD * Lg_b

            Lin_r = loss_intensity_roi(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
            Lg_r  = loss_grad_roi(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
            fus_r = cfg.W_INT * Lin_r + cfg.W_GRAD * Lg_r

            fus = (1.0 - alpha) * fus_b + alpha * fus_r
            total = fus

            if cfg.LAMBDA_AUX_BASE > 0:
                base_Lin_b = loss_intensity(base, Y, ir)
                base_Lg_b  = loss_grad(base, Y, ir)
                base_fus_b = cfg.W_INT * base_Lin_b + cfg.W_GRAD * base_Lg_b

                base_Lin_r = loss_intensity_roi(base, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                base_Lg_r  = loss_grad_roi(base, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                base_fus_r = cfg.W_INT * base_Lin_r + cfg.W_GRAD * base_Lg_r

                base_fus = (1.0 - alpha) * base_fus_b + alpha * base_fus_r
                total = total + cfg.LAMBDA_AUX_BASE * base_fus

            if cfg.LAMBDA_TV > 0:
                total = total + cfg.LAMBDA_TV * tv_loss(g)
            if cfg.LAMBDA_GMEAN > 0:
                gm = g.mean()
                total = total + cfg.LAMBDA_GMEAN * (gm - 0.5) * (gm - 0.5)

            hit = 0.0
            pose_loss_val = None
            if use_pose and pose_net and (epoch > cfg.POSE_WARMUP_EPOCHS):
                pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
                valid = (pose_mask.view(-1) > 0.5)
                if int(valid.sum().item()) > 0:
                    fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if cfg.POSE_INPUT_RGB else fused_Y
                    pred_pose = pose_net(fused_in)
                    pose_loss_val = quat_loss(pred_pose[valid], pose_gt[valid])
                    total = total + cfg.LAMBDA_POSE * pose_loss_val

            optim.zero_grad(set_to_none=True)
            if pose_optim:
                pose_optim.zero_grad(set_to_none=True)
            total.backward()
            optim.step()
            if pose_optim:
                pose_optim.step()

            meters["tot"].append(float(total.detach().cpu()))
            meters["fus"].append(float(fus.detach().cpu()))
            meters["Lin"].append(float(((1-alpha)*Lin_b + alpha*Lin_r).detach().cpu()))
            meters["Lgrad"].append(float(((1-alpha)*Lg_b + alpha*Lg_r).detach().cpu()))
            meters["hit"].append(hit)
            meters["gmean"].append(float(g.mean().detach().cpu()))
            meters["gstd"].append(float(g.std().detach().cpu()))
            meters["bypass"].append(float((1.0 - g).mean().detach().cpu()))
            if pose_loss_val is not None:
                meters["pose"].append(float(pose_loss_val.detach().cpu()))

            msg = (
                f"tot={np.mean(meters['tot']):.4f} | fus={np.mean(meters['fus']):.4f} | "
                f"Lin={np.mean(meters['Lin']):.4f} | Lgrad={np.mean(meters['Lgrad']):.4f} | "
                f"g(mean/std)={np.mean(meters['gmean']):.3f}/{np.mean(meters['gstd']):.3f} | "
                f"bypass={np.mean(meters['bypass']):.4f}"
            )
            if use_pose and epoch > cfg.POSE_WARMUP_EPOCHS:
                msg += f" | hit={np.mean(meters['hit']):.3f}"
                if meters["pose"]:
                    msg += f" | pose={np.mean(meters['pose']):.4f}"
            pbar.set_postfix_str(msg)

        # -------- validation --------
        model.eval()
        if pose_net:
            pose_net.eval()

        val_tot, val_fus, val_pose, val_hit = [], [], [], []
        all_angles = []

        with torch.no_grad():
            for (vis, ir, _pose_dummy, _mask_dummy, names) in val_loader:
                vis = vis.to(device)
                ir  = ir.to(device)
                Y, Cb, Cr = RGB2YCrCb(vis)
                sal = joint_saliency(Y, ir, cfg.SAL_ROI_EXPAND_K, cfg.SAL_THRESH)

                fused_Y, g, base = model(Y, ir)
                if epoch <= cfg.GATE_WARMUP_EPOCHS:
                    fused_Y = base
                    g = torch.ones_like(base)

                Lin_b = loss_intensity(fused_Y, Y, ir)
                Lg_b  = loss_grad(fused_Y, Y, ir)
                fus_b = cfg.W_INT * Lin_b + cfg.W_GRAD * Lg_b

                Lin_r = loss_intensity_roi(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                Lg_r  = loss_grad_roi(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                fus_r = cfg.W_INT * Lin_r + cfg.W_GRAD * Lg_r

                fus = (1.0 - alpha) * fus_b + alpha * fus_r
                total = fus

                if cfg.LAMBDA_AUX_BASE > 0:
                    base_Lin_b = loss_intensity(base, Y, ir)
                    base_Lg_b  = loss_grad(base, Y, ir)
                    base_fus_b = cfg.W_INT * base_Lin_b + cfg.W_GRAD * base_Lg_b

                    base_Lin_r = loss_intensity_roi(base, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                    base_Lg_r  = loss_grad_roi(base, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                    base_fus_r = cfg.W_INT * base_Lin_r + cfg.W_GRAD * base_Lg_r

                    base_fus = (1.0 - alpha) * base_fus_b + alpha * base_fus_r
                    total = total + cfg.LAMBDA_AUX_BASE * base_fus

                if cfg.LAMBDA_TV > 0:
                    total = total + cfg.LAMBDA_TV * tv_loss(g)

                hit = 0.0
                pl = None
                if use_pose and pose_net and (epoch > cfg.POSE_WARMUP_EPOCHS):
                    pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
                    valid = (pose_mask.view(-1) > 0.5)
                    if int(valid.sum().item()) > 0:
                        fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if cfg.POSE_INPUT_RGB else fused_Y
                        pred_pose = pose_net(fused_in)
                        pl = quat_loss(pred_pose[valid], pose_gt[valid])
                        total = total + cfg.LAMBDA_POSE * pl
                        ang = quat_angle_deg(pred_pose[valid], pose_gt[valid])
                        all_angles.append(ang.detach().cpu())

                val_tot.append(float(total.cpu()))
                val_fus.append(float(fus.cpu()))
                val_hit.append(hit)
                if pl is not None:
                    val_pose.append(float(pl.cpu()))

        angle_mean = angle_med = angle_p95 = float("nan")
        if len(all_angles) > 0:
            ang_all = torch.cat(all_angles, dim=0).numpy()
            angle_mean = float(np.mean(ang_all))
            angle_med  = float(np.median(ang_all))
            angle_p95  = float(np.percentile(ang_all, 95))

        train_total = float(np.mean(meters["tot"]))
        train_fusion = float(np.mean(meters["fus"]))
        train_pose = float(np.mean(meters["pose"])) if meters["pose"] else float("nan")
        train_hit  = float(np.mean(meters["hit"])) if meters["hit"] else float("nan")

        v_total = float(np.mean(val_tot))
        v_fusion = float(np.mean(val_fus))
        v_pose = float(np.mean(val_pose)) if val_pose else float("nan")
        v_hit  = float(np.mean(val_hit)) if val_hit else float("nan")

        el = time.time() - st
        logger.info(
            f"[Epoch {epoch}] alpha={alpha:.2f} "
            f"train_total={train_total:.4f} train_fusion={train_fusion:.4f} train_pose={train_pose} train_hit={train_hit:.3f} | "
            f"val_total={v_total:.4f} val_fusion={v_fusion:.4f} val_pose={v_pose} val_hit={v_hit:.3f} | "
            f"val_angle_mean/med/p95={angle_mean:.2f}/{angle_med:.2f}/{angle_p95:.2f} deg | time={el:.1f}s"
        )

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fusion_epoch{epoch:03d}.pth"))
        if pose_net:
            torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, f"pose_epoch{epoch:03d}.pth"))

        if v_total < best_total:
            best_total = v_total
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "fusion_best_total.pth"))
            logger.info(f"[BEST] best_total updated: {best_total:.6f}")

        if (not np.isnan(v_pose)) and v_pose < best_pose:
            best_pose = v_pose
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "fusion_best_pose.pth"))
            if pose_net:
                torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, "pose_best_pose.pth"))
            logger.info(f"[BEST] best_pose updated: {best_pose:.6f}")

        if (not np.isnan(angle_mean)) and angle_mean < best_angle:
            best_angle = angle_mean
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "fusion_best_angle.pth"))
            logger.info(f"[BEST] best_angle(mean deg) updated: {best_angle:.4f}")

        export_fused(model, val_loader, device, fused_root, force_base=(epoch <= cfg.GATE_WARMUP_EPOCHS))

    logger.info("Done.")


if __name__ == "__main__":
    train()
