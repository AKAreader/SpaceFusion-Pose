#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
train_A_v2_fixed.py

修复点（针对你这次报错）：
1) 不再依赖 TaskFusion_dataset.py 的 cv2.imread（Windows/中文路径不稳），改用 np.fromfile + cv2.imdecode
2) 不再按 Excel/内部逻辑拼文件名，改为扫描 RGB/IR 同名交集（不会取到 IR 不存在的样本）
3) Pose lookup 支持 'xxx__000179.png' -> '000179' 的尾号匹配（和你 train_C 一致）
4) fused_val 按 epoch 输出子目录，避免覆盖
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

import cv2  # 用 imdecode + np.fromfile 解决中文路径读取问题

from FusionNet import FusionNet
from utils import RGB2YCrCb, YCbCr2RGB


# =========================
# 0) 一键配置区
# =========================
@dataclass
class CFG:
    # 数据
    RGB_DIR: str = r"E:\ALL\RGB"
    IR_DIR: str  = r"E:\ALL\IR"
    XLSX: str    = r"E:\ALL\All_Poses_Merged.xlsx"
    EXCEL_ID_COL: str = "pair_id"

    # 训练
    GPU_ID: int = 0
    EPOCHS: int = 10
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4  # 如需定位问题，先改 0
    LR_FUSION: float = 1e-3
    LR_POSE: float = 5e-4
    RESIZE_HW: Tuple[int, int] = (480, 640)  # (H,W)

    # pose
    USE_POSE_SUPERVISION: bool = True
    POSE_WARMUP_EPOCHS: int = 1
    LAMBDA_POSE: float = 0.2
    POSE_INPUT_RGB: bool = True

    # RouteA（显著/ROI）
    SAL_ROI_EXPAND_K: int = 7
    SAL_THRESH: float = 0.60
    LAMBDA_FG: float = 1.6
    LAMBDA_BG: float = 0.8
    W_INT: float = 1.0
    W_GRAD: float = 10.0

    # RouteA 引入策略
    FUSION_WARMUP_EPOCHS: int = 2
    ANNEAL_EPOCHS: int = 3

    # 输出根目录
    RUNS_ROOT: str = r"./runs_fusion/routeA_v2"


# =========================
# 1) 可靠读/写图（支持中文路径/文件名）
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def imread_unicode(path: str, flags: int):
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img

def imwrite_unicode(path: str, img_bgr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    buf.tofile(path)

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
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"IR读取失败：{path}")

    # 兼容 8-bit / 16-bit
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = resize_hw
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

    if img.dtype == np.uint16:
        arr = img.astype(np.float32) / 65535.0
    else:
        arr = img.astype(np.float32) / 255.0

    arr = arr[None, ...]  # 1,H,W
    return torch.from_numpy(arr)

@torch.no_grad()
def save_rgb_tensor_unicode(rgb_chw: torch.Tensor, path: str) -> None:
    x = rgb_chw.detach().cpu().clamp(0, 1)
    x = (x * 255.0 + 0.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()  # HWC RGB
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    imwrite_unicode(path, bgr)


# =========================
# 2) 目录配对 Dataset（RGB/IR 同名交集）
# =========================
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
    返回 (vis_rgb, ir_gray, pose_dummy, mask_dummy, name)
    """
    def __init__(self, rgb_dir: str, ir_dir: str, split: str,
                 resize_hw: Tuple[int, int], split_seed: int = 3407, train_ratio: float = 0.9):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.ir_dir = ir_dir
        self.resize_hw = resize_hw

        rgb_names = _list_images_one_level(rgb_dir)
        ir_names = set(_list_images_one_level(ir_dir))
        common = [n for n in rgb_names if n in ir_names]

        if len(common) == 0:
            raise RuntimeError(
                f"RGB/IR 同名交集为 0。\nRGB_DIR={rgb_dir}\nIR_DIR={ir_dir}\n请确认两目录文件名完全一致。"
            )

        rng = np.random.RandomState(split_seed)
        idx = np.arange(len(common))
        rng.shuffle(idx)
        n_train = int(len(common) * float(train_ratio))
        sel = idx[:n_train] if split == "train" else idx[n_train:]
        self.names = [common[i] for i in sel.tolist()]

        # 对齐你之前日志字段
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


# =========================
# 3) Pose regressor + quat loss
# =========================
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


# =========================
# 4) RouteA loss（显著/ROI）
# =========================
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
    s = 0.5 * _sobel_mag(Y) + 0.5 * _sobel_mag(IR)

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

def _weight_map_from_sal(sal: torch.Tensor, lam_fg: float, lam_bg: float) -> torch.Tensor:
    w = lam_fg * sal + lam_bg * (1.0 - sal)
    w = w / (w.mean(dim=(1, 2, 3), keepdim=True) + 1e-8)  # mean=1 归一化
    return w

def routeA_fusion_loss(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor,
                       sal: torch.Tensor, lam_fg: float, lam_bg: float,
                       w_int: float, w_grad: float):
    w = _weight_map_from_sal(sal, lam_fg, lam_bg)

    tgt_i = torch.max(Y, IR)
    Lin = torch.mean(w * torch.abs(fused - tgt_i))

    gf = _sobel_mag(fused)
    gy = _sobel_mag(Y)
    gi = _sobel_mag(IR)
    tgt_g = torch.max(gy, gi)
    Lgrad = torch.mean(w * torch.abs(gf - tgt_g))

    fus = w_int * Lin + w_grad * Lgrad
    return fus, Lin, Lgrad

def baseline_fusion_loss(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor, w_int: float, w_grad: float):
    tgt_i = torch.max(Y, IR)
    Lin = torch.mean(torch.abs(fused - tgt_i))
    gf = _sobel_mag(fused)
    tgt_g = torch.max(_sobel_mag(Y), _sobel_mag(IR))
    Lgrad = torch.mean(torch.abs(gf - tgt_g))
    fus = w_int * Lin + w_grad * Lgrad
    return fus, Lin, Lgrad


# =========================
# 5) Excel pose map（支持 xxx__000179.png 尾号命中）
# =========================
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

    return list(dict.fromkeys(out))

def load_pose_map_from_excel(xlsx: str, id_col: str, logger: logging.Logger) -> Dict[str, np.ndarray]:
    df = pd.read_excel(xlsx)
    df.columns = [str(c).strip() for c in df.columns]

    logger.info(f"Excel: {xlsx}")
    logger.info(f"Loaded rows={len(df)} cols={len(df.columns)}")
    logger.info(f"Excel cols head: {df.columns.tolist()[:20]}")

    if id_col not in df.columns:
        if "pair_id" in df.columns:
            id_col = "pair_id"
        else:
            raise RuntimeError(f"[POSE] id column not found: {id_col}")

    need = ["卫星四元数_w", "卫星四元数_x", "卫星四元数_y", "卫星四元数_z"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"[POSE] quat col not found: {c}")

    logger.info(f"[POSE] Using id column: {id_col}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[id_col] + need)

    q = df[need].to_numpy(dtype=np.float32)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    df.loc[:, need] = q

    pose_map: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        rid = _safe_str(r[id_col])
        if not rid:
            continue
        quat = r[need].to_numpy(dtype=np.float32)
        for k in _key_variants(rid):
            pose_map[k] = quat

    logger.info(f"[POSE] Excel loaded. rows={len(df)} pose_map_keys={len(pose_map)}")
    logger.info(f"[POSE] quat cols: {','.join(need)}")
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
                found = pose_map[k]; break
        if found is None:
            st = _stem(s)
            for k in _key_variants(st):
                if k in pose_map:
                    found = pose_map[k]; break
        if found is not None:
            pose[i] = found
            mask[i, 0] = 1.0

    hit = float(mask.mean()) if B > 0 else 0.0
    pose_t = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
    return pose_t, mask_t, hit


# =========================
# 6) 日志/输出
# =========================
def setup_logger(run_root: str):
    ensure_dir(run_root)
    log_path = os.path.join(run_root, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("train")

def now_tag():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

@torch.no_grad()
def export_fused(fusion: FusionNet, loader: DataLoader, device: torch.device, save_dir: str):
    ensure_dir(save_dir)
    fusion.eval()
    pbar = tqdm(loader, desc=f"Export -> {save_dir}", leave=False)
    for (vis, ir, _pose, _mask, names) in pbar:
        vis = vis.to(device)
        ir = ir.to(device)
        Y, Cb, Cr = RGB2YCrCb(vis)
        fused_Y = fusion(Y, ir)
        fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)
        for k in range(len(names)):
            save_rgb_tensor_unicode(fused_rgb[k], os.path.join(save_dir, names[k]))


# =========================
# 7) 主训练
# =========================
def train():
    cfg = CFG()

    run_root = os.path.abspath(os.path.join(cfg.RUNS_ROOT, now_tag()))
    ckpt_dir = os.path.join(run_root, "ckpt")
    fused_root = os.path.join(run_root, "fused_val")
    ensure_dir(ckpt_dir)
    ensure_dir(fused_root)

    logger = setup_logger(run_root)

    logger.info(f"RUN_ROOT: {run_root}")
    logger.info("LOSS_MODE: routeA_v2 (fixed dataset+unicode IO)")
    logger.info(f"RGB_DIR: {cfg.RGB_DIR}")
    logger.info(f"IR_DIR : {cfg.IR_DIR}")
    logger.info(f"XLSX   : {cfg.XLSX}")

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    pose_map: Dict[str, np.ndarray] = {}
    if cfg.USE_POSE_SUPERVISION:
        pose_map = load_pose_map_from_excel(cfg.XLSX, cfg.EXCEL_ID_COL, logger)
        logger.info("Pose supervision: ON")
    else:
        logger.info("Pose supervision: OFF")

    # 用目录同名交集，避免引用不存在的 IR 文件
    train_ds = PairFolderDataset(cfg.RGB_DIR, cfg.IR_DIR, "train", cfg.RESIZE_HW, split_seed=3407, train_ratio=0.9)
    val_ds   = PairFolderDataset(cfg.RGB_DIR, cfg.IR_DIR, "val",   cfg.RESIZE_HW, split_seed=train_ds.split_seed, train_ratio=train_ds.train_ratio)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    logger.info(f"[Dataset raw] Pose type: {train_ds.pose_type} | Pose dim: {train_ds.pose_dim}")
    logger.info(f"[OUT] ckpt_dir: {ckpt_dir}")
    logger.info(f"[OUT] fused_root: {fused_root}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.BATCH_SIZE // 2), shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    fusion = FusionNet(output=1).to(device)
    fusion_optim = torch.optim.Adam(fusion.parameters(), lr=cfg.LR_FUSION)

    use_pose = cfg.USE_POSE_SUPERVISION and (len(pose_map) > 0)
    pose_net = None
    pose_optim = None
    if use_pose:
        in_ch = 3 if cfg.POSE_INPUT_RGB else 1
        pose_net = PoseRegressor(out_dim=4, in_ch=in_ch).to(device)
        pose_optim = torch.optim.Adam(pose_net.parameters(), lr=cfg.LR_POSE)

    best_total = float("inf")
    best_pose = float("inf")
    best_angle = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        fusion.train()
        if pose_net:
            pose_net.train()

        if epoch <= cfg.FUSION_WARMUP_EPOCHS:
            alpha = 0.0
        else:
            t = epoch - cfg.FUSION_WARMUP_EPOCHS
            alpha = min(1.0, t / max(1, cfg.ANNEAL_EPOCHS))

        st = time.time()
        meters = {"tot": [], "fus": [], "Lin": [], "Lgrad": [], "hit": [], "pose": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS} (alpha={alpha:.2f})")
        for (vis, ir, _pose_ds, _mask_ds, names) in pbar:
            vis = vis.to(device, non_blocking=True)
            ir  = ir.to(device, non_blocking=True)

            Y, Cb, Cr = RGB2YCrCb(vis)
            fused_Y = fusion(Y, ir)

            # baseline & routeA
            base_fus, base_Lin, base_Lgrad = baseline_fusion_loss(fused_Y, Y, ir, cfg.W_INT, cfg.W_GRAD)
            sal = joint_saliency(Y, ir, cfg.SAL_ROI_EXPAND_K, cfg.SAL_THRESH)
            ra_fus, ra_Lin, ra_Lgrad = routeA_fusion_loss(
                fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG, cfg.W_INT, cfg.W_GRAD
            )

            fus = (1.0 - alpha) * base_fus + alpha * ra_fus
            Lin = (1.0 - alpha) * base_Lin + alpha * ra_Lin
            Lgrad = (1.0 - alpha) * base_Lgrad + alpha * ra_Lgrad

            total = fus
            pose_loss_val = None
            hit = 0.0

            if use_pose and epoch > cfg.POSE_WARMUP_EPOCHS:
                pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
                valid = (pose_mask.view(-1) > 0.5)
                if int(valid.sum().item()) > 0:
                    fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if cfg.POSE_INPUT_RGB else fused_Y
                    pred_pose = pose_net(fused_in)
                    pose_loss_val = quat_loss(pred_pose[valid], pose_gt[valid])
                    total = total + cfg.LAMBDA_POSE * pose_loss_val

            fusion_optim.zero_grad(set_to_none=True)
            if pose_optim:
                pose_optim.zero_grad(set_to_none=True)

            total.backward()
            fusion_optim.step()
            if pose_optim:
                pose_optim.step()

            meters["tot"].append(float(total.detach().cpu()))
            meters["fus"].append(float(fus.detach().cpu()))
            meters["Lin"].append(float(Lin.detach().cpu()))
            meters["Lgrad"].append(float(Lgrad.detach().cpu()))
            meters["hit"].append(hit)
            if pose_loss_val is not None:
                meters["pose"].append(float(pose_loss_val.detach().cpu()))

            msg = (f"tot={np.mean(meters['tot']):.4f} | fus={np.mean(meters['fus']):.4f} | "
                   f"Lin={np.mean(meters['Lin']):.4f} | Lgrad={np.mean(meters['Lgrad']):.4f} | "
                   f"hit={np.mean(meters['hit']):.3f}")
            if meters["pose"]:
                msg += f" | pose={np.mean(meters['pose']):.4f}"
            pbar.set_postfix_str(msg)

        # ----- val -----
        fusion.eval()
        if pose_net:
            pose_net.eval()

        v_tot, v_fus, v_pose, v_hit = [], [], [], []
        v_angles = []

        with torch.no_grad():
            for (vis, ir, _pose_ds, _mask_ds, names) in val_loader:
                vis = vis.to(device)
                ir  = ir.to(device)
                Y, Cb, Cr = RGB2YCrCb(vis)
                fused_Y = fusion(Y, ir)

                base_fus, _, _ = baseline_fusion_loss(fused_Y, Y, ir, cfg.W_INT, cfg.W_GRAD)
                sal = joint_saliency(Y, ir, cfg.SAL_ROI_EXPAND_K, cfg.SAL_THRESH)
                ra_fus, _, _ = routeA_fusion_loss(
                    fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG, cfg.W_INT, cfg.W_GRAD
                )

                fus = (1.0 - alpha) * base_fus + alpha * ra_fus
                total = fus

                hit = 0.0
                pl = None
                if use_pose and epoch > cfg.POSE_WARMUP_EPOCHS:
                    pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device)
                    valid = (pose_mask.view(-1) > 0.5)
                    if int(valid.sum().item()) > 0:
                        fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if cfg.POSE_INPUT_RGB else fused_Y
                        pred_pose = pose_net(fused_in)
                        pl = quat_loss(pred_pose[valid], pose_gt[valid])
                        total = total + cfg.LAMBDA_POSE * pl
                        v_angles.append(quat_angle_deg(pred_pose[valid], pose_gt[valid]).cpu())

                v_tot.append(float(total.cpu()))
                v_fus.append(float(fus.cpu()))
                v_hit.append(hit)
                if pl is not None:
                    v_pose.append(float(pl.cpu()))

        if v_angles:
            ang_all = torch.cat(v_angles, dim=0).numpy()
            a_mean = float(np.mean(ang_all))
            a_med  = float(np.median(ang_all))
            a_p95  = float(np.percentile(ang_all, 95))
        else:
            a_mean = a_med = a_p95 = float("nan")

        train_total = float(np.mean(meters["tot"]))
        train_fusion = float(np.mean(meters["fus"]))
        train_pose = float(np.mean(meters["pose"])) if meters["pose"] else float("nan")
        train_hit  = float(np.mean(meters["hit"])) if meters["hit"] else float("nan")

        val_total = float(np.mean(v_tot))
        val_fusion = float(np.mean(v_fus))
        val_pose = float(np.mean(v_pose)) if v_pose else float("nan")
        val_hit  = float(np.mean(v_hit)) if v_hit else float("nan")

        el = time.time() - st
        logger.info(
            f"[Epoch {epoch}] alpha={alpha:.2f} "
            f"train_total={train_total:.4f} train_fusion={train_fusion:.4f} train_pose={train_pose} train_hit={train_hit:.3f} | "
            f"val_total={val_total:.4f} val_fusion={val_fusion:.4f} val_pose={val_pose} val_hit={val_hit:.3f} | "
            f"val_angle_mean/med/p95={a_mean:.2f}/{a_med:.2f}/{a_p95:.2f} deg | time={el:.1f}s"
        )

        torch.save(fusion.state_dict(), os.path.join(ckpt_dir, f"fusion_epoch{epoch:03d}.pth"))
        if pose_net:
            torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, f"pose_epoch{epoch:03d}.pth"))

        if val_total < best_total:
            best_total = val_total
            torch.save(fusion.state_dict(), os.path.join(ckpt_dir, "fusion_best_total.pth"))
            if pose_net:
                torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, "pose_best_total.pth"))
            logger.info(f"[BEST] best_total updated: {best_total:.6f}")

        if (not np.isnan(val_pose)) and (val_pose < best_pose):
            best_pose = val_pose
            torch.save(fusion.state_dict(), os.path.join(ckpt_dir, "fusion_best_pose.pth"))
            if pose_net:
                torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, "pose_best_pose.pth"))
            logger.info(f"[BEST] best_pose updated: {best_pose:.6f}")

        if (not np.isnan(a_mean)) and (a_mean < best_angle):
            best_angle = a_mean
            torch.save(fusion.state_dict(), os.path.join(ckpt_dir, "fusion_best_angle.pth"))
            if pose_net:
                torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, "pose_best_angle.pth"))
            logger.info(f"[BEST] best_angle(mean deg) updated: {best_angle:.4f}")

        export_dir = os.path.join(fused_root, f"epoch{epoch:03d}")
        export_fused(fusion, val_loader, device, export_dir)

    logger.info("Done.")


if __name__ == "__main__":
    # Windows DataLoader 稳定性
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    train()
