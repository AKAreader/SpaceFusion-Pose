#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
一键运行（无需命令行）：
- 训练 SeAFusion 融合网络（无语义GT）
- 可选：使用 Excel 中四元数姿态做 task-driven 约束（pose regression loss）

本版本解决你遇到的关键问题：
1) Excel 支持 Unified_ID / pair_id（你现在已经有了），也支持只有“帧索引”的情况（自动生成 0001..）
2) hit 命中率每个 epoch 都统计（train/val 都统计），不会再出现“其实能命中但日志 hit=0”的假象
3) 输出绝不覆盖：每次运行生成独立 RUN_ROOT；每个 epoch 的 fused 输出单独子目录
4) baseline 与 RouteA(ROI/显著性引导损失) 使用同一个脚本，通过 CFG.LOSS_MODE 切换'Baseline': r"./runs_fusion/baseline/ckpt/fusion_best_total.pth",
    'TrainA (ROI)': r"./runs_fusion/routeA/ckpt/fusion_best_angle.pth",
    'TrainB (Gate)': r"./runs_fusion/routeB/ckpt/fusion_best_angle.pth",
    'TrainC (Ours)': r"./runs_fusion/routeC/ckpt/fusion_best_angle.pth",
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
from loss import Fusionloss
from utils import RGB2YCrCb, YCbCr2RGB, save_img_single


# =========================
# 0) 配置区：只改这里
# =========================
@dataclass
class CFG:
    # 数据路径
    IR_DIR: str = r"E:\ALL\IR"
    RGB_DIR: str = r"E:\ALL\RGB"
    XLSX: str = r"E:\Test\Aqua_with_unified_id.xlsx"  # 建议用你 align 后的 aligned.xlsx

    # 运行标签（用于区分 baseline / routeA 等）
    RUN_TAG: str = "baseline"     # 例如 "baseline" 或 "routeA"
    RUNS_ROOT: str = r"./runs_fusion"

    # loss 模式： "baseline" 使用工程原 Fusionloss；"routeA" 使用 ROI/显著性引导损失
    LOSS_MODE: str = "baseline"   # "baseline" or "routeA"

    # 训练
    GPU_ID: int = 0
    EPOCHS: int = 10
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    LR_FUSION: float = 1e-3
    LR_POSE: float = 5e-4
    RESIZE_HW: Tuple[int, int] = (480, 640)  # (H,W)

    # 导出 fused：每个 epoch 输出到 RUN_ROOT/fused_val/epochXX，不覆盖
    EXPORT_FUSED_EACH_EPOCH: bool = True
    EXPORT_MAX_PER_EPOCH: int = 999999  # 想快速看效果可改小，比如 50

    # 姿态监督（四元数）
    USE_POSE_SUPERVISION: bool = True
    POSE_WARMUP_EPOCHS: int = 1      # 前N个epoch只做融合损失（但 hit 仍统计）
    LAMBDA_POSE: float = 0.2
    POSE_INPUT_RGB: bool = True      # True: 用融合RGB做姿态回归；False: 用融合Y(1通道)

    # Excel 识别策略：优先用 pair_id / Unified_ID；没有则用帧索引生成
    EXCEL_ID_COL_PREFER: str = "pair_id"     # 你现在有 pair_id 和 Unified_ID；这里建议优先 pair_id
    EXCEL_FRAME_COL_FALLBACK: str = "帧索引"
    GENERATED_ID_WIDTH: int = 4

    # RouteA（ROI/显著性引导）参数：只在 LOSS_MODE="routeA" 时生效
    SAL_ROI_EXPAND_K: int = 9   # ROI 扩张核 7/9/11
    SAL_THRESH: float = 0.55    # 阈值 0.45~0.60
    LAMBDA_FG: float = 2.5      # ROI 区域权重
    LAMBDA_BG: float = 0.8      # 非 ROI 权重
    W_INT: float = 1.0          # 强度项权重
    W_GRAD: float = 10.0        # 梯度项权重（量级对齐 baseline 常见写法）


# =========================
# 1) Pose 回归网络（轻量）
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
    """loss = 1 - (dot)^2，四元数符号不敏感"""
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).clamp(-1.0, 1.0)
    return torch.mean(1.0 - dot * dot)


@torch.no_grad()
def quat_angle_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    返回每个样本的旋转夹角（degree）
    angle = 2 * arccos(|dot|)
    """
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot) * (180.0 / np.pi)
    return ang


# =========================
# 2) RouteA：显著性/ROI 引导损失
# =========================
def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    # x: [B,1,H,W]
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
    """
    联合梯度显著图 -> 阈值 ROI -> maxpool 扩张 -> soft saliency
    输出 sal in [0,1], shape [B,1,H,W]
    """
    Y = Y.clamp(0, 1)
    IR = IR.clamp(0, 1)

    s = 0.5 * _sobel_mag(Y) + 0.5 * _sobel_mag(IR)
    B = s.shape[0]
    sf = s.view(B, -1)
    s_min = sf.min(dim=1)[0].view(B, 1, 1, 1)
    s_max = sf.max(dim=1)[0].view(B, 1, 1, 1)
    s = (s - s_min) / (s_max - s_min + 1e-8)

    roi = (s > thresh).float()
    if expand_k and expand_k >= 3:
        roi = F.max_pool2d(roi, kernel_size=expand_k, stride=1, padding=expand_k // 2)

    sal = (0.7 * s + 0.3 * roi).clamp(0, 1)
    return sal


def saliency_weighted_intensity_loss(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor,
                                    sal: torch.Tensor, lam_fg: float, lam_bg: float) -> torch.Tensor:
    target = torch.max(Y, IR)
    w = lam_fg * sal + lam_bg * (1.0 - sal)
    return torch.mean(w * torch.abs(fused - target))


def saliency_weighted_grad_loss(fused: torch.Tensor, Y: torch.Tensor, IR: torch.Tensor,
                               sal: torch.Tensor, lam_fg: float, lam_bg: float) -> torch.Tensor:
    gf = _sobel_mag(fused)
    target = torch.max(_sobel_mag(Y), _sobel_mag(IR))
    w = lam_fg * sal + lam_bg * (1.0 - sal)
    return torch.mean(w * torch.abs(gf - target))


# =========================
# 3) Excel pose_map（支持 pair_id/Unified_ID，缺失则用帧索引生成）
# =========================
def _norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    for ch in [" ", "\t", "\n", "\r", "_", "-", "—"]:
        s = s.replace(ch, "")
    return s


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_norm = [_norm_col(c) for c in cols]
    cand_norm = [_norm_col(c) for c in candidates]
    for cn in cand_norm:
        for i, x in enumerate(cols_norm):
            if x == cn:
                return cols[i]
    for cn in cand_norm:
        for i, x in enumerate(cols_norm):
            if cn in x:
                return cols[i]
    return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return int(float(x))
    except Exception:
        return None


def _stem(name: str) -> str:
    base = os.path.basename(str(name))
    return os.path.splitext(base)[0]


def _key_variants_from_id(id_str: str, zfill_width: int) -> List[str]:
    """
    给定 ID（'0001' / '1' / '0001.png'），生成尽可能多的 key 变体
    """
    out: List[str] = []
    s = str(id_str).strip()
    if not s:
        return out
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    def add(v: str):
        if v and v not in out:
            out.append(v)

    st = _stem(s)
    add(s); add(st)

    if st.isdigit():
        n = str(int(st))
        add(n)
        add(n.zfill(zfill_width))

    # png variants
    base_variants = list(out)
    for v in base_variants:
        if not v.lower().endswith(".png"):
            add(v + ".png")

    return out


def load_pose_map_from_excel(cfg: CFG, logger: logging.Logger) -> Dict[str, np.ndarray]:
    if not os.path.exists(cfg.XLSX):
        raise FileNotFoundError(f"Excel not found: {cfg.XLSX}")

    df = pd.read_excel(cfg.XLSX)
    logger.info(f"Excel: {cfg.XLSX}")
    logger.info(f"Loaded rows={len(df)} cols={len(df.columns)}")
    logger.info(f"Excel cols head: {list(df.columns)[:20]}")

    # quat cols（支持中文与 qw/qx/qy/qz）
    col_qw = _find_col(df, ["卫星四元数_w", "qw", "quat_w", "q_w"])
    col_qx = _find_col(df, ["卫星四元数_x", "qx", "quat_x", "q_x"])
    col_qy = _find_col(df, ["卫星四元数_y", "qy", "quat_y", "q_y"])
    col_qz = _find_col(df, ["卫星四元数_z", "qz", "quat_z", "q_z"])
    if any(c is None for c in [col_qw, col_qx, col_qy, col_qz]):
        raise RuntimeError(f"Quaternion columns not found. qw={col_qw},qx={col_qx},qy={col_qy},qz={col_qz}")

    # id col（优先 pair_id / Unified_ID）
    col_id = _find_col(df, [cfg.EXCEL_ID_COL_PREFER, "pair_id", "Unified_ID", "unified_id", "id", "ID"])
    col_frame = _find_col(df, [cfg.EXCEL_FRAME_COL_FALLBACK, "frame", "frameindex", "帧", "帧号"])

    if col_id is None:
        if col_frame is None:
            raise RuntimeError("Excel 里既找不到 pair_id/Unified_ID，也找不到帧索引列。")
        logger.warning(f"[POSE] ID column not found; fallback to frame column: {col_frame}")
    else:
        logger.info(f"[POSE] Using id column: {col_id}")

    for c in [col_qw, col_qx, col_qy, col_qz]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[col_qw, col_qx, col_qy, col_qz])

    pose_map: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        quat = np.array([float(r[col_qw]), float(r[col_qx]), float(r[col_qy]), float(r[col_qz])], dtype=np.float32)
        quat = quat / (np.linalg.norm(quat) + 1e-8)

        if col_id is not None:
            rid = str(r[col_id]).strip()
        else:
            fi = _safe_int(r[col_frame])
            if fi is None:
                continue
            rid = str(fi).zfill(cfg.GENERATED_ID_WIDTH)

        for k in _key_variants_from_id(rid, cfg.GENERATED_ID_WIDTH):
            pose_map[k] = quat

    sample_keys = list(pose_map.keys())[:10]
    logger.info(f"[POSE] Excel loaded. rows={len(df)} pose_map_keys={len(pose_map)}")
    logger.info(f"[POSE] quat cols: {col_qw},{col_qx},{col_qy},{col_qz}")
    logger.info(f"[POSE] id_col: {col_id if col_id is not None else '(generated from '+str(col_frame)+')'}")
    logger.info(f"[POSE] sample keys: {sample_keys}")
    return pose_map


def lookup_pose_batch(names, pose_map: Dict[str, np.ndarray], device: torch.device, id_width: int):
    names_list = list(names)
    B = len(names_list)
    gt = np.zeros((B, 4), dtype=np.float32)
    mask = np.zeros((B,), dtype=np.float32)

    for i, nm in enumerate(names_list):
        st = _stem(nm)  # 0001
        found = None
        for k in _key_variants_from_id(st, id_width):
            if k in pose_map:
                found = pose_map[k]
                break
        if found is not None:
            gt[i] = found
            mask[i] = 1.0

    hit = float(mask.mean()) if B > 0 else 0.0
    gt_t = torch.from_numpy(gt).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
    return gt_t, mask_t, hit


# =========================
# 4) 工具：run_root / logger / fused export
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_run_root(cfg: CFG) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_root = os.path.join(cfg.RUNS_ROOT, cfg.RUN_TAG, ts)
    ensure_dir(run_root)
    return run_root


def setup_logger(run_root: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(run_root, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@torch.no_grad()
def export_fused(model: FusionNet, loader: DataLoader, device: torch.device, save_dir: str, max_n: int):
    ensure_dir(save_dir)
    model.eval()
    count = 0
    pbar = tqdm(loader, desc=f"Export -> {os.path.basename(save_dir)}", leave=False)
    for (vis, ir, _pose_dummy, _mask_dummy, names) in pbar:
        vis = vis.to(device)
        ir = ir.to(device)

        Y, Cb, Cr = RGB2YCrCb(vis)
        fused_Y = model(Y, ir)
        fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)

        for k in range(len(names)):
            save_path = os.path.join(save_dir, names[k])
            save_img_single(fused_rgb[k], save_path)
            count += 1
            if count >= max_n:
                return


# =========================
# 5) 训练主流程（含 val）
# =========================
def train():
    cfg = CFG()
    run_root = make_run_root(cfg)
    logger = setup_logger(run_root)

    torch.backends.cudnn.benchmark = True

    logger.info(f"RUN_ROOT: {os.path.abspath(run_root)}")
    logger.info(f"LOSS_MODE: {cfg.LOSS_MODE}")
    logger.info(f"RGB_DIR: {cfg.RGB_DIR}")
    logger.info(f"IR_DIR : {cfg.IR_DIR}")
    logger.info(f"XLSX   : {cfg.XLSX}")

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- output dirs ---
    ckpt_dir = os.path.join(run_root, "ckpt")
    fused_root = os.path.join(run_root, "fused_val")
    ensure_dir(ckpt_dir)
    ensure_dir(fused_root)

    # --- pose map ---
    pose_map: Dict[str, np.ndarray] = {}
    if cfg.USE_POSE_SUPERVISION:
        try:
            pose_map = load_pose_map_from_excel(cfg, logger)
        except Exception as e:
            logger.error(f"[POSE] Failed to build pose_map: {e}")
            pose_map = {}

    pose_map_present = (len(pose_map) > 0)
    use_pose_for_loss = bool(cfg.USE_POSE_SUPERVISION) and pose_map_present
    logger.info(f"Pose supervision: {'ON' if use_pose_for_loss else 'OFF'}")

    # --- dataset / loader ---
    train_ds = Fusion_dataset(
        split="train",
        ir_path=cfg.IR_DIR,
        vi_path=cfg.RGB_DIR,
        xlsx_path=cfg.XLSX,
        resize_hw=cfg.RESIZE_HW,
    )
    val_ds = Fusion_dataset(
        split="val",
        ir_path=cfg.IR_DIR,
        vi_path=cfg.RGB_DIR,
        xlsx_path=cfg.XLSX,
        resize_hw=cfg.RESIZE_HW,
        split_seed=train_ds.split_seed,
        train_ratio=train_ds.train_ratio,
    )
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    logger.info(f"[Dataset raw] Pose type: {train_ds.pose_type} | Pose dim: {train_ds.pose_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.BATCH_SIZE // 2),
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # --- models ---
    fusion = FusionNet(output=1).to(device)
    fusion_optim = torch.optim.Adam(fusion.parameters(), lr=cfg.LR_FUSION)

    # baseline fusion criterion
    fusion_crit = Fusionloss().to(device)

    # pose net
    pose_net = None
    pose_optim = None
    if use_pose_for_loss:
        pose_in_ch = 3 if cfg.POSE_INPUT_RGB else 1
        pose_net = PoseRegressor(out_dim=4, in_ch=pose_in_ch).to(device)
        pose_optim = torch.optim.Adam(pose_net.parameters(), lr=cfg.LR_POSE)

    # --- best trackers ---
    best_val_total = float("inf")
    best_val_pose = float("inf")

    def save_ckpt(tag: str):
        torch.save(fusion.state_dict(), os.path.join(ckpt_dir, f"fusion_{tag}.pth"))
        if pose_net is not None:
            torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, f"pose_{tag}.pth"))

    logger.info(f"[OUT] ckpt_dir: {os.path.abspath(ckpt_dir)}")
    logger.info(f"[OUT] fused_root: {os.path.abspath(fused_root)}")

    try:
        for epoch in range(1, cfg.EPOCHS + 1):
            fusion.train()
            if pose_net:
                pose_net.train()

            st = time.time()

            tr_tot, tr_fus, tr_pose, tr_hit = [], [], [], []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")

            for (vis, ir, _pose_ds, _mask_ds, names) in pbar:
                vis = vis.to(device, non_blocking=True)
                ir = ir.to(device, non_blocking=True)

                # 姿态查表：无论是否 warmup，都算 hit（避免你看到 hit=0 的误导）
                if cfg.USE_POSE_SUPERVISION and pose_map_present:
                    pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device, cfg.GENERATED_ID_WIDTH)
                else:
                    pose_gt = None
                    pose_mask = None
                    hit = float("nan")

                Y, Cb, Cr = RGB2YCrCb(vis)
                fused_Y = fusion(Y, ir)

                # --- fusion loss ---
                if cfg.LOSS_MODE.lower() == "baseline":
                    fus_loss, Lin, Lgrad = fusion_crit(Y, ir, None, fused_Y, 0)
                    fus = fus_loss
                elif cfg.LOSS_MODE.lower() == "routea":
                    sal = joint_saliency(Y, ir, cfg.SAL_ROI_EXPAND_K, cfg.SAL_THRESH)
                    Lin = saliency_weighted_intensity_loss(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                    Lgrad = saliency_weighted_grad_loss(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                    fus = cfg.W_INT * Lin + cfg.W_GRAD * Lgrad
                else:
                    raise ValueError(f"Unknown LOSS_MODE: {cfg.LOSS_MODE}")

                total = fus
                pose_loss_val = None

                # --- pose loss（过 warmup 后才加入 total） ---
                if use_pose_for_loss and epoch > cfg.POSE_WARMUP_EPOCHS:
                    valid = (pose_mask > 0.5)
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

                tr_tot.append(float(total.detach().cpu()))
                tr_fus.append(float(fus.detach().cpu()))
                if pose_loss_val is not None:
                    tr_pose.append(float(pose_loss_val.detach().cpu()))
                tr_hit.append(hit if not np.isnan(hit) else 0.0)

                msg = f"tot={np.mean(tr_tot):.4f} | fus={np.mean(tr_fus):.4f}"
                msg += f" | Lin={float(Lin.detach().cpu()):.4f} | Lgrad={float(Lgrad.detach().cpu()):.4f}"
                if cfg.USE_POSE_SUPERVISION and pose_map_present:
                    msg += f" | hit={np.mean(tr_hit):.3f}"
                if tr_pose:
                    msg += f" | pose={np.mean(tr_pose):.4f}"
                pbar.set_postfix_str(msg)

            # --- val ---
            fusion.eval()
            if pose_net:
                pose_net.eval()

            val_tot, val_fus, val_pose, val_hit = [], [], [], []
            val_angles = []

            with torch.no_grad():
                for (vis, ir, _pose_ds, _mask_ds, names) in val_loader:
                    vis = vis.to(device)
                    ir = ir.to(device)

                    if cfg.USE_POSE_SUPERVISION and pose_map_present:
                        pose_gt, pose_mask, hit = lookup_pose_batch(names, pose_map, device, cfg.GENERATED_ID_WIDTH)
                    else:
                        pose_gt = None
                        pose_mask = None
                        hit = float("nan")

                    Y, Cb, Cr = RGB2YCrCb(vis)
                    fused_Y = fusion(Y, ir)

                    if cfg.LOSS_MODE.lower() == "baseline":
                        fus_loss, Lin_v, Lgrad_v = fusion_crit(Y, ir, None, fused_Y, 0)
                        fus_v = fus_loss
                    else:
                        sal = joint_saliency(Y, ir, cfg.SAL_ROI_EXPAND_K, cfg.SAL_THRESH)
                        Lin_v = saliency_weighted_intensity_loss(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                        Lgrad_v = saliency_weighted_grad_loss(fused_Y, Y, ir, sal, cfg.LAMBDA_FG, cfg.LAMBDA_BG)
                        fus_v = cfg.W_INT * Lin_v + cfg.W_GRAD * Lgrad_v

                    total_v = fus_v
                    pl = None

                    if use_pose_for_loss and epoch > cfg.POSE_WARMUP_EPOCHS:
                        valid = (pose_mask > 0.5)
                        if int(valid.sum().item()) > 0:
                            fused_in = YCbCr2RGB(fused_Y, Cb, Cr) if cfg.POSE_INPUT_RGB else fused_Y
                            pred_pose = pose_net(fused_in)
                            pl = quat_loss(pred_pose[valid], pose_gt[valid])
                            total_v = total_v + cfg.LAMBDA_POSE * pl

                            # 角度误差（更直观）
                            ang = quat_angle_deg(pred_pose[valid], pose_gt[valid])
                            val_angles.append(ang.detach().cpu())

                    val_tot.append(float(total_v.cpu()))
                    val_fus.append(float(fus_v.cpu()))
                    if pl is not None:
                        val_pose.append(float(pl.cpu()))
                    val_hit.append(hit if not np.isnan(hit) else 0.0)

            # 汇总
            tr_total = float(np.mean(tr_tot))
            tr_fusion = float(np.mean(tr_fus))
            tr_pose_mean = float(np.mean(tr_pose)) if tr_pose else float("nan")
            tr_hit_mean = float(np.mean(tr_hit)) if tr_hit else float("nan")

            v_total = float(np.mean(val_tot))
            v_fusion = float(np.mean(val_fus))
            v_pose_mean = float(np.mean(val_pose)) if val_pose else float("nan")
            v_hit_mean = float(np.mean(val_hit)) if val_hit else float("nan")

            angle_mean = angle_med = angle_p95 = float("nan")
            if val_angles:
                a = torch.cat(val_angles).numpy()
                angle_mean = float(np.mean(a))
                angle_med = float(np.median(a))
                angle_p95 = float(np.percentile(a, 95))

            el = time.time() - st
            logger.info(
                f"[Epoch {epoch}] "
                f"train_total={tr_total:.4f} train_fusion={tr_fusion:.4f} train_pose={tr_pose_mean} train_hit={tr_hit_mean:.3f} | "
                f"val_total={v_total:.4f} val_fusion={v_fusion:.4f} val_pose={v_pose_mean} val_hit={v_hit_mean:.3f} | "
                f"val_angle_mean/med/p95={angle_mean:.2f}/{angle_med:.2f}/{angle_p95:.2f} deg | "
                f"time={el:.1f}s"
            )

            # 保存 epoch ckpt
            torch.save(fusion.state_dict(), os.path.join(ckpt_dir, f"fusion_epoch{epoch:03d}.pth"))
            if pose_net is not None:
                torch.save(pose_net.state_dict(), os.path.join(ckpt_dir, f"pose_epoch{epoch:03d}.pth"))

            # best by total
            if v_total < best_val_total:
                best_val_total = v_total
                save_ckpt("best_total")
                logger.info(f"[BEST] best_total updated: {best_val_total:.6f}")

            # best by pose（如果有）
            if not np.isnan(v_pose_mean) and v_pose_mean < best_val_pose:
                best_val_pose = v_pose_mean
                save_ckpt("best_pose")
                logger.info(f"[BEST] best_pose updated: {best_val_pose:.6f}")

            # export fused（每个 epoch 独立目录，不覆盖）
            if cfg.EXPORT_FUSED_EACH_EPOCH:
                epoch_dir = os.path.join(fused_root, f"epoch{epoch:03d}")
                export_fused(fusion, val_loader, device, epoch_dir, cfg.EXPORT_MAX_PER_EPOCH)

        logger.info("Done.")

    except KeyboardInterrupt:
        # 你想中途停也能保留模型：会额外保存一个 interrupt 版本
        logger.warning("KeyboardInterrupt caught. Saving interrupt checkpoint...")
        save_ckpt("interrupt")
        logger.info("Saved interrupt ckpt. Exit.")


if __name__ == "__main__":
    train()
