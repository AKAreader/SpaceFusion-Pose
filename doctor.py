#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
compare_ABC.py
不使用命令行参数；在代码顶部填写 A/B/C 三个模型的 ckpt 路径即可。

功能：
1) 统一 val 切分（按目录同名交集、固定 seed/ratio）对 A/B/C 做推理；
2) 导出每个方法的 fused 图到 out_dir/<method>/；
3) 计算融合质量指标（Entropy/STD/AG/SF/MI/SSIM 等）并导出 CSV；
4) 若提供 pose_ckpt：计算姿态角误差 mean/median/p95（使用 Excel 四元数）。

你只需要改：CFG 里的 RGB_DIR/IR_DIR/XLSX，以及 MODEL_A/B/C 的 ckpt 空白处。
"""

from __future__ import annotations

import os
import math
import time
import csv
import logging
import zipfile
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

from FusionNet import FusionNet
from utils import RGB2YCrCb, YCbCr2RGB


def _xlsx_col_to_index(ref: str) -> int:
    ref = str(ref)
    col = []
    for ch in ref:
        if ch.isalpha():
            col.append(ch.upper())
        else:
            break
    out = 0
    for ch in col:
        out = out * 26 + (ord(ch) - ord("A") + 1)
    return max(0, out - 1)


def _read_xlsx_rows_fallback(xlsx_path: str) -> List[List[object]]:
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(".//{*}si"):
                texts = [t.text or "" for t in si.findall(".//{*}t")]
                shared.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.find("{*}sheets")
        if sheets is None or len(list(sheets)) == 0:
            raise RuntimeError("No worksheets found in xlsx.")
        first_sheet = list(sheets)[0]
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")

        target = None
        if rel_id and "xl/_rels/workbook.xml.rels" in zf.namelist():
            rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            for rel in rels.findall("{*}Relationship"):
                if rel.attrib.get("Id") == rel_id:
                    target = rel.attrib.get("Target")
                    break
        if not target:
            target = "worksheets/sheet1.xml"
        target = str(target).replace("\\", "/").lstrip("/")
        if target.startswith("xl/"):
            sheet_path = target
        else:
            sheet_path = "xl/" + target

        sheet = ET.fromstring(zf.read(sheet_path))
        rows: List[List[object]] = []
        for row in sheet.findall(".//{*}sheetData/{*}row"):
            vals: List[object] = []
            for cell in row.findall("{*}c"):
                col_idx = _xlsx_col_to_index(cell.attrib.get("r", "A1"))
                while len(vals) < col_idx:
                    vals.append(None)

                cell_type = cell.attrib.get("t", "")
                value = None
                if cell_type == "inlineStr":
                    node = cell.find("{*}is")
                    if node is not None:
                        value = "".join(t.text or "" for t in node.findall(".//{*}t"))
                else:
                    node = cell.find("{*}v")
                    if node is not None:
                        raw = node.text
                        if cell_type == "s":
                            idx = int(raw)
                            value = shared[idx] if 0 <= idx < len(shared) else raw
                        elif cell_type == "b":
                            value = (raw == "1")
                        else:
                            value = raw
                vals.append(value)
            rows.append(vals)

        if not rows:
            return []
        width = max(len(r) for r in rows)
        return [r + [None] * (width - len(r)) for r in rows]


def _load_excel_df_fallback(xlsx_path: str) -> pd.DataFrame:
    rows = _read_xlsx_rows_fallback(xlsx_path)
    if not rows:
        return pd.DataFrame()
    header = [str(x).strip() if x is not None else "" for x in rows[0]]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def read_excel_robust(xlsx_path: str, logger: logging.Logger) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path)
    except ImportError as e:
        logger.warning(f"[XLSX] pandas/openpyxl unavailable, using xml fallback reader: {e}")
        return _load_excel_df_fallback(xlsx_path)


# =========================================================
# 0) 你只需要改这里：数据路径 + 三个模型 ckpt（空白处）
# =========================================================
@dataclass
class CFG:
    # 数据
    RGB_DIR: str = r"E:\ALL\RGB"
    IR_DIR: str  = r"E:\ALL\IR"
    XLSX: str    = r"E:\ALL\All_Poses_Merged.xlsx"

    # split（与你训练一致）
    TRAIN_RATIO: float = 0.9
    SPLIT_SEED: int = 3407
    RESIZE_HW: Tuple[int, int] = (480, 640)  # (H,W)

    # 推理
    GPU_ID: int = 0
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "2"))

    # 输出
    OUT_ROOT: str = r"./runs_compare"
    TAG: str = "ABC_compare"


# -------- 三个模型配置（留出空白处，你粘贴路径即可）--------
# 你日志对应的“第几个模型（epoch）”：
#   A: epoch=5  (trainA best_angle)
#   B: epoch=4  (trainB best_angle)
#   C: epoch=7  (trainC best_angle)

MODEL_A = {
    "name": "A_routeA_v2_epoch5_best_angle",
    "type": "fusionnet",     # A 是纯 FusionNet
    "use_rectify": False,    # A 不用 gate/rectify
    "fusion_ckpt": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeA_v2\20260106-145605\ckpt\fusion_best_angle.pth",      # <<< 填这里：trainA 的 ckpt/fusion_best_angle.pth
    "pose_ckpt":   r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeA_v2\20260106-145605\ckpt\pose_best_angle.pth",      # <<< 可选：trainA 的 ckpt/pose_best_angle.pth（没有就留空）
    "pose_arch": "plain",    # A 的 pose 一般是 plain（无 CA）
}

MODEL_B = {
    "name": "B_gate_epoch4_best_angle",
    "type": "gate",          # B 是 FusionWithGate
    "use_rectify": False,    # 你的 trainB 日志未显示 rectify 调度；默认关（如你确认训练里开了，再改 True）
    "rectify_alpha": 0.35,
    "rectify_detach_sal": True,
    "fusion_ckpt": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeB_gate\20260106-170140\ckpt\fusion_epoch005.pth",      # <<< 填这里：trainB 的 ckpt/fusion_best_angle.pth
    "pose_ckpt":   r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeB_gate\20260106-170140\ckpt\pose_epoch005.pth",      # <<< 可选：trainB 的 ckpt/pose_best_angle.pth
    "pose_arch": "plain",    # 若你的 trainB 用了 CA，则改 "ca"
}

MODEL_C = {
    "name": "C_gate_epoch7_best_angle",
    "type": "gate",          # C 是 FusionWithGate
    "use_rectify": False,     # 你的 trainC 明确有 rectify 调度（alpha_max=0.35）
    "rectify_alpha": 0.35,
    "rectify_detach_sal": True,
    "fusion_ckpt": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeC_v2_ssim\20260108-183933\ckpt\fusion_best_angle.pth",      # <<< 填这里：trainC 的 ckpt/fusion_best_angle.pth
    "pose_ckpt":   r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeC_v2_ssim\20260108-183933\ckpt\pose_best_pose.pth",      # <<< 可选：trainC 的 ckpt/pose_best_angle.pth（或 pose_best_pose.pth）
    "pose_arch": "ca",       # trainC 用 CA
}

METHODS = [MODEL_A, MODEL_B, MODEL_C]


# =========================================================
# 1) 日志 / 工具
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def make_out_root(out_root: str, tag: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(os.path.join(out_root, tag, ts))

def setup_logger(run_root: str) -> logging.Logger:
    ensure_dir(run_root)
    log_path = os.path.join(run_root, "compare.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("compare")

def torch_load_safe(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =========================================================
# 2) Unicode 安全读写图（Windows 中文路径）
# =========================================================
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
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"IR读取失败：{path}")
    H, W = resize_hw
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
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


# =========================================================
# 3) Dataset：按目录同名交集（与你 trainB/trainC 一致）
# =========================================================
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
    def __init__(self, rgb_dir: str, ir_dir: str, split: str,
                 resize_hw: Tuple[int, int], split_seed: int, train_ratio: float):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.ir_dir  = ir_dir
        self.resize_hw = resize_hw

        rgb_names = _list_images_one_level(rgb_dir)
        ir_names = set(_list_images_one_level(ir_dir))
        common = [n for n in rgb_names if n in ir_names]
        if len(common) == 0:
            raise RuntimeError("RGB/IR 同名交集为 0，请检查两目录文件名是否完全一致。")

        rng = np.random.RandomState(split_seed)
        idx = np.arange(len(common))
        rng.shuffle(idx)
        n_train = int(len(common) * float(train_ratio))

        if split == "train":
            sel = idx[:n_train]
        elif split == "val":
            sel = idx[n_train:]
        else:
            raise ValueError("split must be train/val")

        self.names = [common[i] for i in sel.tolist()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        vis = read_rgb_tensor(os.path.join(self.rgb_dir, name), self.resize_hw)
        ir  = read_ir_tensor(os.path.join(self.ir_dir,  name), self.resize_hw)
        return vis, ir, name


# =========================================================
# 4) 显著图（给 gate/rectify 用）
# =========================================================
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

def joint_saliency(Y: torch.Tensor, IR: torch.Tensor, expand_k: int = 9, thresh: float = 0.55) -> torch.Tensor:
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


# =========================================================
# 5) Gate / Rectify（给 B/C 用）
# =========================================================
class SaliencyRectify(nn.Module):
    def __init__(self, ch: int, alpha: float = 0.35, detach_sal: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.detach_sal = bool(detach_sal)
        self.gate = nn.Sequential(
            nn.Conv2d(1, ch, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, feat: torch.Tensor, sal: Optional[torch.Tensor]) -> torch.Tensor:
        if sal is None:
            return feat
        if self.detach_sal:
            sal = sal.detach()
        sal = F.interpolate(sal, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(sal)
        return feat + self.alpha * g * feat

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
        g = torch.sigmoid(self.net(x))
        return self.g_min + (1.0 - self.g_min) * g

class FusionWithGate(nn.Module):
    def __init__(self, g_min: float = 0.10, use_rectify: bool = True,
                 rectify_alpha: float = 0.35, rectify_detach_sal: bool = True):
        super().__init__()
        self.fusion = FusionNet(output=1)
        self.gate = GateNet(in_ch=4, g_min=g_min)
        self.use_rectify = bool(use_rectify)
        self.rectify = SaliencyRectify(ch=1, alpha=rectify_alpha, detach_sal=rectify_detach_sal) if self.use_rectify else None

    def forward(self, Y: torch.Tensor, IR: torch.Tensor, sal: Optional[torch.Tensor] = None):
        base = self.fusion(Y, IR)  # [B,1,H,W]
        if self.use_rectify and (sal is not None):
            base = self.rectify(base, sal)

        inp = torch.cat([Y, IR, base, torch.abs(Y - IR)], dim=1)
        g = self.gate(inp)
        fallback = torch.max(Y, IR)
        fused = g * base + (1.0 - g) * fallback
        return fused, g, base


# =========================================================
# 6) Pose（Excel GT + angle）
# =========================================================
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
                out.append(str(n).zfill(w) + ".png")
            out.append(str(n) + ".png")

    add_digit_forms(st)
    if tail:
        add_digit_forms(tail)

    return list(dict.fromkeys(out))

def load_pose_map_from_excel(xlsx_path: str, logger: logging.Logger) -> Dict[str, np.ndarray]:
    df = read_excel_robust(xlsx_path, logger)
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
    if not all(c in df.columns for c in need):
        raise RuntimeError(f"[POSE] Quaternion columns not found. cols={df.columns.tolist()}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[id_col] + need)

    q = df[need].to_numpy(dtype=np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    q = q / n
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
    return pose_map

def lookup_pose_one(name: str, pose_map: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    s = _safe_str(name)
    for k in _key_variants(s):
        if k in pose_map:
            return pose_map[k]
    st = _stem(s)
    for k in _key_variants(st):
        if k in pose_map:
            return pose_map[k]
    return None

def quat_angle_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    gt   = gt   / (gt.norm(dim=1, keepdim=True) + 1e-8)
    dot = torch.sum(pred * gt, dim=1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot) * (180.0 / np.pi)
    return ang


# ---- Pose 网络（plain + CA 两种，按你的 ckpt 选择）----
class CoordAtt(nn.Module):
    def __init__(self, inp: int, reduction: int = 32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, inp, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, inp, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = F.adaptive_avg_pool2d(x, (H, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, W)).permute(0, 1, 3, 2)  # [B,C,W,1]
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w

class PoseRegressorPlain(nn.Module):
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
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

class PoseRegressorCA(nn.Module):
    def __init__(self, out_dim: int = 4, in_ch: int = 3, reduction: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.ca1   = CoordAtt(32, reduction=reduction)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.ca2   = CoordAtt(64, reduction=reduction)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.ca3   = CoordAtt(128, reduction=reduction)
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.ca4   = CoordAtt(256, reduction=reduction)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )
    def forward(self, x):
        x = self.ca1(self.conv1(x))
        x = self.ca2(self.conv2(x))
        x = self.ca3(self.conv3(x))
        x = self.ca4(self.conv4(x))
        return self.head(x)


# =========================================================
# 7) 融合质量指标（轻量、稳定：Entropy/STD/AG/SF/MI/SSIM）
# =========================================================
def _to_uint8(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def metric_entropy(gray01: np.ndarray) -> float:
    u8 = _to_uint8(gray01)
    hist = np.bincount(u8.flatten(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def metric_std(gray01: np.ndarray) -> float:
    return float(np.std(gray01))

def metric_ag(gray01: np.ndarray) -> float:
    # average gradient
    gx = np.diff(gray01, axis=1)
    gy = np.diff(gray01, axis=0)
    g = np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2 + 1e-12)
    return float(np.mean(g))

def metric_sf(gray01: np.ndarray) -> float:
    # spatial frequency
    rf = np.diff(gray01, axis=0)
    cf = np.diff(gray01, axis=1)
    return float(np.sqrt(np.mean(rf**2) + np.mean(cf**2)))

def metric_mi(a01: np.ndarray, b01: np.ndarray) -> float:
    # mutual information (256 bins)
    a = _to_uint8(a01).flatten()
    b = _to_uint8(b01).flatten()
    joint = np.histogram2d(a, b, bins=256, range=[[0, 255], [0, 255]])[0].astype(np.float64)
    joint /= (joint.sum() + 1e-12)
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    mi = (joint[nz] * (np.log2(joint[nz] + 1e-12) - np.log2(pa[nz.any(axis=1)] + 1e-12).repeat(256)[0:joint[nz].shape[0]]*0)).sum()
    # 上面为了避免复杂索引，这里改成稳定写法：
    # 重新算一次更稳定的 MI
    pa1 = joint.sum(axis=1)
    pb1 = joint.sum(axis=0)
    mi_val = 0.0
    for i in range(256):
        if pa1[i] <= 0:
            continue
        for j in range(256):
            p = joint[i, j]
            if p <= 0 or pb1[j] <= 0:
                continue
            mi_val += p * (math.log(p + 1e-12, 2) - math.log(pa1[i] + 1e-12, 2) - math.log(pb1[j] + 1e-12, 2))
    return float(mi_val)

def metric_ssim(a01: np.ndarray, b01: np.ndarray) -> float:
    # 简化 SSIM（单尺度，全图）
    a = a01.astype(np.float64)
    b = b01.astype(np.float64)
    mu_a = a.mean(); mu_b = b.mean()
    var_a = a.var();  var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    ssim = ((2*mu_a*mu_b + c1) * (2*cov + c2)) / ((mu_a*mu_a + mu_b*mu_b + c1) * (var_a + var_b + c2) + 1e-12)
    return float(ssim)


# =========================================================
# 8) 推理 + 评测
# =========================================================
def _load_fusion_model(method: dict, device: torch.device, logger: logging.Logger):
    ckpt = method["fusion_ckpt"]
    if not ckpt or (not os.path.isfile(ckpt)):
        raise RuntimeError(f"[{method['name']}] fusion_ckpt 为空或不存在：{ckpt}")

    if method["type"] == "fusionnet":
        model = FusionNet(output=1).to(device)
        sd = torch_load_safe(ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model

    elif method["type"] == "gate":
        model = FusionWithGate(
            g_min=0.10,
            use_rectify=bool(method.get("use_rectify", False)),
            rectify_alpha=float(method.get("rectify_alpha", 0.35)),
            rectify_detach_sal=bool(method.get("rectify_detach_sal", True)),
        ).to(device)
        sd = torch_load_safe(ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown method type: {method['type']}")

def _load_pose_model(method: dict, device: torch.device, logger: logging.Logger):
    ckpt = method.get("pose_ckpt", "")
    if not ckpt or (not os.path.isfile(ckpt)):
        logger.info(f"[{method['name']}] pose_ckpt 未提供或不存在，跳过姿态评测。")
        return None

    arch = method.get("pose_arch", "plain").lower()
    # 默认用 RGB 输入（和你训练一致）
    if arch == "ca":
        pose_net = PoseRegressorCA(out_dim=4, in_ch=3, reduction=32).to(device)
    else:
        pose_net = PoseRegressorPlain(out_dim=4, in_ch=3).to(device)

    sd = torch_load_safe(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    pose_net.load_state_dict(sd, strict=False)
    pose_net.eval()
    return pose_net

@torch.no_grad()
def run_one_method(method: dict, loader: DataLoader, device: torch.device,
                   pose_map: Dict[str, np.ndarray], out_dir: str, logger: logging.Logger):
    fusion = _load_fusion_model(method, device, logger)
    pose_net = _load_pose_model(method, device, logger)

    ensure_dir(out_dir)

    rows = []
    angles = []

    pbar = tqdm(loader, desc=f"Eval {method['name']}")
    for vis, ir, names in pbar:
        vis = vis.to(device)
        ir  = ir.to(device)

        Y, Cb, Cr = RGB2YCrCb(vis)

        # forward
        if method["type"] == "fusionnet":
            fused_Y = fusion(Y, ir)
            g = None
        else:
            sal = joint_saliency(Y, ir, expand_k=9, thresh=0.55)  # 推理用固定即可
            fused_Y, g, base = fusion(Y, ir, sal=sal)

        fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)

        # metrics（全部在 CPU 上用 numpy）
        Y_np  = Y.detach().cpu().numpy()[:, 0]
        IR_np = ir.detach().cpu().numpy()[:, 0]
        F_np  = fused_Y.detach().cpu().numpy()[:, 0]

        for i in range(len(names)):
            nm = names[i]
            y  = Y_np[i]
            irg = IR_np[i]
            f  = F_np[i]

            ent = metric_entropy(f)
            sd  = metric_std(f)
            ag  = metric_ag(f)
            sf  = metric_sf(f)

            # MI(F, Y)+MI(F, IR) 更符合融合场景
            mi = metric_mi(f, y) + metric_mi(f, irg)

            # SSIM：F 与两源的平均
            ssim = 0.5 * (metric_ssim(f, y) + metric_ssim(f, irg))

            # pose angle（如果有）
            ang = None
            if pose_net is not None:
                gt = lookup_pose_one(nm, pose_map)
                if gt is not None:
                    pred = pose_net(fused_rgb[i:i+1])  # [1,4]
                    gt_t = torch.from_numpy(gt[None, :]).to(device=device, dtype=torch.float32)
                    a = quat_angle_deg(pred, gt_t).item()
                    angles.append(a)
                    ang = a

            # save fused image
            save_rgb_tensor_unicode(fused_rgb[i], os.path.join(out_dir, nm))

            rows.append({
                "name": nm,
                "entropy": ent,
                "std": sd,
                "ag": ag,
                "sf": sf,
                "mi_sum": mi,
                "ssim_avg": ssim,
                "pose_angle_deg": ("" if ang is None else ang),
            })

    # summary
    def _mean(key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float, np.floating))]
        return float(np.mean(vals)) if vals else float("nan")

    angle_mean = angle_med = angle_p95 = float("nan")
    if len(angles) > 0:
        angle_mean = float(np.mean(angles))
        angle_med  = float(np.median(angles))
        angle_p95  = float(np.percentile(angles, 95))

    summary = {
        "method": method["name"],
        "N": len(rows),
        "entropy_mean": _mean("entropy"),
        "std_mean": _mean("std"),
        "ag_mean": _mean("ag"),
        "sf_mean": _mean("sf"),
        "mi_sum_mean": _mean("mi_sum"),
        "ssim_avg_mean": _mean("ssim_avg"),
        "pose_angle_mean": angle_mean,
        "pose_angle_med": angle_med,
        "pose_angle_p95": angle_p95,
    }
    return rows, summary


def main():
    cfg = CFG()
    run_root = make_out_root(cfg.OUT_ROOT, cfg.TAG)
    ensure_dir(run_root)
    logger = setup_logger(run_root)

    logger.info(f"RUN_ROOT: {run_root}")
    logger.info(f"RGB_DIR: {cfg.RGB_DIR}")
    logger.info(f"IR_DIR : {cfg.IR_DIR}")
    logger.info(f"XLSX   : {cfg.XLSX}")

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # dataset（val split）
    val_ds = PairFolderDataset(cfg.RGB_DIR, cfg.IR_DIR, "val", cfg.RESIZE_HW, cfg.SPLIT_SEED, cfg.TRAIN_RATIO)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    logger.info(f"Val samples: {len(val_ds)}")

    # pose map
    pose_map = load_pose_map_from_excel(cfg.XLSX, logger)

    all_summaries = []
    for m in METHODS:
        out_dir = os.path.join(run_root, "fused", m["name"])
        rows, summary = run_one_method(m, val_loader, device, pose_map, out_dir, logger)
        all_summaries.append(summary)

        # per-image csv
        per_csv = os.path.join(run_root, f"metrics_{m['name']}.csv")
        with open(per_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                w.writeheader()
                w.writerows(rows)

        logger.info(f"[DONE] {m['name']} -> fused_dir={out_dir}")
        logger.info(f"[SUMMARY] {summary}")

    # summary csv
    sum_csv = os.path.join(run_root, "summary_ABC.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as f:
        keys = list(all_summaries[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_summaries)

    logger.info(f"All done. summary: {sum_csv}")


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    main()
