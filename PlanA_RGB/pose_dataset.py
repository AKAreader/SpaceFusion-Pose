# pose_dataset.py (PlanA_RGB)
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from utils_img import imread_any, to_float01


def _norm_rgb_imagenet(rgb: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb - mean) / std


def _norm_ir_default(ir: np.ndarray) -> np.ndarray:
    mean = 0.5
    std = 0.25
    return (ir - mean) / std


def _resize_square(img: np.ndarray, out_size: int) -> np.ndarray:
    return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)


def _clip_box(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W, x2)))
    y2 = int(max(0, min(H, y2)))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return x1, y1, x2, y2


def _sample_crop_box(roi_xyxy, H, W, expand, jitter, rng: np.random.RandomState):
    x1, y1, x2, y2 = [float(v) for v in roi_xyxy]
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw2 = bw * float(expand)
    bh2 = bh * float(expand)

    if jitter > 0:
        dx = (rng.rand() * 2.0 - 1.0) * jitter * bw2
        dy = (rng.rand() * 2.0 - 1.0) * jitter * bh2
        cx += dx
        cy += dy

    nx1 = cx - bw2 / 2.0
    ny1 = cy - bh2 / 2.0
    nx2 = cx + bw2 / 2.0
    ny2 = cy + bh2 / 2.0
    return _clip_box(nx1, ny1, nx2, ny2, W=W, H=H)


def _crop(img: np.ndarray, box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    if img.ndim == 2:
        return img[y1:y2, x1:x2]
    return img[y1:y2, x1:x2, :]


class PoseDataset(Dataset):
    """
    mode:
      - "ir":    IR ROI -> [1,H,W]
      - "rgb":   RGB ROI -> [3,H,W]
      - "rgbir": RGB(3)+IR(1) -> [4,H,W]
    """
    def __init__(self,
                 data_root: str,
                 json_path: str,
                 mode: str = "ir",
                 out_size: int = 224,
                 train: bool = True,
                 roi_expand: float = 1.30,
                 roi_jitter: float = 0.10,
                 seed: int = 42,
                 roi_expand_mode: str = "fixed",
                 roi_expand_min: float = 1.20,
                 roi_expand_max: float = 1.70,
                 share_crop: bool = True):
        super().__init__()
        assert mode in ["ir", "rgb", "rgbir"]
        assert roi_expand_mode in ["fixed", "random"]
        self.data_root = data_root
        self.mode = mode
        self.out_size = int(out_size)
        self.train = bool(train)

        self.roi_expand = float(roi_expand)
        self.roi_jitter = float(roi_jitter)
        self.roi_expand_mode = roi_expand_mode
        self.roi_expand_min = float(roi_expand_min)
        self.roi_expand_max = float(roi_expand_max)

        self.share_crop = bool(share_crop)
        self.rng = np.random.RandomState(seed if train else seed + 999)

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def _sample_expand(self) -> float:
        if self.train and self.roi_expand_mode == "random":
            return float(self.rng.uniform(self.roi_expand_min, self.roi_expand_max))
        return float(self.roi_expand)

    def __getitem__(self, idx):
        item = self.samples[idx]
        roi = item["roi_xyxy"]
        q_gt = np.array(item["pose_quat_wxyz"], dtype=np.float32)  # wxyz

        # ---- read IR first to get H,W and shared crop ----
        ir_path = os.path.join(self.data_root, item["ir_path"])
        ir = imread_any(ir_path, flag=cv2.IMREAD_UNCHANGED)
        if ir.ndim == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        ir = to_float01(ir)
        H, W = ir.shape[:2]

        expand = self._sample_expand()
        jitter = self.roi_jitter if self.train else 0.0
        box = _sample_crop_box(roi, H=H, W=W, expand=expand, jitter=jitter, rng=self.rng)

        # ---- build input by mode ----
        if self.mode == "ir":
            ir_roi = _crop(ir, box)
            ir_roi = _resize_square(ir_roi, self.out_size)
            ir_roi = _norm_ir_default(ir_roi)
            x = ir_roi[None, :, :]  # 1xHxW

        else:
            # RGB needed for rgb and rgbir
            rgb_path = os.path.join(self.data_root, item["rgb_path"])
            rgb = imread_any(rgb_path, flag=cv2.IMREAD_COLOR)  # BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = to_float01(rgb)

            rgb_roi = _crop(rgb, box)
            rgb_roi = _resize_square(rgb_roi, self.out_size)
            rgb_roi = _norm_rgb_imagenet(rgb_roi)  # HxWx3

            if self.mode == "rgb":
                x = rgb_roi.transpose(2, 0, 1)  # 3xHxW
            else:
                # rgbir: concatenate RGB + IR
                ir_roi = _crop(ir, box)
                ir_roi = _resize_square(ir_roi, self.out_size)
                ir_roi = _norm_ir_default(ir_roi)
                x = np.concatenate([rgb_roi.transpose(2, 0, 1), ir_roi[None, :, :]], axis=0)  # 4xHxW

        x = torch.from_numpy(x.astype(np.float32))
        q_gt = torch.from_numpy(q_gt)
        return x, q_gt, item["id"]
