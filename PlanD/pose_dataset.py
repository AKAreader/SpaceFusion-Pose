# pose_dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from roi_tools import cv_imread_cn, load_roi_cache, sample_square_from_roi

def to_float01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0

def norm_rgb_imagenet(rgb: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb - mean) / std

def norm_ir(ir: np.ndarray) -> np.ndarray:
    mean = 0.5
    std = 0.25
    return (ir - mean) / std

def crop(img, box):
    x1, y1, x2, y2 = box
    if img.ndim == 2:
        return img[y1:y2, x1:x2]
    return img[y1:y2, x1:x2, :]

class PoseDataset(Dataset):
    def __init__(
        self,
        data_root,
        json_path,
        mode="ir",                   # "ir" / "rgbir" / "dual"
        out_size=384,
        train=True,
        roi_expand=1.35,
        roi_jitter=0.02,
        seed=42,
        roi_expand_mode="fixed",     # "fixed" / "random"
        roi_expand_min=1.25,
        roi_expand_max=1.70,
        roi_cache_path=None,
        use_soft_mask=False,
        soft_mask_source="saliency"  # "saliency" / "ir"
    ):
        super().__init__()
        assert mode in ["ir", "rgbir", "dual"]
        self.data_root = data_root
        self.mode = mode
        self.out_size = int(out_size)
        self.train = bool(train)

        self.saliency_dir = os.path.join(self.data_root, "Saliency_Maps")

        self.roi_expand = float(roi_expand)
        self.roi_jitter = float(roi_jitter)
        self.roi_expand_mode = roi_expand_mode
        self.roi_expand_min = float(roi_expand_min)
        self.roi_expand_max = float(roi_expand_max)

        self.rng = np.random.RandomState(seed if train else seed + 999)
        self.roi_cache = load_roi_cache(roi_cache_path)

        self.use_soft_mask = bool(use_soft_mask)
        self.soft_mask_source = soft_mask_source

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def _sample_expand(self):
        if self.train and self.roi_expand_mode == "random":
            return float(self.rng.uniform(self.roi_expand_min, self.roi_expand_max))
        return float(self.roi_expand)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sid = int(s["id"])

        # --- GT quat ---
        q_gt = np.array(s["pose_quat_wxyz"], dtype=np.float32)
        # canonical：w>=0（避免双覆盖不稳定）
        if q_gt[0] < 0:
            q_gt = -q_gt

        # --- K, t ---
        raw_k = np.array(s["camera_k"], dtype=np.float32)
        raw_t = np.array(s["pose_t"], dtype=np.float32)
        t_gt = np.array([raw_t[0], -raw_t[2], raw_t[1]], dtype=np.float32)

        # --- read IR (get H,W) ---
        ir_path = os.path.join(self.data_root, s["ir_path"])
        ir_u8 = cv_imread_cn(ir_path, cv2.IMREAD_GRAYSCALE)
        if ir_u8 is None:
            ir_u8 = np.zeros((1080, 1920), dtype=np.uint8)
        H, W = ir_u8.shape[:2]
        ir = to_float01(ir_u8)

        # --- read saliency (optional but used in dual / softmask) ---
        fname = os.path.basename(s["ir_path"])
        sal_path = os.path.join(self.saliency_dir, fname)
        sal_u8 = cv_imread_cn(sal_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sal_path) else None
        if sal_u8 is None:
            sal = np.zeros((H, W), dtype=np.float32)
        else:
            if sal_u8.shape[:2] != (H, W):
                sal_u8 = cv2.resize(sal_u8, (W, H), interpolation=cv2.INTER_NEAREST)
            sal = to_float01(sal_u8)

        # --- base ROI (cache优先) ---
        base_roi = self.roi_cache.get(sid, s["roi_xyxy"])

        # --- sample crop box (关键：方形 + 贴边不缩小) ---
        expand = self._sample_expand()
        jitter = self.roi_jitter if self.train else 0.0
        box = sample_square_from_roi(base_roi, H, W, expand=expand, jitter=jitter, rng=self.rng)

        crop_x1, crop_y1 = box[0], box[1]
        crop_w = box[2] - box[0]
        crop_h = box[3] - box[1]

        # --- update K (crop shift + resize scale) ---
        new_fx = raw_k[0, 0]
        new_fy = raw_k[1, 1]
        new_cx = raw_k[0, 2] - crop_x1
        new_cy = raw_k[1, 2] - crop_y1
        scale_x = self.out_size / float(crop_w)
        scale_y = self.out_size / float(crop_h)
        k_final = np.array([
            [new_fx * scale_x, 0, new_cx * scale_x],
            [0, new_fy * scale_y, new_cy * scale_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # --- crop & resize ---
        ir_roi = crop(ir, box)
        sal_roi = crop(sal, box)

        ir_roi = cv2.resize(ir_roi, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        sal_roi = cv2.resize(sal_roi, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)

        # --- optional soft mask (背景抑制，不改框) ---
        if self.use_soft_mask:
            if self.soft_mask_source == "saliency":
                m = (sal_roi >= float(np.quantile(sal_roi, 0.85))).astype(np.float32)
            else:
                m = (ir_roi >= 0.02).astype(np.float32)
            m = cv2.GaussianBlur(m, (0, 0), 1.5)
            ir_roi = ir_roi * (0.3 + 0.7 * m)

        ir_tensor = torch.from_numpy(norm_ir(ir_roi)).unsqueeze(0)
        sal_tensor = torch.from_numpy(sal_roi).unsqueeze(0)

        # --- RGB part ---
        if self.mode == "ir":
            rgb_tensor = torch.zeros(3, self.out_size, self.out_size, dtype=torch.float32)
            x = ir_tensor
        else:
            rgb_path = os.path.join(self.data_root, s["rgb_path"])
            rgb_bgr = cv_imread_cn(rgb_path, cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                rgb_bgr = np.zeros((H, W, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            rgb = to_float01(rgb)

            rgb_roi = crop(rgb, box)
            rgb_roi = cv2.resize(rgb_roi, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)

            if self.use_soft_mask:
                # 用同一个 m 背景压暗
                rgb_roi = rgb_roi * (0.3 + 0.7 * m[..., None])

            rgb_tensor = torch.from_numpy(norm_rgb_imagenet(rgb_roi).transpose(2, 0, 1))
            x = torch.cat([rgb_tensor, ir_tensor], dim=0)  # 4ch

        return {
            "x": x,
            "rgb": rgb_tensor,
            "ir": ir_tensor,
            "saliency": sal_tensor,
            "q_gt": torch.from_numpy(q_gt),
            "t_gt": torch.from_numpy(t_gt),
            "K": torch.from_numpy(k_final),
            "id": sid
        }
