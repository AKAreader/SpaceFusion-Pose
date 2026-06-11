# datasets.py
# coding:utf-8
import os
import pandas as pd
import torchvision.transforms.functional as TF
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from natsort import natsorted

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _is_img(fn: str) -> bool:
    return os.path.splitext(fn)[1].lower() in IMG_EXTS

class Fusion_dataset(Dataset):
    """
    split: 'train' | 'val' | 'test'
    使用你自己的数据：
      - ir_dir:  红外文件夹
      - vis_dir: 可见光文件夹
      - excel:   配对表（可选）。若提供则优先用 excel 配对，否则按文件名 stem 交集配对
      - label_dir: 语义标签文件夹（可选）。只有你要保留语义闭环训练时才需要
    """
    def __init__(self, split, ir_dir=None, vis_dir=None, excel=None, label_dir=None):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.split = split
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.label_dir = label_dir
        self.pairs = self._build_pairs(excel)

        self.length = len(self.pairs)

    def _build_pairs(self, excel):
        # 1) Excel 优先：尝试寻找 ir/rgb 列
        if excel and os.path.exists(excel):
            df = pd.read_excel(excel)
            cols = {c.lower(): c for c in df.columns}

            def pick(cands):
                for k in cands:
                    if k in cols:
                        return cols[k]
                return None

            c_rgb = pick(["rgb", "vis", "visible", "rgb_path", "vis_path"])
            c_ir  = pick(["ir", "infrared", "ir_path"])
            c_id  = pick(["id", "name", "filename", "pair_id"])

            pairs = []
            if c_rgb and c_ir:
                for i, r in df.iterrows():
                    pid = str(r[c_id]).strip() if c_id else f"{i:06d}"
                    rgbv = str(r[c_rgb]).strip()
                    irv  = str(r[c_ir]).strip()
                    rgb_path = rgbv if os.path.isabs(rgbv) else os.path.join(self.vis_dir, rgbv)
                    ir_path  = irv  if os.path.isabs(irv)  else os.path.join(self.ir_dir, irv)
                    if os.path.exists(rgb_path) and os.path.exists(ir_path):
                        pairs.append((pid, ir_path, rgb_path))
                if len(pairs) > 0:
                    return pairs

        # 2) fallback：按文件名 stem 交集自动配对
        vis_files = [f for f in os.listdir(self.vis_dir) if _is_img(f)]
        ir_files  = [f for f in os.listdir(self.ir_dir)  if _is_img(f)]
        vis_map = {os.path.splitext(f)[0].lower(): f for f in vis_files}
        ir_map  = {os.path.splitext(f)[0].lower(): f for f in ir_files}
        common = natsorted(list(set(vis_map.keys()) & set(ir_map.keys())))
        return [(k, os.path.join(self.ir_dir, ir_map[k]), os.path.join(self.vis_dir, vis_map[k])) for k in common]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pid, ir_path, vis_path = self.pairs[idx]
        img_vis = self.imread(vis_path, vis_flag=True)     # [3,H,W]
        img_ir  = self.imread(ir_path,  vis_flag=False)    # [1,H,W]

        # label：如果没提供 label_dir，就给一个 dummy（全 0），避免 train.py 崩
        if self.label_dir and os.path.exists(self.label_dir):
            # 支持：label 文件名与 pid 对齐（pid.png / pid.jpg 等）
            cand = None
            for ext in IMG_EXTS:
                p = os.path.join(self.label_dir, f"{pid}{ext}")
                if os.path.exists(p):
                    cand = p
                    break
            if cand is None:
                # 也允许与可见光同名
                cand = os.path.join(self.label_dir, os.path.basename(vis_path))
            label = self.imread(cand, label=True).type(torch.LongTensor)  # [1,H,W], 0..255
        else:
            _, H, W = img_ir.shape
            label = torch.zeros((1, H, W), dtype=torch.long)

        # 为了兼容你当前 train.py/run_fusion 的 (vis, ir, label, name) 形式
        return img_vis, img_ir, label, f"{pid}.png"

    @staticmethod
    def imread(path, label=False, vis_flag=True):
        if label:
            img = Image.open(path)
            return TF.to_tensor(img) * 255
        if vis_flag:
            img = Image.open(path).convert("RGB")
            return TF.to_tensor(img)
        img = Image.open(path).convert("L")
        return TF.to_tensor(img)
