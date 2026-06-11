#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
from utils import RGB2YCrCb, YCbCr2RGB, save_img_single


@dataclass
class CFG:
    IR_DIR: str = r"D:\BaiduNetdiskDownload\All\All_IR"
    RGB_DIR: str = r"D:\BaiduNetdiskDownload\All\All_RGB"
    XLSX: str = r"D:\BaiduNetdiskDownload\All\All_Data.xlsx"
    RESIZE_HW: Tuple[int, int] = (480, 640)
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    GPU_ID: int = 0


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to fusion *.pth")
    ap.add_argument("--out", required=True, help="output dir for fused png")
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    args = ap.parse_args()

    cfg = CFG()
    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out)

    # dataset
    if args.split == "all":
        ds = Fusion_dataset(split="all", ir_path=cfg.IR_DIR, vi_path=cfg.RGB_DIR, xlsx_path=cfg.XLSX, resize_hw=cfg.RESIZE_HW)
    else:
        train_ds = Fusion_dataset(split="train", ir_path=cfg.IR_DIR, vi_path=cfg.RGB_DIR, xlsx_path=cfg.XLSX, resize_hw=cfg.RESIZE_HW)
        ds = Fusion_dataset(split=args.split, ir_path=cfg.IR_DIR, vi_path=cfg.RGB_DIR, xlsx_path=cfg.XLSX, resize_hw=cfg.RESIZE_HW,
                            split_seed=train_ds.split_seed, train_ratio=train_ds.train_ratio)

    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # model
    model = FusionNet(output=1).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    pbar = tqdm(loader, desc=f"Infer -> {args.out}")
    for (vis, ir, _pose_dummy, _mask_dummy, names) in pbar:
        vis = vis.to(device)
        ir = ir.to(device)

        Y, Cb, Cr = RGB2YCrCb(vis)
        fused_Y = model(Y, ir)
        fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)

        for k in range(len(names)):
            save_path = os.path.join(args.out, names[k])
            save_img_single(fused_rgb[k], save_path)

    print(f"[DONE] saved to: {args.out}")


if __name__ == "__main__":
    main()
