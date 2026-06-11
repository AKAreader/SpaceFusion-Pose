# loss_geo.py
import torch
import torch.nn as nn

class GeometricLoss(nn.Module):
    """
    你的工程里曾经引入几何损失，但没有提供稳定的 3D 点/投影约束。
    为避免“代码缺文件/跑不起来”，这里提供安全占位：
    - 当 geo_weight=0：完全不参与训练（推荐先这样）
    - 如果以后你要做真实几何损失，我再按你的相机/模型点补完整版本
    """
    def __init__(self):
        super().__init__()

    def forward(self, q_pred, q_gt, t_gt, K):
        return torch.zeros((), device=q_pred.device, dtype=q_pred.dtype)
