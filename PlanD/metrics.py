# metrics.py
# 四元数统一：wxyz，sign-invariant，geodesic loss 与 angle 指标一致

import math
import torch


def quat_normalize_wxyz(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # q: [B,4] wxyz
    q = q / (q.norm(dim=1, keepdim=True) + eps)
    # canonical sign: w>=0
    sign = torch.where(q[:, :1] < 0, -torch.ones_like(q[:, :1]), torch.ones_like(q[:, :1]))
    return q * sign


def quat_angle_deg(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    q_pred = quat_normalize_wxyz(q_pred)
    q_gt = quat_normalize_wxyz(q_gt)
    dot = (q_pred * q_gt).sum(dim=1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot) * (180.0 / math.pi)
    return ang


def quat_geodesic_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    # loss = 1 - dot^2  (稳定、与角度一致、对 q 与 -q 等价)
    q_pred = quat_normalize_wxyz(q_pred)
    q_gt = quat_normalize_wxyz(q_gt)
    dot = (q_pred * q_gt).sum(dim=1).abs().clamp(0.0, 1.0)
    loss = 1.0 - dot * dot
    return loss.mean()


# 兼容你原来脚本里可能叫 quat_loss 的写法
quat_loss = quat_geodesic_loss
