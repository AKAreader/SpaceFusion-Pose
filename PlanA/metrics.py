# metrics.py
import torch
import math

@torch.no_grad()
def quat_angle_deg(q_pred: torch.Tensor, q_gt: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    q_pred/q_gt: [B,4] (wxyz)
    角度误差（度），考虑 q 与 -q 等价
    """
    q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + eps)
    q_gt   = q_gt   / (q_gt.norm(dim=1, keepdim=True) + eps)
    dot = (q_pred * q_gt).sum(dim=1).abs().clamp(0.0, 1.0)
    ang = 2.0 * torch.acos(dot) * (180.0 / math.pi)
    return ang

def quat_loss(q_pred: torch.Tensor, q_gt: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    L = 1 - |<q_pred, q_gt>|
    """
    q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + eps)
    q_gt   = q_gt   / (q_gt.norm(dim=1, keepdim=True) + eps)
    dot = (q_pred * q_gt).sum(dim=1).abs().clamp(0.0, 1.0)
    return (1.0 - dot).mean()
