# geometry_loss.py
import torch
import torch.nn as nn


def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: [B,4] wxyz
    return R: [B,3,3]
    """
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=-1)

    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z

    R = torch.stack([
        ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy),
        2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx),
        2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz
    ], dim=-1).reshape(-1, 3, 3)
    return R


def project_points(pts_body: torch.Tensor,
                   q_wxyz: torch.Tensor,
                   t_xyz: torch.Tensor,
                   K: torch.Tensor) -> torch.Tensor:
    """
    pts_body: [N,3] in body frame (meters)
    q_wxyz:   [B,4]
    t_xyz:    [B,3] translation in camera frame (meters)
    K:        [B,3,3] intrinsics for the *ROI-resized* image (pixels)

    Return:
      uv: [B,N,2] in pixel coords (on out_size image)
    """
    B = q_wxyz.shape[0]
    N = pts_body.shape[0]

    R = quat_wxyz_to_rotmat(q_wxyz)  # [B,3,3]
    pts = pts_body.unsqueeze(0).expand(B, N, 3)  # [B,N,3]

    # X_cam = R * X_body + t
    X_cam = torch.bmm(pts, R.transpose(1, 2)) + t_xyz.unsqueeze(1)  # [B,N,3]

    X = X_cam[..., 0]
    Y = X_cam[..., 1]
    Z = X_cam[..., 2].clamp(min=1e-6)

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return torch.stack([u, v], dim=-1)  # [B,N,2]


def make_default_sat_points(device="cpu", scale: float = 1.0) -> torch.Tensor:
    """
    一个“无CAD也能用”的虚拟稀疏点集（单位：米）。
    你后续拿到真实尺寸后，只要改这里即可。
    """
    pts = torch.tensor([
        [0.0, 0.0, 0.0],     # center
        [0.5, 0.5, 1.0],     # body corners (rough)
        [0.5, -0.5, 1.0],
        [-0.5, -0.5, 1.0],
        [-0.5, 0.5, 1.0],
        [3.0, 0.0, 0.0],     # solar panel tips
        [-3.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],     # antenna-ish
    ], dtype=torch.float32, device=device)
    return pts * float(scale)


class GeometricReprojLoss(nn.Module):
    """
    loss = mean( huber( ||uv_pred - uv_gt||_2 ) )
    """
    def __init__(self, pts_body: torch.Tensor, huber_delta: float = 10.0):
        super().__init__()
        self.register_buffer("pts_body", pts_body)
        self.huber_delta = float(huber_delta)

    def forward(self,
                q_pred: torch.Tensor,
                q_gt: torch.Tensor,
                t_gt: torch.Tensor,
                K: torch.Tensor) -> torch.Tensor:
        uv_pred = project_points(self.pts_body, q_pred, t_gt, K)
        uv_gt   = project_points(self.pts_body, q_gt,   t_gt, K)

        diff = uv_pred - uv_gt  # [B,N,2]
        err = torch.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2 + 1e-9)  # [B,N]

        d = self.huber_delta
        # Huber on scalar err
        loss = torch.where(err < d, 0.5 * (err ** 2) / d, err - 0.5 * d)
        return loss.mean()
