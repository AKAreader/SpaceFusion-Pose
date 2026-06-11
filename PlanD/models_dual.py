# models_dual.py
# Python 3.8+ compatible, torchvision old/new API compatible

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import resnet18
    ResNet18_Weights = None
    _HAS_WEIGHTS = False


class CoordinateAttention(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)                 # [B,C,H,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B,C,W,1]

        y = torch.cat([x_h, x_w], dim=2)     # [B,C,H+W,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)        # [B,mip,1,W]

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class SaliencyGuidedFusion(nn.Module):
    """用显著图做空间门控：显著区域偏IR，非显著偏RGB。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv_ir = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_rgb = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, f_ir: torch.Tensor, f_rgb: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        if saliency_map.shape[-2:] != f_ir.shape[-2:]:
            sal = F.interpolate(saliency_map, size=f_ir.shape[-2:], mode="bilinear", align_corners=False)
        else:
            sal = saliency_map

        sal = sal.clamp(0.0, 1.0)
        f_ir_enh = f_ir + sal * self.conv_ir(f_ir)
        f_rgb_enh = f_rgb + (1.0 - sal) * self.conv_rgb(f_rgb)
        return 0.5 * (f_ir_enh + f_rgb_enh)


def _resnet18_pretrained():
    if _HAS_WEIGHTS and ResNet18_Weights is not None:
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    return resnet18(pretrained=True)


class PhysioSaliencyNet(nn.Module):
    """
    Dual-stream (IR/RGB) + Coordinate Attention + Saliency Guided Fusion -> quaternion
    输入：
      rgb: [B,3,H,W], ir:[B,1,H,W], saliency:[B,1,H,W]
    输出：
      q: [B,4] (w,x,y,z)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()

        base_rgb = _resnet18_pretrained()
        base_ir = _resnet18_pretrained()

        # IR conv1: 3->1，用RGB权重均值初始化，保证收敛稳定
        old_rgb_conv1 = base_rgb.conv1
        base_ir.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            base_ir.conv1.weight.copy_(old_rgb_conv1.weight.mean(dim=1, keepdim=True))

        self.enc_ir = nn.Sequential(
            base_ir.conv1, base_ir.bn1, base_ir.relu, base_ir.maxpool,
            base_ir.layer1, base_ir.layer2, base_ir.layer3
        )
        self.enc_rgb = nn.Sequential(
            base_rgb.conv1, base_rgb.bn1, base_rgb.relu, base_rgb.maxpool,
            base_rgb.layer1, base_rgb.layer2, base_rgb.layer3
        )

        self.ca_ir = CoordinateAttention(256, 256)
        self.ca_rgb = CoordinateAttention(256, 256)

        self.fusion = SaliencyGuidedFusion(256)

        self.post = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 4),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        f_ir = self.enc_ir(ir)
        f_rgb = self.enc_rgb(rgb)

        f_ir = self.ca_ir(f_ir)
        f_rgb = self.ca_rgb(f_rgb)

        f = self.fusion(f_ir, f_rgb, saliency)
        f = self.post(f)
        q = self.fc(f)
        return q
