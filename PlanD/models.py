# models.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def _adapt_first_conv(conv1: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    assert conv1.kernel_size == (7, 7)
    new_conv = nn.Conv2d(
        in_ch, conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False
    )
    with torch.no_grad():
        w = conv1.weight
        if in_ch == 1:
            new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
        elif in_ch == 4:
            new_conv.weight[:, :3].copy_(w)
            new_conv.weight[:, 3:4].copy_(w.mean(dim=1, keepdim=True))
        else:
            rep = (in_ch + 2) // 3
            ww = w.repeat(1, rep, 1, 1)[:, :in_ch]
            new_conv.weight.copy_(ww / rep)
    return new_conv

class PoseResNet50(nn.Module):
    def __init__(self, in_ch=1, pretrained=True, feat_dim=2048, dropout=0.1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.conv1 = _adapt_first_conv(backbone.conv1, in_ch)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        q = self.head(feat)
        return q
