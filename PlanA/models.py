# models.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def _adapt_first_conv(conv1: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """
    将 resnet 的 conv1 从 3通道适配到 1通道或 4通道，并尽量复用预训练权重
    """
    assert conv1.kernel_size == (7, 7)
    new_conv = nn.Conv2d(
        in_ch, conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False
    )

    with torch.no_grad():
        w = conv1.weight  # [64,3,7,7]
        if in_ch == 1:
            # 平均到单通道
            new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
        elif in_ch == 4:
            # 前3通道复制RGB权重，第4通道用RGB均值初始化
            new_conv.weight[:, :3].copy_(w)
            new_conv.weight[:, 3:4].copy_(w.mean(dim=1, keepdim=True))
        else:
            # 其他情况：重复/截断（很少用）
            rep = (in_ch + 2) // 3
            ww = w.repeat(1, rep, 1, 1)[:, :in_ch]
            new_conv.weight.copy_(ww / rep)
    return new_conv

class PoseResNet50(nn.Module):
    def __init__(self, in_ch: int, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(weights=None)

        if in_ch != 3:
            backbone.conv1 = _adapt_first_conv(backbone.conv1, in_ch)

        # 去掉分类头
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 4)  # quat wxyz
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        q = self.head(feat)
        # 输出不在这里强制归一化（loss/metric里会归一化）
        return q
