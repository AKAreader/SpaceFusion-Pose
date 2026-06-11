# loss.py
#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    """保留接口：若你完全关闭语义分支，这个类不会被用到。"""
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super().__init__()
        self.register_buffer("thresh_tensor", -torch.log(torch.tensor(float(thresh), dtype=torch.float32)))
        self.n_min = int(n_min)
        self.ignore_lb = int(ignore_lb)
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        thresh = self.thresh_tensor.to(loss.device)
        if loss[self.n_min] > thresh:
            loss = loss[loss > thresh]
        else:
            loss = loss[: self.n_min]
        return torch.mean(loss)


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernely = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class Fusionloss(nn.Module):
    """
    无监督融合损失（不依赖语义标签）：
      L_int  = L1( max(Y_vis, IR), Y_fuse )
      L_grad = L1( max(|∇Y_vis|, |∇IR|), |∇Y_fuse| )
      L = L_int + 10 * L_grad
    """
    def __init__(self):
        super().__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, labels, generate_img, i):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        loss_total = loss_in + 10.0 * loss_grad
        return loss_total, loss_in, loss_grad
