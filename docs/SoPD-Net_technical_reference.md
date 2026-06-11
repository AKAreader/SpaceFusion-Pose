# SoPD-Net 网络结构技术参考文件

## 1. 文档用途

本文档用于统一说明当前仓库中 SoPD-Net 的实际实现，作为以下工作的技术依据：

- 论文 Methods 和 Supplementary Methods 撰写；
- 网络架构图复核与重绘；
- 消融实验定义；
- 推理流程复现；
- 代码与论文表述一致性检查。

本文档不是绘图提示词。文档优先描述实际代码行为，不使用未经实现验证的概念包装。

## 2. 最终版本范围

论文主模型应以以下实现为准：

- 基础融合网络：`FusionNet.py`
- SoPD-Net 训练入口：`train_Cv2.py`
- RGB 与 YCbCr 转换：`utils.py`
- 最终主实验日志：`runs_fusion/routeC_v2_ssim_safe_bs1_acc8/20260416-143257/train.log`
- 统一下游姿态评估：`pose_unified_table.py`

最终主实验日志表明：

| 配置项 | 最终主实验状态 |
| --- | --- |
| `USE_CA_POSE` | `True` |
| `USE_SAL_RECTIFY` | `False` |
| `POSE_LOSS_TO_TOTAL` | `True` |
| `USE_SSIM_LOSS` | `True` |
| `GATE_WARMUP_EPOCHS` | `2` |
| `G_MIN` | 默认 `0.15` |

因此，论文主图中不应把 saliency rectification 画成 SoPD-Net 的必要模块。

## 3. 总体任务

SoPD-Net 用于可见光与红外图像融合，并在训练阶段引入姿态估计辅助监督，使融合结果保留与航天器姿态相关的结构信息。

输入：

- 可见光图像：`I_vis ∈ R^(B×3×H×W)`
- 红外图像：`I_ir ∈ R^(B×1×H×W)`

输出：

- 融合亮度图：`Y_hat ∈ R^(B×1×H×W)`
- 融合 RGB 图像：`I_fused ∈ R^(B×3×H×W)`

总体推理流程：

```text
Visible RGB
  └─ RGB-to-YCbCr ──> Y, Cb, Cr
                         │
                         ├─ Y ──────────────┐
Infrared IR ────────────────────────────────┼─> FusionNet backbone ─> F_base
                                           │
Y, IR, F_base, |Y - IR| ───────────────────┴─> GateNet ─> g
Y, IR ────────────────────────────────────────> pixel-wise max ─> M

Y_hat = g ⊙ F_base + (1 - g) ⊙ M

Y_hat, Cb, Cr ─> YCbCr-to-RGB ─> Fused RGB
```

其中：

- `F_base` 是双分支基础融合网络的输出；
- `M = max(Y, IR)` 是逐像素强度回退项；
- `g` 是 GateNet 生成的空间信任图；
- `Cb` 和 `Cr` 直接来自可见光图像，不参与融合主干训练。

## 4. 颜色空间分解与重建

### 4.1 RGB-to-YCbCr

代码位置：`utils.py::RGB2YCrCb`

可见光图像首先被确定性地转换为亮度和色度分量：

```text
Y  = 0.299 R + 0.587 G + 0.114 B
Cr = (R - Y) × 0.713 + 0.5
Cb = (B - Y) × 0.564 + 0.5
```

随后将 `Y`、`Cr` 和 `Cb` 限制在 `[0, 1]`。代码接口返回顺序为：

```text
return Y, Cb, Cr
```

### 4.2 Chroma bypass

只有 `Y` 进入融合网络。`Cb` 和 `Cr` 作为可见光色度旁路，直接用于最终 RGB 重建。

这部分不应被描述为可训练模块，也不应被命名为“chroma fusion”。

### 4.3 YCbCr-to-RGB

代码位置：`utils.py::YCbCr2RGB`

融合亮度 `Y_hat` 与原始可见光色度 `Cb`、`Cr` 组合后，通过确定性的 YCbCr-to-RGB 变换输出 `I_fused`。

正确表达：

```text
[Y_hat, Cb, Cr] ─> YCbCr-to-RGB ─> I_fused
```

这一步是通道组合与颜色空间转换，不是张量相加。

## 5. 双分支基础融合网络 FusionNet

代码位置：`FusionNet.py::FusionNet`

### 5.1 网络特征

FusionNet 包含两个对称分支：

- 可见光亮度分支：输入 `Y`
- 红外分支：输入 `I_ir`

两个分支始终保持空间分辨率 `H×W`。当前实现中不存在：

- 下采样；
- 上采样；
- U-Net skip connection；
- 特征金字塔；
- Transformer；
- 融合主干内部的注意力模块。

### 5.2 编码分支

两个分支结构相同：

| 阶段 | 运算 | 输入通道 | 输出通道 | 空间尺寸 |
| --- | --- | ---: | ---: | --- |
| 输入 | `Y` 或 `I_ir` | 1 | 1 | `H×W` |
| Stem | `3×3 Conv + LeakyReLU` | 1 | 16 | `H×W` |
| RGBD block 1 | Dense + Sobel 双路增强 | 16 | 32 | `H×W` |
| RGBD block 2 | Dense + Sobel 双路增强 | 32 | 48 | `H×W` |

编码输出：

```text
F_Y  ∈ R^(B×48×H×W)
F_IR ∈ R^(B×48×H×W)
```

### 5.3 特征拼接与解码

两个编码输出在通道维度拼接：

```text
F_cat = Concat(F_Y, F_IR)
F_cat ∈ R^(B×96×H×W)
```

随后进入四层卷积解码器：

| 阶段 | 运算 | 输入通道 | 输出通道 | 激活 |
| --- | --- | ---: | ---: | --- |
| Decoder 1 | `3×3 Conv` | 96 | 64 | LeakyReLU |
| Decoder 2 | `3×3 Conv` | 64 | 32 | LeakyReLU |
| Decoder 3 | `3×3 Conv` | 32 | 16 | LeakyReLU |
| Decoder 4 | `3×3 Conv` | 16 | 1 | `tanh(x)/2 + 0.5` |

输出：

```text
F_base ∈ R^(B×1×H×W)
```

正确结构：

```text
F_Y  ─┐
      ├─ Concat ─> four-layer 3×3 Conv decoder ─> F_base
F_IR ─┘
```

错误结构：

```text
F_Y  ─┐
      ├─ Concat ─> 1×1 Conv ─> F_base
F_IR ─┘
```

`1×1 Conv` 存在于 RGBD block 内部，不位于双分支拼接之后。

## 6. RGBD block

代码位置：

- `FusionNet.py::DenseBlock`
- `FusionNet.py::Sobelxy`
- `FusionNet.py::RGBD`

RGBD block 接收：

```text
X ∈ R^(B×C_in×H×W)
```

并产生：

```text
X_out ∈ R^(B×C_out×H×W)
```

### 6.1 Dense feature branch

Dense branch 通过逐步拼接保留原始特征和新提取特征：

```text
Z1 = Conv3×3(X)
Z2 = Concat(X, Z1)
Z3 = Conv3×3(Z2)
Z4 = Concat(Z2, Z3)
F_dense = Conv1×1(Z4)
```

通道变化：

```text
C_in ─> C_in
Concat ─> 2C_in
2C_in ─> C_in
Concat ─> 3C_in
1×1 Conv: 3C_in ─> C_out
```

### 6.2 Sobel edge branch

Sobel branch 提取显式梯度特征：

```text
F_edge = Conv1×1(Sobelxy(X))
```

其中 `Sobelxy` 使用水平和垂直 Sobel 卷积，并计算：

```text
|Sobel_x(X)| + |Sobel_y(X)|
```

### 6.3 分支合并

```text
X_out = LeakyReLU(F_dense + F_edge)
```

适合论文图中的紧凑表达：

```text
Input X
 ├─ Dense branch: Conv3×3 ─> concat ─> Conv3×3 ─> concat ─> Conv1×1
 └─ Edge branch : Sobelxy ─> Conv1×1

Add ─> LeakyReLU ─> Output
```

## 7. Trust-guided fallback fusion

代码位置：

- `train_Cv2.py::GateNet`
- `train_Cv2.py::FusionWithGate`

### 7.1 Fallback map

逐像素回退项为：

```text
M = max(Y, I_ir)
```

注意：`F_base` 不进入 `max` 节点。

### 7.2 GateNet 输入

GateNet 接收四通道拼接特征：

```text
F_gate = Concat(Y, I_ir, F_base, |Y - I_ir|)
F_gate ∈ R^(B×4×H×W)
```

四个输入分别表达：

| 输入 | 含义 |
| --- | --- |
| `Y` | 可见光亮度 |
| `I_ir` | 红外强度 |
| `F_base` | 学习式基础融合结果 |
| `|Y - I_ir|` | 两种模态的局部差异 |

### 7.3 GateNet 结构

| 阶段 | 运算 | 输入通道 | 输出通道 | 激活 |
| --- | --- | ---: | ---: | --- |
| Gate 1 | `3×3 Conv` | 4 | 16 | ReLU |
| Gate 2 | `3×3 Conv` | 16 | 16 | ReLU |
| Gate 3 | `3×3 Conv` | 16 | 1 | Sigmoid |

为了防止门控完全关闭学习式基础融合结果，代码使用下界约束：

```text
g = g_min + (1 - g_min) × sigmoid(logits)
```

训练脚本默认：

```text
g_min = 0.15
```

### 7.4 最终融合亮度

```text
Y_hat = g ⊙ F_base + (1 - g) ⊙ M
```

其中 `⊙` 表示逐元素乘法。

这使模型可以在学习式融合结果与强度保留回退项之间进行空间自适应平衡。

## 8. 训练期结构感知显著性

代码位置：`train_Cv2.py::joint_saliency`

结构感知显著图仅用于训练损失加权。最终主实验中不应把它画成必要的推理模块。

### 8.1 Sobel 梯度

首先计算：

```text
G_Y  = SobelMagnitude(Y)
G_IR = SobelMagnitude(I_ir)
```

训练脚本中的 `_sobel_mag` 使用：

```text
sqrt(G_x^2 + G_y^2 + 1e-12)
```

### 8.2 联合显著图

```text
S_raw = 0.5 G_Y + 0.5 G_IR
```

对每张图执行 min-max 归一化：

```text
S_norm = (S_raw - min(S_raw)) / (max(S_raw) - min(S_raw) + 1e-8)
```

阈值化与膨胀：

```text
ROI = 1[S_norm > threshold]
ROI = MaxPool(ROI, kernel_size = 9)
```

最终：

```text
S = clamp(0.7 S_norm + 0.3 ROI, 0, 1)
```

默认参数：

| 参数 | 默认值 |
| --- | ---: |
| `SAL_THRESH` | `0.55` |
| `SAL_ROI_EXPAND_K` | `9` |
| `LAMBDA_FG` | `2.5` |
| `LAMBDA_BG` | `0.8` |

### 8.3 显著性权重

```text
W = lambda_fg S + lambda_bg (1 - S)
```

显著区域获得更高权重，背景区域保留较低但非零权重。

## 9. 融合损失

代码位置：

- `train_Cv2.py::loss_intensity`
- `train_Cv2.py::loss_grad`
- `train_Cv2.py::loss_intensity_roi`
- `train_Cv2.py::loss_grad_roi`

### 9.1 基础亮度损失

```text
T_int = max(Y, I_ir)
L_int = mean(|Y_hat - T_int|)
```

### 9.2 基础梯度损失

```text
T_grad = max(Grad(Y), Grad(I_ir))
L_grad = mean(|Grad(Y_hat) - T_grad|)
```

### 9.3 显著性加权损失

```text
L_int_roi  = mean(W × |Y_hat - T_int|)
L_grad_roi = mean(W × |Grad(Y_hat) - T_grad|)
```

### 9.4 渐进式组合

训练前四个 epoch 使用 `alpha` 将基础损失逐步切换为显著性加权损失：

```text
L_fus_base = W_INT × L_int + W_GRAD × L_grad
L_fus_roi  = W_INT × L_int_roi + W_GRAD × L_grad_roi

L_fus = (1 - alpha) L_fus_base + alpha L_fus_roi
```

默认：

```text
W_INT  = 1.0
W_GRAD = 10.0
RAMP_EPOCHS = 4
```

## 10. Base auxiliary loss

为了避免 GateNet 绕过 `F_base`，导致基础融合网络学习不足，代码对 `F_base` 施加同类辅助融合损失：

```text
L_base = fusion_loss(F_base, Y, I_ir, S)
```

默认权重：

```text
lambda_aux = 0.25
```

## 11. Gate regularization

代码位置：`train_Cv2.py::tv_loss`

门控图使用总变分正则：

```text
L_TV(g) =
  mean(|g[:, :, 1:, :] - g[:, :, :-1, :]|)
  +
  mean(|g[:, :, :, 1:] - g[:, :, :, :-1]|)
```

默认权重：

```text
lambda_TV = 0.02
```

代码还保留可选的 gate mean regularization：

```text
lambda_gmean × (mean(g) - 0.5)^2
```

但最终默认：

```text
lambda_gmean = 0.00
```

因此论文主损失公式中可不展示该项。

## 12. SSIM loss

代码位置：`train_Cv2.py::ssim_fusion_loss`

SoPD-Net 最终主实验启用了可微 SSIM loss：

```text
L_SSIM = 1 - 0.5 [SSIM(Y_hat, Y) + SSIM(Y_hat, I_ir)]
```

默认配置：

| 参数 | 默认值 |
| --- | ---: |
| `USE_SSIM_LOSS` | `True` |
| `LAMBDA_SSIM` | `0.14` |
| `SSIM_START_EPOCH` | `5` |
| `SSIM_RAMP_EPOCHS` | `4` |
| `SSIM_WIN` | `11` |
| `SSIM_SIGMA` | `1.5` |

SSIM loss 是训练期约束，不增加推理模块。

## 13. Pose-guided learning

代码位置：

- `train_Cv2.py::CoordAtt`
- `train_Cv2.py::PoseRegressor`
- `train_Cv2.py::quat_loss`

### 13.1 定位

姿态估计头是 SoPD-Net 训练期间使用的辅助任务分支。它通过姿态损失反向传播影响融合网络，使融合结果更有利于航天器姿态估计。

部署融合网络时，姿态头不是必需模块。

### 13.2 输入

最终默认：

```text
POSE_INPUT_RGB = True
```

因此姿态头输入为：

```text
I_fused = YCbCr2RGB(Y_hat, Cb, Cr)
```

### 13.3 PoseRegressor 结构

| 阶段 | 运算 | 输入通道 | 输出通道 | 空间变化 |
| --- | --- | ---: | ---: | --- |
| Pose block 1 | `3×3 Conv, stride 2 + BN + ReLU + CA` | 3 | 32 | `H/2 × W/2` |
| Pose block 2 | `3×3 Conv, stride 2 + BN + ReLU + CA` | 32 | 64 | `H/4 × W/4` |
| Pose block 3 | `3×3 Conv, stride 2 + BN + ReLU + CA` | 64 | 128 | `H/8 × W/8` |
| Pose block 4 | `3×3 Conv, stride 2 + BN + ReLU + CA` | 128 | 256 | `H/16 × W/16` |
| Head | Global average pooling + Flatten | 256 | 256 | `1×1` |
| FC 1 | Linear + ReLU | 256 | 256 | - |
| FC 2 | Linear | 256 | 4 | - |

输出：

```text
q_hat ∈ R^(B×4)
```

### 13.4 Coordinate Attention

每个 pose convolution block 后均应用 Coordinate Attention：

```text
X ─> pool along H and W
  ─> concatenate
  ─> 1×1 Conv + BN + ReLU
  ─> split into H and W attention
  ─> sigmoid
  ─> X × A_H × A_W
```

Coordinate Attention 只存在于姿态辅助头中，不存在于融合 backbone 中。

### 13.5 Quaternion loss

预测和 GT 四元数先执行归一化：

```text
q_hat = q_hat / ||q_hat||
q_gt  = q_gt  / ||q_gt||
```

随后计算符号不敏感损失：

```text
L_pose = mean(1 - (q_hat · q_gt)^2)
```

默认配置：

| 参数 | 默认值 |
| --- | ---: |
| `USE_POSE_SUPERVISION` | `True` |
| `POSE_INPUT_RGB` | `True` |
| `USE_CA_POSE` | `True` |
| `LAMBDA_POSE` | `0.08` |
| `POSE_START_EPOCH` | `3` |
| `POSE_RAMP_EPOCHS` | `4` |
| `POSE_GRAD_CLIP` | `1.0` |

## 14. 最终训练目标

根据最终主实验，建议论文中使用以下总目标：

```text
L_total =
    L_fus
  + lambda_aux  L_base
  + lambda_TV   L_TV(g)
  + lambda_SSIM L_SSIM
  + lambda_pose L_pose
```

其中：

- `L_fus`：由基础融合损失渐进切换至显著性加权融合损失；
- `L_base`：对 `F_base` 的辅助融合约束；
- `L_TV(g)`：门控图空间平滑正则；
- `L_SSIM`：结构相似性约束；
- `L_pose`：训练期姿态引导约束。

## 15. 训练调度

### 15.1 Gate warmup

前两个 epoch 强制：

```text
Y_hat = F_base
g = 1
```

从而先训练基础融合网络，再逐步启用 trust-guided fallback fusion。

### 15.2 Saliency-weighted loss ramp

```text
alpha: 0 → 1
RAMP_EPOCHS = 4
```

### 15.3 Pose loss ramp

```text
epoch 1-2: pose weight = 0
epoch 3:   pose weight = 0.02
epoch 4:   pose weight = 0.04
epoch 5:   pose weight = 0.06
epoch 6+:  pose weight = 0.08
```

### 15.4 SSIM loss ramp

```text
epoch 1-4: SSIM weight = 0
epoch 5:   SSIM weight = 0.035
epoch 6:   SSIM weight = 0.070
epoch 7:   SSIM weight = 0.105
epoch 8+:  SSIM weight = 0.140
```

## 16. 可选但最终关闭的结构

代码保留 saliency rectification：

```text
Y_hat <- Y_hat + rectify_alpha × S × (F_base - Y_hat)
```

但最终主实验日志中：

```text
USE_SAL_RECTIFY = False
```

因此：

- 不应在主网络图中作为必要模块展示；
- 如需讨论，只适合放入补充材料或失败实验分析；
- 不应把它写入 SoPD-Net 的核心贡献。

## 17. 推理伪代码

```python
Y, Cb, Cr = RGB2YCrCb(I_vis)

F_base = FusionNet(Y, I_ir)
F_gate = concat(Y, I_ir, F_base, abs(Y - I_ir))
g = GateNet(F_gate)
M = maximum(Y, I_ir)

Y_hat = g * F_base + (1.0 - g) * M
I_fused = YCbCr2RGB(Y_hat, Cb, Cr)
```

部署推理不需要：

- saliency map；
- SSIM loss；
- pose head；
- quaternion GT；
- 任何训练期损失。

## 18. 主图应表达的结构

主图至少应准确表达：

```text
Visible RGB ─> RGB-to-YCbCr ─> Y ───────────────────┐
                          └─> Cb, Cr bypass ───────┼────────────────────┐
                                                  │                    │
Infrared IR ──────────────────────────────────────┼─> FusionNet ─> F_base
                                                  │
Y, IR ────────────────────────────────────────────┴─> max ─> M
[Y, IR, F_base, |Y - IR|] ──────────────────────────> GateNet ─> g

F_base, M, g ─> trust-guided blend ─> Y_hat
Y_hat, Cb, Cr ─> YCbCr-to-RGB ─> Fused RGB
```

训练期约束应放在独立区域，以虚线连接：

```text
Structure-aware fusion loss
Base auxiliary loss
Gate TV regularization
SSIM loss
Pose-guided quaternion loss
```

## 19. 下游统一姿态评估与训练期 pose head 的区别

`pose_unified_table.py` 中的统一下游评估会针对每一种输入重新训练一个独立的 `PoseRegressorPlain`：

```text
Visible only
Infrared only
SeAFusion
SoPD-Net
SoPD-Net w/o pose loss
```

该评估协议用于公平比较融合结果是否有利于姿态估计。

需要区分：

| 分支 | 作用 | 是否属于 SoPD-Net 推理 | 是否为 CA head |
| --- | --- | --- | --- |
| `train_Cv2.py::PoseRegressor` | 融合训练期辅助监督 | 否 | 是 |
| `pose_unified_table.py::PoseRegressorPlain` | 下游公平评估器 | 否 | 否 |

论文主图可以展示训练期 CA pose head，但不能把它与统一下游评估器混为一谈。

## 20. 已识别的实现注意事项

以下事项应在论文定稿或复跑实验前核查。

### 20.1 训练与统一评估的 `g_min` 不一致

训练脚本：

```text
train_Cv2.py: G_MIN 默认值 = 0.15
```

统一评估加载器：

```text
doctor.py::_load_fusion_model
g_min = 0.10
```

如果最终训练运行没有通过环境变量覆盖 `G_MIN`，则统一评估推理与训练期模型定义存在轻微差异。建议统一为训练时使用的值后复跑评估。

### 20.2 FusionNet 中定义了 BN，但前向未调用

`ConvBnLeakyRelu2d` 和 `ConvBnTanh2d` 创建了 `BatchNorm2d` 层，但其 `forward()` 实际仅调用卷积和激活函数，没有调用 `self.bn`。

因此论文中不应宣称 FusionNet decoder 使用 BatchNorm，除非后续修改代码并重新训练。

### 20.3 RGBD 内 Sobel 卷积当前不是冻结算子

`FusionNet.py::Sobelxy` 使用 Sobel 核初始化 depthwise convolution，但未设置：

```python
requires_grad = False
```

因此这些卷积权重可以在训练过程中更新。准确表述应为：

```text
Sobel-initialized depthwise edge branch
```

而不是：

```text
fixed Sobel operator
```

如果论文希望宣称固定 Sobel 算子，需要先冻结权重并重新训练验证。

### 20.4 训练损失中的 Sobel 与 RGBD Sobel branch 不同

- RGBD block 内部使用 `Sobelxy` 类，其卷积核可训练；
- 训练显著图和梯度损失使用 `_sobel_mag` 函数，每次构造固定 Sobel tensor。

两者不能在论文中混写为同一个算子。

## 21. 实现文件映射

| 内容 | 文件与对象 |
| --- | --- |
| RGB-to-YCbCr | `utils.py::RGB2YCrCb` |
| YCbCr-to-RGB | `utils.py::YCbCr2RGB` |
| Dense block | `FusionNet.py::DenseBlock` |
| RGBD block | `FusionNet.py::RGBD` |
| Sobel-initialized edge branch | `FusionNet.py::Sobelxy` |
| 双分支基础融合网络 | `FusionNet.py::FusionNet` |
| GateNet | `train_Cv2.py::GateNet` |
| Trust-guided blend | `train_Cv2.py::FusionWithGate` |
| 显著图 | `train_Cv2.py::joint_saliency` |
| 融合损失 | `train_Cv2.py::loss_intensity*`, `loss_grad*` |
| SSIM loss | `train_Cv2.py::ssim_fusion_loss` |
| Gate TV loss | `train_Cv2.py::tv_loss` |
| Coordinate Attention | `train_Cv2.py::CoordAtt` |
| 训练期姿态辅助头 | `train_Cv2.py::PoseRegressor` |
| 四元数损失 | `train_Cv2.py::quat_loss` |
| 统一下游姿态评估 | `pose_unified_table.py` |

## 22. 一句话技术摘要

SoPD-Net 将可见光亮度与红外图像送入包含 dense feature branch 和 Sobel-initialized edge branch 的双流融合骨干网络，并通过空间信任图在学习式基础融合结果与逐像素强度回退项之间进行自适应组合；训练阶段进一步使用结构感知加权、SSIM、门控平滑和姿态四元数辅助损失提升融合结果的结构保真度与下游姿态估计价值。
