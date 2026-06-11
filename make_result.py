import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# 引入你的网络定义 (确保 FusionNet.py 和 utils.py 在同级目录)
from FusionNet import FusionNet
from utils import RGB2YCrCb, YCbCr2RGB

# ================= 1. 配置区域 =================
# 原始图片路径
DIR_IR = r"E:\ALL\IR"
DIR_VIS = r"E:\ALL\RGB"

# 训练好的权重路径 (你刚才报错的那个路径)
MODEL_PATH = r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\baseline\20251228-234201\ckpt\fusion_best_pose.pth"

# 结果保存路径
OUTPUT_DIR = r"E:\Other\seafusion"

# 是否调整尺寸 (根据你训练时的设置，通常是 (480, 640))
RESIZE_HW = None  # 格式: (H, W)


# ==============================================

def load_img(path, size=None):
    """读取图片并转为 Tensor"""
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize((size[1], size[0]), Image.Resampling.BILINEAR)

    # 归一化到 0-1 并转 Tensor: [C, H, W]
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return img


def save_tensor_img(tensor, save_path):
    """保存 Tensor 为图片"""
    # tensor: [1, C, H, W] -> [H, W, C]
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def main():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 2. 加载模型 (包含修复逻辑)
    print(f"加载模型: {MODEL_PATH}")
    model = FusionNet(output=1).to(device)

    if os.path.exists(MODEL_PATH):
        # --- 核心修复代码开始 ---
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        new_state_dict = {}
        for k, v in checkpoint.items():
            # 情况1：如果是 fusion. 开头，去掉前缀
            if k.startswith('fusion.'):
                new_key = k[7:]  # 去掉 'fusion.'
                new_state_dict[new_key] = v
            # 情况2：如果是 gate. 开头 (姿态网络参数)，直接忽略
            elif k.startswith('gate.'):
                continue
            # 情况3：其他情况，保持原样
            else:
                new_state_dict[k] = v

        # 加载清洗后的权重，strict=False 允许容错
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ 权重加载成功 (已自动修复键名不匹配问题)！")
        except Exception as e:
            print(f"❌ 权重加载依然失败: {e}")
            return
        # --- 核心修复代码结束 ---
    else:
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    model.eval()

    # 3. 获取文件列表
    files = [f for f in os.listdir(DIR_IR) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    print(f"共找到 {len(files)} 张图片，开始全量生成...")

    with torch.no_grad():
        for fname in tqdm(files):
            p_ir = os.path.join(DIR_IR, fname)
            p_vis = os.path.join(DIR_VIS, fname)

            # 检查可见光图是否存在
            if not os.path.exists(p_vis):
                # 尝试找不同后缀的同名文件
                base = os.path.splitext(fname)[0]
                found = False
                for ext in ['.png', '.jpg', '.bmp']:
                    temp_path = os.path.join(DIR_VIS, base + ext)
                    if os.path.exists(temp_path):
                        p_vis = temp_path
                        found = True
                        break
                if not found:
                    # print(f"跳过: 找不到对应的 RGB 图片 {fname}")
                    continue

            # 4. 读取并预处理
            vis_tensor = load_img(p_vis, RESIZE_HW).to(device)

            # IR 读取逻辑
            ir_pil = Image.open(p_ir).convert('L')
            if RESIZE_HW is not None:
                ir_pil = ir_pil.resize((RESIZE_HW[1], RESIZE_HW[0]), Image.Resampling.BILINEAR)
            ir_np = np.array(ir_pil, dtype=np.float32) / 255.0
            ir_tensor = torch.from_numpy(ir_np).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

            # 5. 推理 (只融合 Y 通道)
            Y, Cb, Cr = RGB2YCrCb(vis_tensor)
            fused_Y = model(Y, ir_tensor)
            fused_rgb = YCbCr2RGB(fused_Y, Cb, Cr)

            # 6. 保存
            save_path = os.path.join(OUTPUT_DIR, fname)
            save_tensor_img(fused_rgb, save_path)

    print(f"\n🎉 全部完成！所有图片已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()