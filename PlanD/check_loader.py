# check_loader.py
import torch
from torch.utils.data import DataLoader
from pose_dataset import PoseDataset
import matplotlib.pyplot as plt

# 配置你的路径
DATA_ROOT = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
JSON_PATH = r"D:\BaiduNetdiskDownload\data\Aqua_60度\test_pose.json"  # 用test测比较快


def test_loader():
    print("Initializing Dataset...")
    # 使用 rgbir 模式 (或者你代码里默认的逻辑)
    ds = PoseDataset(DATA_ROOT, JSON_PATH, mode="rgbir", out_size=320)

    loader = DataLoader(ds, batch_size=4, shuffle=True)

    print("Fetching one batch...")
    batch = next(iter(loader))

    # 检查字典键值
    print("Keys:", batch.keys())

    # 检查维度
    # 预期: rgb=[B,3,320,320], ir=[B,1,320,320], saliency=[B,1,320,320]
    print(f"RGB Shape: {batch['rgb'].shape}")
    print(f"IR Shape:  {batch['ir'].shape}")
    print(f"Saliency:  {batch['saliency'].shape}")
    print(f"GT Pose:   {batch['q_gt'].shape}")

    # 简单的可视化检查 (检查 RGB 和 Saliency 裁剪是否对齐)
    rgb_sample = batch['rgb'][0].permute(1, 2, 0).numpy()  # HWC
    sal_sample = batch['saliency'][0, 0].numpy()  # HW
    ir_sample = batch['ir'][0, 0].numpy()  # HW

    # 反归一化 RGB 以便显示 (粗略)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    rgb_show = (rgb_sample * std.numpy() + mean.numpy()).clip(0, 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1);
    plt.imshow(rgb_show);
    plt.title("RGB Crop")
    plt.subplot(1, 3, 2);
    plt.imshow(ir_sample, cmap='gray');
    plt.title("IR Crop")
    plt.subplot(1, 3, 3);
    plt.imshow(sal_sample, cmap='jet');
    plt.title("Saliency Crop")
    plt.show()


if __name__ == "__main__":
    test_loader()