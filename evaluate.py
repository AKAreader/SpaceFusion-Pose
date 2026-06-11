import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm

# ================= 1. 路径配置区域 (已更新) =================
# 源图像路径 (请确认这里是你存放原图的位置！)
DIR_VIS = r"E:\ALL\RGB"  # 可见光原图
DIR_IR = r"E:\ALL\IR"  # 红外原图

# 6种算法的输出路径 (已填入你提供的路径)
ALGO_DIRS = {
    'DenseFuse': r"E:\Other\dense",
    'Pixel': r"E:\Other\pixel",
    'Pyramid': r"E:\Other\pyramid",
    'RFN-Nest': r"E:\Other\rfn",
    'SeAFusion': r"E:\Other\seafusion",
    'SoPD-Net': r"E:\Other\SoPD-Net"  # 你的自研算法
}

# 结果保存路径
OUTPUT_DIR = r"E:\Other\Final_Comparison_Report"


# ========================================================

def read_img(path, gray=True):
    """读取图片 (处理中文路径)"""
    if not os.path.exists(path): return None
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        return img
    except:
        return None


# --- 指标计算函数 ---
def EN(img):
    """信息熵"""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-8)
    return -np.sum(hist * np.log2(hist + 1e-8))


def SD(img):
    """标准差"""
    return np.std(img)


def SF(img):
    """空间频率"""
    RF = np.diff(img, axis=0)
    CF = np.diff(img, axis=1)
    return np.sqrt(np.mean(RF ** 2) + np.mean(CF ** 2))


def AG(img):
    """平均梯度"""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def compute_Qabf(imgF, imgA, imgB):
    """边缘保持度 Qabf (核心指标)"""

    def get_grad(img):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
        gy = convolve2d(img, sobel_y, mode='same', boundary='symm')
        return np.sqrt(gx ** 2 + gy ** 2), np.arctan2(gy, gx)

    gA, aA = get_grad(imgA);
    gB, aB = get_grad(imgB);
    gF, aF = get_grad(imgF)

    def sigmoid(x, T=0.9994, k=-15, D=0.5):
        return D + (1 - D) / (1 + np.exp(-k * (x - T)))

    def preservation(gS, aS, gF, aF):
        Qg = np.where(gS > gF, gF / (gS + 1e-8), gS / (gF + 1e-8))
        diff_a = np.abs(aS - aF)
        diff_a[diff_a > np.pi] = 2 * np.pi - diff_a[diff_a > np.pi]
        Qa = np.maximum(0, 1 - diff_a / (np.pi / 2))
        return sigmoid(Qg) * sigmoid(Qa, T=0.9879, k=-22, D=0.8)

    Q_AF = preservation(gA, aA, gF, aF)
    Q_BF = preservation(gB, aB, gF, aF)
    return np.sum(gA * Q_AF + gB * Q_BF) / np.sum(gA + gB + 1e-8)


def evaluate_one(img_f, img_ir, img_vis):
    # 统一尺寸
    h, w = img_ir.shape
    if img_f.shape != (h, w): img_f = cv2.resize(img_f, (w, h))
    if img_vis.shape != (h, w): img_vis = cv2.resize(img_vis, (w, h))

    # 转浮点
    img_f = img_f.astype(np.float32)
    img_ir = img_ir.astype(np.float32)
    img_vis = img_vis.astype(np.float32)

    return {
        'EN': EN(img_f),
        'SD': SD(img_f),
        'SF': SF(img_f),
        'AG': AG(img_f),
        'Qabf': compute_Qabf(img_f, img_ir, img_vis)
    }


def draw_radar_chart(summary_data, metrics, save_dir):
    """绘制漂亮的六边形雷达图"""
    print("正在绘制雷达图...")
    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 配色方案 (给 SoPD-Net 红色以突出)
    colors = {
        'Pixel': '#d3d3d3', 'Pyramid': '#a9a9a9',  # 灰色给传统
        'DenseFuse': '#1f77b4', 'RFN-Nest': '#2ca02c', 'SeAFusion': '#ff7f0e',  # 彩色给竞品
        'SoPD-Net': '#d62728'  # 红色给自研
    }

    # 数据归一化处理 (Min-Max Scaling)
    # 结构: {metric: [val_algo1, val_algo2...]}
    metric_max = {}
    metric_min = {}
    for m in metrics:
        vals = []
        for algo in summary_data.keys():
            # 取该算法该指标的平均值
            vals.append(summary_data[algo][m])
        metric_max[m] = max(vals)
        metric_min[m] = min(vals)

    # 画图
    for algo in summary_data.keys():
        values = []
        for m in metrics:
            val = summary_data[algo][m]
            # 归一化到 0.2 ~ 1.0 (避免贴底)
            mx, mn = metric_max[m], metric_min[m]
            if mx - mn == 0:
                norm = 1.0
            else:
                norm = 0.2 + 0.8 * (val - mn) / (mx - mn)
            values.append(norm)

        values += values[:1]

        # 样式设置
        color = colors.get(algo, 'blue')
        lw = 3 if algo == 'SoPD-Net' else 1.5
        alpha = 0.2 if algo == 'SoPD-Net' else 0.05

        ax.plot(angles, values, linewidth=lw, label=algo, color=color)
        ax.fill(angles, values, alpha=alpha, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels([])  # 隐藏径向刻度
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Fusion Capability Comparison", size=15, y=1.05)
    plt.savefig(os.path.join(save_dir, "Radar_Comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 获取源文件列表
    files = [f for f in os.listdir(DIR_IR) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
    all_data = []

    # 统计数据容器
    summary = {algo: {m: 0.0 for m in ['EN', 'SD', 'SF', 'AG', 'Qabf']} for algo in ALGO_DIRS}
    counts = {algo: 0 for algo in ALGO_DIRS}

    print(f"🚀 开始评估 {len(files)} 组图片 x {len(ALGO_DIRS)} 种算法...")

    for fname in tqdm(files):
        p_ir = os.path.join(DIR_IR, fname)
        # 匹配可见光
        p_vis = os.path.join(DIR_VIS, fname)
        if not os.path.exists(p_vis):
            # 尝试不同后缀
            base = os.path.splitext(fname)[0]
            for ext in ['.jpg', '.png', '.bmp']:
                temp = os.path.join(DIR_VIS, base + ext)
                if os.path.exists(temp):
                    p_vis = temp
                    break

        if not os.path.exists(p_vis): continue

        img_ir = read_img(p_ir)
        img_vis = read_img(p_vis)

        row = {'Image': fname}

        for algo_name, algo_dir in ALGO_DIRS.items():
            p_res = os.path.join(algo_dir, fname)
            # 尝试匹配结果图
            if not os.path.exists(p_res):
                base = os.path.splitext(fname)[0]
                for ext in ['.jpg', '.png', '.bmp']:
                    temp = os.path.join(algo_dir, base + ext)
                    if os.path.exists(temp):
                        p_res = temp
                        break

            if os.path.exists(p_res):
                img_res = read_img(p_res)
                if img_res is not None:
                    metrics = evaluate_one(img_res, img_ir, img_vis)
                    for k, v in metrics.items():
                        row[f"{algo_name}_{k}"] = v
                        summary[algo_name][k] += v
                    counts[algo_name] += 1
            else:
                for k in ['EN', 'SD', 'SF', 'AG', 'Qabf']:
                    row[f"{algo_name}_{k}"] = np.nan

        all_data.append(row)

    # 保存详细CSV
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(OUTPUT_DIR, "detailed_metrics.csv"), index=False)

    # 计算平均值并打印
    print("\n====== 🏆 最终成绩单 ======")
    final_avg = {}
    metrics_list = ['EN', 'SD', 'SF', 'AG', 'Qabf']

    for algo in ALGO_DIRS:
        final_avg[algo] = {}
        if counts[algo] > 0:
            for m in metrics_list:
                final_avg[algo][m] = summary[algo][m] / counts[algo]
        else:
            for m in metrics_list: final_avg[algo][m] = 0

    # 打印表格
    print(f"{'Algorithm':<15} | {'EN':<8} | {'SD':<8} | {'SF':<8} | {'AG':<8} | {'Qabf':<8}")
    print("-" * 75)
    for algo, scores in final_avg.items():
        print(
            f"{algo:<15} | {scores['EN']:<8.4f} | {scores['SD']:<8.4f} | {scores['SF']:<8.4f} | {scores['AG']:<8.4f} | {scores['Qabf']:<8.4f}")

    # 绘制雷达图
    draw_radar_chart(final_avg, metrics_list, OUTPUT_DIR)

    print(f"\n✅ 全部完成！请查看文件夹: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()