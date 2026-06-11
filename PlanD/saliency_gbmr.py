import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.spatial.distance import cdist
import scipy.linalg


def get_saliency_gbmr(img_ir, n_segments=200, sigma=0.1, alpha=0.99):
    """
    输入:
        img_ir: 红外图像 (灰度图, numpy array), 建议尺寸 320x320
        n_segments: 超像素数量 (根据图像复杂度调整, 200-300通常足够)
        sigma: 控制权重的参数
        alpha: 排序算法的阻尼系数
    输出:
        saliency_map: 归一化到 [0, 255] 的显著性图
    """

    # 1. 预处理：转为浮点数并归一化
    img = img_as_float(img_ir)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)  # SLIC需要3通道形式输入，即使是灰度

    # [cite_start]2. 超像素分割 (Superpixel Segmentation) [cite: 27]
    # 将图像分割为 n_segments 个小块，减少计算量
    segments = slic(img, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
    num_superpixels = segments.max() + 1

    # 3. 构建图 (Graph Construction)
    # 计算每个超像素的平均颜色/强度
    features = []
    for i in range(num_superpixels):
        mask = (segments == i)
        mean_feature = np.mean(img[mask], axis=0)
        features.append(mean_feature)
    features = np.array(features)  # [N, 3]

    # 计算节点间的亲和矩阵 W (Affinity Matrix)
    # 距离越近，权重越大
    dist_matrix = cdist(features, features)
    # 使用径向基函数 (RBF) 计算相似度
    W = np.exp(-dist_matrix / (2.0 * sigma * sigma))
    np.fill_diagonal(W, 0)  # 对角线置0

    # [cite_start]4. 流形排序 (Manifold Ranking) [cite: 26]
    # 构建度矩阵 D
    D = np.diag(np.sum(W, axis=1))

    # 计算排序矩阵 (D - alpha*W)^-1
    # 这是一个 N x N 的矩阵求逆，由于 N 较小 (200-300)，速度很快
    opt_matrix = D - alpha * W
    try:
        inv_matrix = scipy.linalg.inv(opt_matrix)
    except:
        # 如果矩阵奇异，添加微小扰动
        inv_matrix = scipy.linalg.inv(opt_matrix + np.eye(num_superpixels) * 1e-5)

    # [cite_start]5. 定义背景查询种子 (Background Query) [cite: 26]
    # 假设图像的四个边界（上、下、左、右）属于背景
    # 找出所有位于边界的超像素
    boundary_mask = np.zeros(img.shape[:2], dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    # 创建指示向量 y (1表示是背景，0表示不是)
    y = np.zeros(num_superpixels)
    boundary_indices = np.unique(segments[boundary_mask])
    y[boundary_indices] = 1.0

    # 6. 计算显著性得分
    # f^* = (D - alpha*W)^-1 * y
    # 这里计算的是“属于背景的分数”
    background_scores = np.dot(inv_matrix, y)

    # 显著性 = 1 - 背景分数 (即越不像背景，越显著)
    saliency_scores = 1.0 - background_scores

    # 7. 重映射回像素级 (Project back to pixels)
    saliency_map = np.zeros(img.shape[:2])
    for i in range(num_superpixels):
        mask = (segments == i)
        saliency_map[mask] = saliency_scores[i]

    # 8. 归一化处理
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    saliency_map = (saliency_map * 255).astype(np.uint8)

    return saliency_map


# === 简单的测试/调试代码 ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 读取你的一张测试红外图 (确保路径正确)
    test_img_path = "path/to/your/ir_image.png"
    # 如果没有图，这里生成一个模拟图
    img_ir = np.zeros((320, 320), dtype=np.uint8)
    cv2.circle(img_ir, (160, 160), 50, 200, -1)  # 模拟中间有个亮的卫星

    # 运行算法
    sm = get_saliency_gbmr(img_ir)

    # 可视化对比
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input IR")
    plt.imshow(img_ir, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Generated Saliency Map")
    plt.imshow(sm, cmap='jet')  # 使用jet热力图查看高亮区域
    plt.show()