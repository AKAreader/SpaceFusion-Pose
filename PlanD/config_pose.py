# config_pose.py
import os

# 1) 数据根目录（不要以反斜杠结尾）
DATA_ROOT = r"D:\BaiduNetdiskDownload\All"

# 2) 三个 json（如果你已生成，保持这样就行）
TRAIN_JSON = os.path.join(DATA_ROOT, "train_pose.json")
VAL_JSON   = os.path.join(DATA_ROOT, "val_pose.json")
TEST_JSON  = os.path.join(DATA_ROOT, "test_pose.json")

# 3) 模式： "ir" / "rgbir" / "dual"
MODE = "dual"

# 4) 输入尺寸
OUT_SIZE = 384

# 5) ROI 设置（稳定优先）
USE_ROI_CACHE = True            # 强烈建议 True
ROI_EXPAND_FIXED = 1.35         # 保守一点，先保证不截断
ROI_JITTER_TRAIN = 0.02         # 先小；如果稳定了再加到 0.05
ROI_EXPAND_MODE = "fixed"       # 先 fixed，稳定后再 random
ROI_EXPAND_MIN = 1.25
ROI_EXPAND_MAX = 1.70

# 6) 训练参数
EPOCHS = 120
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 5e-2
GEO_WEIGHT = 0.0                # 先置 0，确保你能稳定复现

# 7) dataloader
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# 8) 保存目录（相对 PlanD）
RUNS_DIR = "runs_pose"

# 9) Debug：过拟合小样本（你之前 64 很高就用这个验证管线）
OVERFIT_DEBUG = False
OVERFIT_N = 64

# 10) 可选：soft mask 背景抑制（不裁剪整图的替代思路）
USE_SOFT_MASK = False
SOFT_MASK_SOURCE = "saliency"   # "saliency" 或 "ir"
