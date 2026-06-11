# make_pose_ablation_report_v4.py
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ===================== 防止弹窗卡死 =====================
matplotlib.use('Agg')
# ========================================================

# ===================== 1. 配置区域 (名字已更新) =====================
RUN_LOGS = {
    # 1. SeAFusion (原 Baseline)


    # 2. Baseline (原 TrainA)
    "Baseline": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeA_v2\20260106-145605\train.log",

    # 3. TrainB
    "TrainB": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeB_gate\20260106-170140\train.log",

    # 4. TrainC (SoPD-Net)  <--- 名字已修改
    "TrainC (SoPD-Net)": r"D:\Redundancy\edgedownload\SeAFusion-main\runs_fusion\routeC_v2_ssim\20260108-183933\train.log",
}

OUT_DIR = r"./pose_ablation_report"


# ===================== 2. 颜色映射逻辑 =====================
def get_color_for_method(method_name):
    """
    根据方法名返回 SCI 颜色
    """
    name = method_name.lower()

    # 只要名字里包含 "sopd" 或 "trainc" 就给红色
    if "sopd" in name or "trainc" in name:
        return '#DC0000'  # 【TrainC/SoPD-Net -> 红色】 (SCI Red)
    elif "baseline" in name:
        return '#3C5488'  # 【Baseline -> 深蓝】 (SCI Dark Blue)
    elif "trainb" in name:
        return '#00A087'  # 【TrainB -> 蓝绿】 (SCI Teal)
    elif "seafusion" in name:
        return '#84919E'  # 【SeAFusion -> 灰色】 (SCI Grey)
    else:
        return '#E64B35'  # 默认颜色


# ==========================================================

RE_LOOSE = re.compile(r"val_angle_mean/med/p95\s*=\s*([^/\s]+)\s*/\s*([^/\s]+)\s*/\s*([^/\s\n]+)", re.IGNORECASE)
RE_EPOCH_TAG = re.compile(r"\[Epoch\s*(\d+)\]", re.IGNORECASE)


def read_text_any_encoding(path: str) -> str:
    if not os.path.exists(path): return ""
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936", "latin1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except:
            pass
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""


def to_float(x: str) -> float:
    try:
        clean_x = re.sub(r"[^0-9eE\.\-\+]", "", x.strip())
        return float(clean_x)
    except:
        return float("nan")


def parse_log(text: str):
    epoch_rows = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m_val = RE_LOOSE.search(line)
        if m_val:
            ep = -1
            m_ep = RE_EPOCH_TAG.search(line)
            if not m_ep and i > 0: m_ep = RE_EPOCH_TAG.search(lines[i - 1])
            if m_ep: ep = int(m_ep.group(1))

            mean = to_float(m_val.group(1))
            med = to_float(m_val.group(2))
            p95 = to_float(m_val.group(3))

            if not np.isnan(mean):
                epoch_rows.append({
                    "epoch": ep if ep != -1 else len(epoch_rows) + 1,
                    "val_angle_mean_deg": mean,
                    "val_angle_med_deg": med,
                    "val_angle_p95_deg": p95
                })
    return pd.DataFrame(epoch_rows)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def print_markdown_table(df_best):
    """
    直接在控制台打印 Markdown 表格，方便复制到论文
    """
    print("\n" + "=" * 20 + " Markdown Table " + "=" * 20)
    print("| 方法 (Method) | Mean Error (平均误差) ↓ | Median Error (中位误差) ↓ | P95 Error (95%分位误差) ↓ |")
    print("| :--- | :---: | :---: | :---: |")

    for _, row in df_best.iterrows():
        name = row['Method']
        mean = row['val_angle_mean_deg']
        med = row['val_angle_med_deg']
        p95 = row['val_angle_p95_deg']

        # 加粗 SoPD-Net 这一行
        if "SoPD-Net" in name or "TrainC" in name:
            name = f"**{name}**"
            mean_str = f"**{mean:.2f}**"
            med_str = f"**{med:.2f}**"
            p95_str = f"**{p95:.2f}**"
        else:
            mean_str = f"{mean:.2f}"
            med_str = f"{med:.2f}"
            p95_str = f"{p95:.2f}"

        print(f"| {name} | {mean_str} | {med_str} | {p95_str} |")
    print("=" * 56 + "\n")


def plot_bar_and_curve(df_best, df_curve, out_dir):
    methods = df_best["Method"].tolist()
    colors = [get_color_for_method(m) for m in methods]
    x = np.arange(len(methods))

    # 绘制 Mean, Median, P95 三张柱状图
    metrics = [
        (
        "val_angle_mean_deg", "Mean Angle Error (deg)", "Mean Angle Error (Lower is Better)", "pose_best_mean_bar.png"),
        ("val_angle_med_deg", "Median Angle Error (deg)", "Median Angle Error (Lower is Better)",
         "pose_best_med_bar.png"),
        ("val_angle_p95_deg", "P95 Angle Error (deg)", "P95 Angle Error (Lower is Better)", "pose_best_p95_bar.png")
    ]

    for col, ylabel, title, fname in metrics:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(x, df_best[col], color=colors, edgecolor='black', alpha=0.85, width=0.6)

        # 旋转=0, 字体稍微改小一点防止重叠
        plt.xticks(x, methods, rotation=0, ha="center", fontsize=10, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.2f}', ha='center', va='bottom', fontsize=11,
                     fontweight='bold')

        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # 绘制曲线图
    if not df_curve.empty:
        plt.figure(figsize=(10, 6))
        unique_methods = list(RUN_LOGS.keys())
        for m in unique_methods:
            g = df_curve[df_curve["Method"] == m].sort_values("epoch")
            if not g.empty:
                c = get_color_for_method(m)
                lw = 2.5 if "SoPD-Net" in m else 1.5
                z = 10 if "SoPD-Net" in m else 1
                plt.plot(g["epoch"], g["val_angle_mean_deg"], marker="o", markersize=4, label=m, color=c, linewidth=lw,
                         zorder=z)

        plt.xlabel("Epoch")
        plt.ylabel("Mean Angle Error (deg)")
        plt.title("Validation Mean Angle vs Epoch")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.savefig(os.path.join(out_dir, "pose_curve_mean.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    ensure_dir(OUT_DIR)

    best_rows = []
    curve_rows = []

    print("Parsing logs...")
    for method_name, log_path in RUN_LOGS.items():
        if not os.path.exists(log_path):
            print(f"[WARN] File not found: {method_name}")
            continue

        text = read_text_any_encoding(log_path)
        df = parse_log(text)

        if df.empty:
            print(f"[ERROR] No data in {method_name}")
            continue

        df["Method"] = method_name
        curve_rows.append(df)

        # 选优：取 val_angle_mean_deg 最小的行
        min_idx = df["val_angle_mean_deg"].idxmin()
        best_row = df.loc[min_idx].to_dict()

        best_rows.append({
            "Method": method_name,
            "best_epoch": int(best_row["epoch"]),
            "val_angle_mean_deg": float(best_row["val_angle_mean_deg"]),
            "val_angle_med_deg": float(best_row["val_angle_med_deg"]),
            "val_angle_p95_deg": float(best_row["val_angle_p95_deg"]),
        })

    if not best_rows: return

    # 排序
    method_order = list(RUN_LOGS.keys())
    df_best = pd.DataFrame(best_rows)
    df_best["Method"] = pd.Categorical(df_best["Method"], categories=method_order, ordered=True)
    df_best = df_best.sort_values("Method")

    df_curve = pd.concat(curve_rows, ignore_index=True)

    # 1. 打印 Markdown 表格
    print_markdown_table(df_best)

    # 2. 保存 CSV 和 图片
    df_best.to_csv(os.path.join(OUT_DIR, "pose_best_summary.csv"), index=False, encoding="utf-8-sig")
    plot_bar_and_curve(df_best, df_curve, OUT_DIR)

    print(f"\n[Success] Charts saved in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()