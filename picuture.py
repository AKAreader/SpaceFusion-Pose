# plot_metrics_summary.py
# -*- coding: utf-8 -*-
"""
一键生成 metrics_summary 对比图（baseline vs trainA）
- 自动定位最新 compare_runs/baseline_vs_trainA_* 文件夹
- 读取 metrics_summary.csv
- 输出：
  1) metrics_compare_grouped.png  (分组柱状图)
  2) metrics_compare_delta.png    (trainA - baseline 差值图)
  3) metrics_compare_table.csv    (汇总表，含差值)

如果你想固定某一次输出目录，把 COMPARE_DIR 改成那次路径即可。
"""

import os
import re
import csv
from datetime import datetime

import numpy as np

# 尽量用 pandas，若没有则自动 fallback
try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib.pyplot as plt


# =========================
# 0) 只改这里（可选）
# =========================
PROJECT_ROOT = r"D:\Redundancy\edgedownload\SeAFusion-main"
COMPARE_DIR = ""  # 留空=自动找最新；或填：r"D:\...\compare_runs\baseline_vs_trainA_20251228_165916"


# =========================
# 1) 工具：定位最新目录
# =========================
def find_latest_compare_dir(project_root: str) -> str:
    compare_root = os.path.join(project_root, "compare_runs")
    if not os.path.isdir(compare_root):
        raise FileNotFoundError(f"compare_runs not found: {compare_root}")

    patt = re.compile(r"baseline_vs_trainA_\d{8}_\d{6}")
    cands = []
    for name in os.listdir(compare_root):
        full = os.path.join(compare_root, name)
        if os.path.isdir(full) and patt.match(name):
            cands.append(full)
    if not cands:
        raise FileNotFoundError(f"No baseline_vs_trainA_* folders under: {compare_root}")

    # 按修改时间选最新
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


# =========================
# 2) 读取 metrics_summary.csv 并标准化为：
#    columns: ["metric", "baseline", "trainA"]
# =========================
def _to_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() in ["nan", "none"]:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def load_metrics_summary(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metrics_summary.csv not found: {csv_path}")

    if pd is not None:
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        # 情况A：长表（常见字段：metric/method/value 或 metric/model/mean）
        lower_cols = [c.lower() for c in df.columns]
        if ("metric" in lower_cols) and (("method" in lower_cols) or ("model" in lower_cols)) and ("value" in lower_cols or "mean" in lower_cols):
            metric_col = df.columns[lower_cols.index("metric")]
            method_col = df.columns[lower_cols.index("method")] if "method" in lower_cols else df.columns[lower_cols.index("model")]
            value_col = df.columns[lower_cols.index("value")] if "value" in lower_cols else df.columns[lower_cols.index("mean")]

            tmp = df[[metric_col, method_col, value_col]].copy()
            tmp[method_col] = tmp[method_col].astype(str)

            # 选 baseline/trainA
            def norm_method(s):
                s2 = str(s).strip().lower()
                if "baseline" in s2:
                    return "baseline"
                if "traina" in s2 or "train_a" in s2 or "traina" in s2:
                    return "trainA"
                return s2

            tmp["__method"] = tmp[method_col].map(norm_method)
            tmp = tmp[tmp["__method"].isin(["baseline", "trainA"])].copy()
            pivot = tmp.pivot_table(index=metric_col, columns="__method", values=value_col, aggfunc="mean").reset_index()
            pivot.columns = ["metric"] + [c for c in pivot.columns[1:]]
            for c in ["baseline", "trainA"]:
                if c not in pivot.columns:
                    pivot[c] = np.nan
            out = pivot[["metric", "baseline", "trainA"]].copy()
            return out

        # 情况B：宽表（每行一个 metric，列里有 baseline/trainA 的 mean 或直接值）
        # 寻找 baseline/trainA 列名
        def pick_col(keys):
            # 优先：baseline_mean / trainA_mean，其次 baseline / trainA
            for k in keys:
                for c in df.columns:
                    if c.lower() == k.lower():
                        return c
            return None

        metric_col = pick_col(["metric", "name", "指标", "指标名"])
        if metric_col is None:
            # 退一步：第一列当 metric
            metric_col = df.columns[0]

        base_col = pick_col(["baseline_mean", "baseline", "base_mean", "base"])
        trna_col = pick_col(["trainA_mean", "traina_mean", "trainA", "traina", "train_a_mean", "train_a"])

        # 若没找到，尝试“包含”匹配
        if base_col is None:
            for c in df.columns:
                if "baseline" in c.lower():
                    base_col = c
                    break
        if trna_col is None:
            for c in df.columns:
                if "traina" in c.lower() or "train_a" in c.lower():
                    trna_col = c
                    break

        if base_col is None or trna_col is None:
            # 最后 fallback：取除 metric 外的前两个数值列
            numeric_cols = []
            for c in df.columns:
                if c == metric_col:
                    continue
                try:
                    pd.to_numeric(df[c], errors="coerce")
                    numeric_cols.append(c)
                except Exception:
                    pass
            if len(numeric_cols) >= 2:
                base_col, trna_col = numeric_cols[0], numeric_cols[1]
            else:
                raise RuntimeError(f"Cannot identify baseline/trainA columns in: {csv_path}. cols={df.columns.tolist()}")

        out = df[[metric_col, base_col, trna_col]].copy()
        out.columns = ["metric", "baseline", "trainA"]
        out["baseline"] = pd.to_numeric(out["baseline"], errors="coerce")
        out["trainA"] = pd.to_numeric(out["trainA"], errors="coerce")
        return out

    # pandas 不存在：用 csv 模块粗读（要求格式为：metric + 两列数值）
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_l = [h.strip().lower() for h in header]

        # 简化：找 metric + baseline + trainA
        def find_idx(names):
            for nm in names:
                if nm in header_l:
                    return header_l.index(nm)
            return None

        i_metric = find_idx(["metric", "name"])
        i_base = find_idx(["baseline", "baseline_mean"])
        i_trna = find_idx(["traina", "traina_mean", "train_a", "train_a_mean", "traina"])

        if i_metric is None:
            i_metric = 0
        if i_base is None or i_trna is None:
            # fallback：取前两列数值
            i_base, i_trna = 1, 2

        for r in reader:
            if not r:
                continue
            rows.append([r[i_metric], _to_float(r[i_base]), _to_float(r[i_trna])])

    return rows  # list


# =========================
# 3) 画图
# =========================
def plot_grouped_bar(df, out_png: str, title: str):
    metrics = df["metric"].tolist()
    base = df["baseline"].to_numpy()
    trna = df["trainA"].to_numpy()

    x = np.arange(len(metrics))
    w = 0.38

    plt.figure(figsize=(max(10, 0.6 * len(metrics)), 6))
    plt.bar(x - w/2, base, width=w, label="baseline")
    plt.bar(x + w/2, trna, width=w, label="trainA")
    plt.xticks(x, metrics, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_delta_bar(df, out_png: str, title: str):
    metrics = df["metric"].tolist()
    delta = (df["trainA"] - df["baseline"]).to_numpy()

    x = np.arange(len(metrics))
    plt.figure(figsize=(max(10, 0.6 * len(metrics)), 5))
    plt.bar(x, delta)
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, metrics, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("trainA - baseline")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    if COMPARE_DIR.strip():
        run_dir = COMPARE_DIR
    else:
        run_dir = find_latest_compare_dir(PROJECT_ROOT)

    csv_path = os.path.join(run_dir, "metrics_summary.csv")
    df = load_metrics_summary(csv_path)

    # 若 df 是 list（无 pandas），转换为 DataFrame-like 结构
    if pd is None:
        # rows: [metric, baseline, trainA]
        rows = df
        # 简单写出一个“伪表”并用 numpy 作图
        metrics = [r[0] for r in rows]
        base = np.array([r[1] for r in rows], dtype=float)
        trna = np.array([r[2] for r in rows], dtype=float)

        # 输出表
        out_table = os.path.join(run_dir, "metrics_compare_table.csv")
        with open(out_table, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["metric", "baseline", "trainA", "delta(trainA-baseline)"])
            for m, b, t in zip(metrics, base, trna):
                wr.writerow([m, b, t, t - b])

        # 绘图
        out1 = os.path.join(run_dir, "metrics_compare_grouped.png")
        out2 = os.path.join(run_dir, "metrics_compare_delta.png")

        x = np.arange(len(metrics))
        w = 0.38
        plt.figure(figsize=(max(10, 0.6 * len(metrics)), 6))
        plt.bar(x - w/2, base, width=w, label="baseline")
        plt.bar(x + w/2, trna, width=w, label="trainA")
        plt.xticks(x, metrics, rotation=45, ha="right")
        plt.title("metrics_summary comparison (baseline vs trainA)")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out1, dpi=200)
        plt.close()

        plt.figure(figsize=(max(10, 0.6 * len(metrics)), 5))
        plt.bar(x, trna - base)
        plt.axhline(0.0, linewidth=1.0)
        plt.xticks(x, metrics, rotation=45, ha="right")
        plt.title("metrics delta (trainA - baseline)")
        plt.ylabel("delta")
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        plt.close()

        print("[OK] Saved:")
        print(out1)
        print(out2)
        print(out_table)
        print("Run dir:", run_dir)
        return

    # pandas 路径
    # 清理掉全 NaN 的行
    df = df.copy()
    df = df.dropna(subset=["baseline", "trainA"], how="all")
    df = df.fillna(np.nan)

    # 写对比表（含 delta）
    df["delta(trainA-baseline)"] = df["trainA"] - df["baseline"]
    out_table = os.path.join(run_dir, "metrics_compare_table.csv")
    df.to_csv(out_table, index=False, encoding="utf-8-sig")

    out1 = os.path.join(run_dir, "metrics_compare_grouped.png")
    out2 = os.path.join(run_dir, "metrics_compare_delta.png")

    plot_grouped_bar(df, out1, "metrics_summary comparison (baseline vs trainA)")
    plot_delta_bar(df, out2, "metrics delta (trainA - baseline)")

    print("[OK] Saved:")
    print(out1)
    print(out2)
    print(out_table)
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
