#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
viz_report.py
读取 compare_ABC.py 的输出目录 RUN_ROOT，生成直观统计图 + HTML 报告。

用法：
1) 先跑 compare_ABC.py，得到 RUN_ROOT（例如 runs_compare/ABC_compare/20260106-xxxxxx）
2) 把 RUN_ROOT 粘贴到下面常量
3) 运行：python viz_report.py
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# =========================
# 只需要改这里：RUN_ROOT
# =========================
RUN_ROOT = r"D:\Redundancy\edgedownload\SeAFusion-main\runs_compare\ABC_compare\20260107-184421"  # <<< 粘贴 compare_ABC.py 打印的 RUN_ROOT，例如 r".\runs_compare\ABC_compare\20260106-123456"

# 你希望展示的融合质量指标（都来自 metrics_*.csv）
FUSION_METRICS = ["entropy", "std", "ag", "sf", "mi_sum", "ssim_avg"]
POSE_METRIC = "pose_angle_deg"

# 指标方向：True=越大越好；False=越小越好
METRIC_HIGHER_BETTER = {
    "entropy": True,
    "std": True,      # 一般越大对比越强，但也可能放大噪声；用分布图判断更可靠
    "ag": True,
    "sf": True,
    "mi_sum": True,
    "ssim_avg": True, # 偏“保真”，越大越接近源图；不等于越锐越好
    "pose_angle_deg": False,
}

TOPK_MONTAGE = 8  # 拼图样例数
MONTAGE_H = 240   # 拼图单图缩放高度（越大越清晰）

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _read_csvs(run_root):
    sum_csv = os.path.join(run_root, "summary_ABC.csv")
    if not os.path.isfile(sum_csv):
        raise RuntimeError(f"找不到 summary_ABC.csv：{sum_csv}")

    summary = pd.read_csv(sum_csv, encoding="utf-8-sig")
    metric_files = glob.glob(os.path.join(run_root, "metrics_*.csv"))
    if len(metric_files) == 0:
        raise RuntimeError(f"找不到 metrics_*.csv：{run_root}")

    per = {}
    for fp in metric_files:
        df = pd.read_csv(fp, encoding="utf-8-sig")
        # method 名字从文件名拿：metrics_<method>.csv
        base = os.path.basename(fp)
        method = base[len("metrics_"):-len(".csv")]
        per[method] = df

    return summary, per

def _plot_bar_with_err(df_stats, out_path, title):
    # df_stats: index=method, columns=[mean,std] for multiple metrics
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    metrics = df_stats.columns.get_level_values(0).unique().tolist()
    methods = df_stats.index.tolist()

    # 组内柱状：每个 metric 一组，组内按 method 画柱
    n_m = len(metrics)
    n_k = len(methods)
    x = np.arange(n_m)
    width = 0.8 / max(n_k, 1)

    for i, m in enumerate(methods):
        means = [df_stats.loc[m, (met, "mean")] for met in metrics]
        stds  = [df_stats.loc[m, (met, "std")]  for met in metrics]
        ax.bar(x + (i - (n_k - 1)/2)*width, means, width=width, yerr=stds, capsize=3, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def _plot_box(per_dict, metric, out_path, title):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    methods = list(per_dict.keys())
    data = []
    for m in methods:
        s = pd.to_numeric(per_dict[m][metric], errors="coerce").dropna()
        data.append(s.values)

    ax.boxplot(data, labels=methods, showfliers=True)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def _plot_cdf(per_dict, metric, out_path, title):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    for m, df in per_dict.items():
        s = pd.to_numeric(df[metric], errors="coerce").dropna().values
        if len(s) == 0:
            continue
        s = np.sort(s)
        y = np.arange(1, len(s)+1) / len(s)
        ax.plot(s, y, label=m)

    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("CDF")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def _wins_by_metric(per_dict, metrics):
    """
    返回：DataFrame rows=metric, cols=method, value=win_count
    win定义：对每张图，按指标方向选最优方法；统计谁赢的次数最多
    """
    methods = list(per_dict.keys())
    # 用 name 对齐
    base = None
    for m in methods:
        df = per_dict[m][["name"] + metrics].copy()
        df = df.rename(columns={k: f"{k}__{m}" for k in metrics})
        base = df if base is None else base.merge(df, on="name", how="inner")
    if base is None or len(base) == 0:
        raise RuntimeError("无法按 name 对齐各方法 metrics（请确认三份 metrics_*.csv 的 name 一致）。")

    win = pd.DataFrame(0, index=metrics, columns=methods, dtype=int)

    for met in metrics:
        cols = [f"{met}__{m}" for m in methods]
        mat = base[cols].apply(pd.to_numeric, errors="coerce")
        # 某些行可能有空，丢掉空行
        mat = mat.dropna(axis=0, how="any")
        if len(mat) == 0:
            continue

        if METRIC_HIGHER_BETTER.get(met, True):
            best_idx = mat.values.argmax(axis=1)
        else:
            best_idx = mat.values.argmin(axis=1)

        for bi in best_idx:
            win.iloc[win.index.get_loc(met), bi] += 1

    return win

def _plot_wins(win_df, out_path, title):
    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()

    metrics = win_df.index.tolist()
    methods = win_df.columns.tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)

    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods)-1)/2)*width, win_df[m].values, width=width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Win count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def _infer_fused_dirs(run_root):
    fused_root = os.path.join(run_root, "fused")
    if not os.path.isdir(fused_root):
        return {}
    out = {}
    for d in os.listdir(fused_root):
        p = os.path.join(fused_root, d)
        if os.path.isdir(p):
            out[d] = p
    return out

def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def _resize_keep_h(img, H):
    h, w = img.shape[:2]
    if h == H:
        return img
    W = int(round(w * (H / float(h))))
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

def _put_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
    return out

def _make_montage(run_root, per_dict, out_path, topk=8, H=240):
    """
    自动挑选“最有代表性”的样本做拼图：
    - 若有 pose_angle：选 C 相对 A 改善最大的样本（更像你论文要强调的）
    - 再补充若干“融合质量差异大”的样本（按 mi_sum 或 ag）
    拼图列：RGB | IR | A | B | C
    """
    # 先拿到 RGB/IR 目录（从 compare_ABC.py 的日志无法直接取，这里按你固定配置推断）
    # 更稳妥做法：从 summary 里不含路径；所以这里用“RUN_ROOT 同级脚本固定数据路径”不现实。
    # 因此拼图只用 fused 输出 + 你原始数据目录需要你手动填两行（见下方）。
    RGB_DIR = None
    IR_DIR = None
    # 你可以在这里填原始目录，保证拼图含 RGB/IR（建议填）
    # RGB_DIR = r"E:\ALL\RGB"
    # IR_DIR  = r"E:\ALL\IR"

    fused_dirs = _infer_fused_dirs(run_root)
    methods = list(per_dict.keys())

    # 取 pose_angle 的表（如果存在）
    pose_available = all(POSE_METRIC in per_dict[m].columns for m in methods)

    # 对齐 A/B/C
    # 这里假设 methods 至少三个（A/B/C）
    # 用第一份为基准 merge
    base = None
    for m in methods:
        cols = ["name"] + FUSION_METRICS + ([POSE_METRIC] if POSE_METRIC in per_dict[m].columns else [])
        df = per_dict[m][cols].copy()
        df = df.rename(columns={k: f"{k}__{m}" for k in cols if k != "name"})
        base = df if base is None else base.merge(df, on="name", how="inner")
    if base is None or len(base) == 0:
        return

    # 尝试识别 A/C（用名称里包含 A_ / C_）
    mA = next((m for m in methods if m.startswith("A_")), methods[0])
    mC = next((m for m in methods if m.startswith("C_")), methods[-1])

    picks = []

    if pose_available:
        angA = pd.to_numeric(base.get(f"{POSE_METRIC}__{mA}"), errors="coerce")
        angC = pd.to_numeric(base.get(f"{POSE_METRIC}__{mC}"), errors="coerce")
        improv = (angA - angC)  # 正值= C 更好
        tmp = base[["name"]].copy()
        tmp["improv"] = improv
        tmp = tmp.dropna().sort_values("improv", ascending=False)
        picks += tmp["name"].head(max(1, topk//2)).tolist()

    # 再补一半：按 mi_sum 差异最大的样本
    met = "mi_sum"
    vA = pd.to_numeric(base.get(f"{met}__{mA}"), errors="coerce")
    vC = pd.to_numeric(base.get(f"{met}__{mC}"), errors="coerce")
    diff = (vC - vA).abs()
    tmp = base[["name"]].copy()
    tmp["diff"] = diff
    tmp = tmp.dropna().sort_values("diff", ascending=False)
    picks += [n for n in tmp["name"].tolist() if n not in picks][:max(1, topk - len(picks))]

    picks = picks[:topk]
    if len(picks) == 0:
        return

    rows = []
    for nm in picks:
        panels = []

        # RGB / IR（可选）
        if RGB_DIR is not None:
            p = os.path.join(RGB_DIR, nm)
            rgb = _imread_unicode(p, cv2.IMREAD_COLOR)
            if rgb is not None:
                rgb = _resize_keep_h(rgb, H)
                panels.append(_put_label(rgb, "RGB"))
        if IR_DIR is not None:
            p = os.path.join(IR_DIR, nm)
            ir = _imread_unicode(p, cv2.IMREAD_GRAYSCALE)
            if ir is not None:
                ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
                ir = _resize_keep_h(ir, H)
                panels.append(_put_label(ir, "IR"))

        # fused A/B/C（或更多方法）
        for m in methods:
            fd = fused_dirs.get(m)
            if fd is None:
                continue
            p = os.path.join(fd, nm)
            img = _imread_unicode(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = _resize_keep_h(img, H)
            panels.append(_put_label(img, m))

        if len(panels) == 0:
            continue
        row = cv2.hconcat(panels)
        rows.append(row)

    if len(rows) == 0:
        return

    canvas = cv2.vconcat(rows)
    _ensure_dir(os.path.dirname(out_path))
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return
    buf.tofile(out_path)

def _make_html_report(viz_dir, out_html, images, title="A/B/C Comparison Report"):
    lines = [
        "<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(title),
        "<h2>{}</h2>".format(title),
        "<p>说明：柱状图看均值；箱线图看稳定性与离群点；CDF 看整体分布（姿态角越靠左越好）。</p>",
    ]
    for caption, img in images:
        rel = os.path.relpath(img, os.path.dirname(out_html))
        lines += [f"<h3>{caption}</h3>", f"<img src='{rel}' style='max-width:1100px;width:100%;'/>", "<hr/>"]
    lines += ["</body></html>"]
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    if not RUN_ROOT or not os.path.isdir(RUN_ROOT):
        raise RuntimeError("请先在 RUN_ROOT 填入 compare_ABC.py 生成的输出目录（必须存在）。")

    viz_dir = os.path.join(RUN_ROOT, "viz")
    _ensure_dir(viz_dir)

    summary, per = _read_csvs(RUN_ROOT)
    methods = list(per.keys())

    # 1) 柱状图：融合指标均值+std（更直观）
    stats = {}
    for m, df in per.items():
        stats[m] = {}
        for met in FUSION_METRICS:
            s = pd.to_numeric(df[met], errors="coerce").dropna()
            stats[m][(met, "mean")] = float(s.mean()) if len(s) else float("nan")
            stats[m][(met, "std")]  = float(s.std())  if len(s) else float("nan")
    df_stats = pd.DataFrame(stats).T
    df_stats.columns = pd.MultiIndex.from_tuples(df_stats.columns)
    bar_path = os.path.join(viz_dir, "summary_bar.png")
    _plot_bar_with_err(df_stats, bar_path, "Fusion metrics (mean ± std)")

    # 2) 箱线图：每个指标一张（看稳定性）
    box_paths = []
    for met in FUSION_METRICS:
        p = os.path.join(viz_dir, f"boxplot_{met}.png")
        _plot_box(per, met, p, f"Boxplot: {met}")
        box_paths.append((f"Boxplot - {met}", p))

    # 3) 姿态角 CDF（如果有）
    cdf_pose_path = None
    pose_ok = all(POSE_METRIC in per[m].columns for m in methods)
    if pose_ok:
        cdf_pose_path = os.path.join(viz_dir, "cdf_pose_angle.png")
        _plot_cdf(per, POSE_METRIC, cdf_pose_path, "Pose angle error CDF (smaller is better)")

    # 4) 胜场统计（每张图谁最好）
    metrics_for_wins = FUSION_METRICS.copy()
    if pose_ok:
        metrics_for_wins.append(POSE_METRIC)
    win = _wins_by_metric(per, metrics_for_wins)
    win_csv = os.path.join(viz_dir, "wins_by_metric.csv")
    win.to_csv(win_csv, encoding="utf-8-sig")
    win_png = os.path.join(viz_dir, "wins_by_metric.png")
    _plot_wins(win, win_png, "Wins by metric (per-image best count)")

    # 5) 自动样例拼图（用于论文展示）
    montage_path = os.path.join(viz_dir, "montage_top_cases.png")
    _make_montage(RUN_ROOT, per, montage_path, topk=TOPK_MONTAGE, H=MONTAGE_H)

    # 6) HTML 报告
    images = [("Mean ± Std (Fusion metrics)", bar_path)]
    if cdf_pose_path:
        images.append(("Pose angle CDF", cdf_pose_path))
    images.append(("Wins by metric", win_png))
    images += box_paths
    images.append(("Montage of representative cases", montage_path))

    html_path = os.path.join(viz_dir, "report.html")
    _make_html_report(viz_dir, html_path, images, title="SeAFusion A/B/C Visual Report")

    print("Done.")
    print("Open:", html_path)

if __name__ == "__main__":
    main()
