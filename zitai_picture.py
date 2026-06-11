# plot_pose_gui_v2.py
# -*- coding: utf-8 -*-
import os, re, time
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _safe_float(x: str):
    x = str(x).strip()
    if x.lower() in ("nan", "none", ""):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def pick_best_log(run_dir: str):
    """
    在目录里找最靠谱的日志：优先 val_angle 行出现次数最多的文件。
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None

    cands = []
    for root, _, files in os.walk(run_dir):
        for fn in files:
            low = fn.lower()
            if not (low.endswith(".log") or low.endswith(".txt") or low.endswith(".out")):
                continue
            p = os.path.join(root, fn)
            try:
                if os.path.getsize(p) < 1024:
                    continue
                cands.append(p)
            except Exception:
                pass
    if not cands:
        return None

    def count_hits(p: str):
        n_angle = 0
        n_epoch = 0
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "val_angle_mean/med/p95" in line and "[Epoch" in line:
                        n_angle += 1
                    if "[Epoch" in line:
                        n_epoch += 1
        except Exception:
            pass
        # 角度行权重最高，其次 epoch 行
        return (n_angle * 1000 + n_epoch)

    cands.sort(key=count_hits, reverse=True)
    return cands[0]

def parse_train_log(log_path: str):
    """
    更宽松解析：允许等号两侧有空格，允许不同分隔符。
    """
    if not log_path or not os.path.exists(log_path):
        raise RuntimeError(f"日志不存在: {log_path}")

    rows = []
    # 宽松：val_total= 或 val_total = ; val_angle_mean/med/p95= 或 val_angle_mean/med/p95 =
    pat = re.compile(
        r"\[Epoch\s+(?P<ep>\d+)\].*?"
        r"val_total\s*=\s*(?P<vt>[-+]?\d*\.?\d+|nan).*?"
        r"val_angle_mean/med/p95\s*=\s*"
        r"(?P<m>[-+]?\d*\.?\d+|nan)\s*/\s*(?P<md>[-+]?\d*\.?\d+|nan)\s*/\s*(?P<p>[-+]?\d*\.?\d+|nan)\s*deg",
        re.IGNORECASE
    )
    pat_alpha = re.compile(r"alpha\s*=\s*(?P<a>[-+]?\d*\.?\d+)", re.IGNORECASE)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "val_angle_mean/med/p95" not in line or "[Epoch" not in line:
                continue
            m = pat.search(line)
            if not m:
                continue
            ep = int(m.group("ep"))
            vt = _safe_float(m.group("vt"))
            mean = _safe_float(m.group("m"))
            med = _safe_float(m.group("md"))
            p95 = _safe_float(m.group("p"))
            a = np.nan
            ma = pat_alpha.search(line)
            if ma:
                a = _safe_float(ma.group("a"))
            rows.append({"epoch": ep, "alpha": a, "val_total": vt,
                         "angle_mean": mean, "angle_med": med, "angle_p95": p95})

    df = pd.DataFrame(rows).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    return df

def plot_curve(df_base, df_route, ycol: str, title: str, out_png: str,
               label_base="baseline", label_route="route"):
    plt.figure(figsize=(11, 5))

    def plot_one(df, lab):
        if df is None or len(df) == 0 or ycol not in df.columns:
            return 0
        x = df["epoch"].to_numpy()
        y = df[ycol].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() == 0:
            return 0
        plt.plot(x[mask], y[mask], marker="o", label=lab)
        return int(mask.sum())

    n1 = plot_one(df_base, label_base)
    n2 = plot_one(df_route, label_route)

    plt.xlabel("Epoch")
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    return n1, n2

def resolve_to_log(src: str):
    if os.path.isfile(src):
        return src
    if os.path.isdir(src):
        lp = pick_best_log(src)
        if not lp:
            raise RuntimeError(f"目录中没找到日志(.log/.txt/.out): {src}")
        return lp
    raise RuntimeError(f"无效路径: {src}")

def run_all(baseline_src: str, route_src: str, out_root: str, route_name="routeA_v2"):
    base_log = resolve_to_log(baseline_src)
    route_log = resolve_to_log(route_src)

    df_base = parse_train_log(base_log)
    df_route = parse_train_log(route_log)

    if len(df_base) == 0:
        raise RuntimeError(f"baseline 解析为空：请确认你选的是包含 val_angle 行的日志。\n{base_log}")
    if len(df_route) == 0:
        raise RuntimeError(f"route 解析为空：请确认你选的是包含 val_angle 行的日志。\n{route_log}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"pose_compare_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    df_base.to_csv(os.path.join(out_dir, "baseline_pose_curve.csv"), index=False, encoding="utf-8-sig")
    df_route.to_csv(os.path.join(out_dir, f"{route_name}_pose_curve.csv"), index=False, encoding="utf-8-sig")

    n_mean = plot_curve(df_base, df_route, "angle_mean",
                        "Pose Angle Error (Mean) on VAL",
                        os.path.join(out_dir, "pose_angle_mean.png"),
                        label_base="baseline", label_route=route_name)
    n_med = plot_curve(df_base, df_route, "angle_med",
                       "Pose Angle Error (Median) on VAL",
                       os.path.join(out_dir, "pose_angle_median.png"),
                       label_base="baseline", label_route=route_name)
    n_p95 = plot_curve(df_base, df_route, "angle_p95",
                       "Pose Angle Error (P95) on VAL",
                       os.path.join(out_dir, "pose_angle_p95.png"),
                       label_base="baseline", label_route=route_name)
    n_vt = plot_curve(df_base, df_route, "val_total",
                      "Fusion Total Loss (val_total) on VAL",
                      os.path.join(out_dir, "val_total_curve.png"),
                      label_base="baseline", label_route=route_name)

    summary = {
        "baseline_log": base_log,
        "route_log": route_log,
        "baseline_epochs_parsed": int(len(df_base)),
        f"{route_name}_epochs_parsed": int(len(df_route)),
        "baseline_valid_points(angle_mean)": n_mean[0],
        f"{route_name}_valid_points(angle_mean)": n_mean[1],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "pose_plot_debug_summary.csv"),
                                  index=False, encoding="utf-8-sig")
    return out_dir, summary

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pose Angle Plotter v2 (baseline vs route)")
        self.geometry("900x330")

        self.baseline_path = tk.StringVar(value="")
        self.route_path = tk.StringVar(value="")
        self.route_name = tk.StringVar(value="routeA_v2")
        self.out_dir = tk.StringVar(value=os.path.abspath("./pose_plots_out"))

        frm = tk.Frame(self)
        frm.pack(padx=12, pady=12, fill="both", expand=True)

        tk.Label(frm, text="baseline（run目录或日志文件）:").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.baseline_path, width=92).grid(row=0, column=1, sticky="w")
        tk.Button(frm, text="选目录", command=self.pick_base_dir).grid(row=0, column=2, padx=6)
        tk.Button(frm, text="选文件", command=self.pick_base_file).grid(row=0, column=3, padx=6)

        tk.Label(frm, text="route（run目录或日志文件）:").grid(row=1, column=0, sticky="w", pady=(10,0))
        tk.Entry(frm, textvariable=self.route_path, width=92).grid(row=1, column=1, sticky="w", pady=(10,0))
        tk.Button(frm, text="选目录", command=self.pick_route_dir).grid(row=1, column=2, padx=6, pady=(10,0))
        tk.Button(frm, text="选文件", command=self.pick_route_file).grid(row=1, column=3, padx=6, pady=(10,0))

        tk.Label(frm, text="route 名称（图例）:").grid(row=2, column=0, sticky="w", pady=(10,0))
        tk.Entry(frm, textvariable=self.route_name, width=20).grid(row=2, column=1, sticky="w", pady=(10,0))

        tk.Label(frm, text="输出目录:").grid(row=3, column=0, sticky="w", pady=(10,0))
        tk.Entry(frm, textvariable=self.out_dir, width=92).grid(row=3, column=1, sticky="w", pady=(10,0))
        tk.Button(frm, text="选目录", command=self.pick_out).grid(row=3, column=2, padx=6, pady=(10,0))

        tk.Button(frm, text="生成姿态角误差对比图（v2）", command=self.run).grid(row=4, column=1, pady=18)

        self.status = tk.StringVar(value="提示：优先选择具体 run 目录（带时间戳的那层），避免选错日志。")
        tk.Label(frm, textvariable=self.status, fg="blue").grid(row=5, column=0, columnspan=4, sticky="w")

    def pick_base_dir(self):
        p = filedialog.askdirectory(title="选择 baseline run 目录")
        if p: self.baseline_path.set(p)

    def pick_base_file(self):
        p = filedialog.askopenfilename(title="选择 baseline 日志文件",
                                       filetypes=[("Log/Txt", "*.log *.txt *.out"), ("All", "*.*")])
        if p: self.baseline_path.set(p)

    def pick_route_dir(self):
        p = filedialog.askdirectory(title="选择 route run 目录")
        if p: self.route_path.set(p)

    def pick_route_file(self):
        p = filedialog.askopenfilename(title="选择 route 日志文件",
                                       filetypes=[("Log/Txt", "*.log *.txt *.out"), ("All", "*.*")])
        if p: self.route_path.set(p)

    def pick_out(self):
        p = filedialog.askdirectory(title="选择输出目录")
        if p: self.out_dir.set(p)

    def run(self):
        bp = self.baseline_path.get().strip()
        rp = self.route_path.get().strip()
        rn = self.route_name.get().strip() or "route"
        od = self.out_dir.get().strip()

        if not bp or not (os.path.isdir(bp) or os.path.isfile(bp)):
            messagebox.showerror("错误", "baseline 路径无效：请选择 run 目录或日志文件")
            return
        if not rp or not (os.path.isdir(rp) or os.path.isfile(rp)):
            messagebox.showerror("错误", "route 路径无效：请选择 run 目录或日志文件")
            return
        if not od:
            messagebox.showerror("错误", "输出目录不能为空")
            return

        try:
            self.status.set("正在解析日志并生成图...")
            out_dir, summary = run_all(bp, rp, od, route_name=rn)
            msg = (
                f"完成。输出目录：\n{out_dir}\n\n"
                f"baseline_log:\n{summary['baseline_log']}\n\n"
                f"{rn}_log:\n{summary['route_log']}\n\n"
                f"解析到 epoch 数：baseline={summary['baseline_epochs_parsed']} | {rn}={summary[f'{rn}_epochs_parsed']}\n"
                f"有效 angle_mean 点：baseline={summary['baseline_valid_points(angle_mean)']} | {rn}={summary[f'{rn}_valid_points(angle_mean)']}\n\n"
                f"如果 baseline 有效点=0：说明你选到的 baseline 日志里 angle 全是 NaN 或选错 run。\n"
                f"请改为选择正确的 baseline 时间戳目录或直接选 baseline 的日志文件。"
            )
            messagebox.showinfo("完成", msg)
            self.status.set("完成：已生成 mean/median/p95 对比图，并输出 CSV + debug_summary。")
            try:
                os.startfile(out_dir)
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("运行失败", str(e))
            self.status.set("失败：请查看错误提示。")

if __name__ == "__main__":
    App().mainloop()
