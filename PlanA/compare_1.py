# compare_errors_csv.py
import os
import csv
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ======= 改这里：两份 CSV 路径 =======
# A = 你的最好方法（scaleaug）
CSV_A = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\analysis_ir_test_BEST\errors_test.csv"
# B = 你的基线（fixed）
CSV_B = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\analysis_ir_test_fixed\errors_test.csv"

OUT_DIR = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\ablation_compare1"
TOPK = 10


def read_csv(path: str):
    assert os.path.exists(path), f"CSV not found: {path}"
    data = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["id"].strip()
            ang = float(row["angle_deg"])
            data[sid] = ang
    return data


def stats(arr: np.ndarray):
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    a = read_csv(CSV_A)
    b = read_csv(CSV_B)

    common = sorted(set(a.keys()) & set(b.keys()))
    missing_a = sorted(set(b.keys()) - set(a.keys()))
    missing_b = sorted(set(a.keys()) - set(b.keys()))

    if missing_a:
        print("[WARN] Missing in A (present in B):", missing_a[:20], "..." if len(missing_a) > 20 else "")
    if missing_b:
        print("[WARN] Missing in B (present in A):", missing_b[:20], "..." if len(missing_b) > 20 else "")

    a_arr = np.array([a[sid] for sid in common], dtype=np.float32)
    b_arr = np.array([b[sid] for sid in common], dtype=np.float32)
    delta = b_arr - a_arr  # 正数表示 A 更好（误差下降）

    sa = stats(a_arr)
    sb = stats(b_arr)
    sd = stats(delta)

    print("\n===== SUMMARY (Common samples) =====")
    print(f"A: mean/median/p90 = {sa['mean']:.3f}/{sa['median']:.3f}/{sa['p90']:.3f} | max={sa['max']:.3f}")
    print(f"B: mean/median/p90 = {sb['mean']:.3f}/{sb['median']:.3f}/{sb['p90']:.3f} | max={sb['max']:.3f}")
    print(f"Delta(B-A): mean/median/p90 = {sd['mean']:.3f}/{sd['median']:.3f}/{sd['p90']:.3f} | max={sd['max']:.3f}\n")

    # Top improved (delta 最大)
    idx_sorted = np.argsort(-delta)
    top_imp = idx_sorted[:TOPK]
    top_worse = idx_sorted[-TOPK:][::-1]  # delta 最小（负数）表示 A 反而更差

    report_path = os.path.join(OUT_DIR, "compare_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Overall ===\n")
        f.write(f"A mean/median/p90 = {sa['mean']:.3f}/{sa['median']:.3f}/{sa['p90']:.3f}, max={sa['max']:.3f}\n")
        f.write(f"B mean/median/p90 = {sb['mean']:.3f}/{sb['median']:.3f}/{sb['p90']:.3f}, max={sb['max']:.3f}\n\n")

        f.write(f"=== Top {TOPK} improved (B - A largest, A much better) ===\n")
        for i in top_imp:
            sid = common[int(i)]
            f.write(f"{sid} | B={b[sid]:.3f}  A={a[sid]:.3f}  (B-A)={b[sid]-a[sid]:.3f}\n")

        f.write(f"\n=== Top {TOPK} worsened (B - A smallest/negative, A worse) ===\n")
        for i in top_worse:
            sid = common[int(i)]
            f.write(f"{sid} | B={b[sid]:.3f}  A={a[sid]:.3f}  (B-A)={b[sid]-a[sid]:.3f}\n")

    print("[OK] Saved:", report_path)

    # Optional plots
    if HAS_PLT:
        # histogram
        plt.figure()
        plt.hist(a_arr, bins=20)
        plt.title("Error distribution - A (deg)")
        plt.xlabel("deg"); plt.ylabel("count")
        plt.savefig(os.path.join(OUT_DIR, "hist_A.png"), dpi=200)

        plt.figure()
        plt.hist(b_arr, bins=20)
        plt.title("Error distribution - B (deg)")
        plt.xlabel("deg"); plt.ylabel("count")
        plt.savefig(os.path.join(OUT_DIR, "hist_B.png"), dpi=200)

        plt.figure()
        plt.hist(delta, bins=20)
        plt.title("Delta distribution (B - A, deg)")
        plt.xlabel("deg"); plt.ylabel("count")
        plt.savefig(os.path.join(OUT_DIR, "hist_delta_B_minus_A.png"), dpi=200)

        print("[OK] Plots saved to:", OUT_DIR)
    else:
        print("[INFO] matplotlib not available; skipped plots.")


if __name__ == "__main__":
    main()
