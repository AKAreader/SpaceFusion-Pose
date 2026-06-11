# compare_ab_csv.py
import os
import pandas as pd
import numpy as np

A_CSV = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports\ir_out320_ex1.30_best_ir\pred_test_full.csv"
B_CSV = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports\rgbir_out320_ex1.30_best_rgbir\pred_test_full.csv"
OUT_DIR = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\ab_compare"

TOPK = 10

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    a = pd.read_csv(A_CSV)
    b = pd.read_csv(B_CSV)

    # 只用 id 和 angle_deg
    a = a[["id", "angle_deg"]].rename(columns={"angle_deg": "A_deg"})
    b = b[["id", "angle_deg"]].rename(columns={"angle_deg": "B_deg"})

    m = a.merge(b, on="id", how="inner")
    m["delta_B_minus_A"] = m["B_deg"] - m["A_deg"]

    # overall
    def stats(x):
        return float(np.mean(x)), float(np.median(x)), float(np.percentile(x, 90)), float(np.max(x))

    A_mean, A_med, A_p90, A_max = stats(m["A_deg"].values)
    B_mean, B_med, B_p90, B_max = stats(m["B_deg"].values)

    improved = (m["delta_B_minus_A"] < 0).sum()
    worsened = (m["delta_B_minus_A"] > 0).sum()
    equal = (m["delta_B_minus_A"] == 0).sum()

    # Top improved: A much better (B-A largest positive means B worse) / 你这里按你习惯两种都给
    top_B_worse = m.sort_values("delta_B_minus_A", ascending=False).head(TOPK)   # B更差
    top_B_better = m.sort_values("delta_B_minus_A", ascending=True).head(TOPK)  # B更好

    # save
    out_csv = os.path.join(OUT_DIR, "ab_merged.csv")
    m.sort_values("delta_B_minus_A", ascending=False).to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_txt = os.path.join(OUT_DIR, "ab_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Overall ===\n")
        f.write(f"A mean/median/p90 = {A_mean:.3f}/{A_med:.3f}/{A_p90:.3f}, max={A_max:.3f}\n")
        f.write(f"B mean/median/p90 = {B_mean:.3f}/{B_med:.3f}/{B_p90:.3f}, max={B_max:.3f}\n\n")

        f.write("=== Count ===\n")
        f.write(f"improved (B<A): {improved}\n")
        f.write(f"worsened (B>A): {worsened}\n")
        f.write(f"equal: {equal}\n\n")

        f.write(f"=== Top {TOPK} B much better (delta most negative) ===\n")
        for _, r in top_B_better.iterrows():
            f.write(f"{r['id']} | A={r['A_deg']:.3f}  B={r['B_deg']:.3f}  (B-A)={r['delta_B_minus_A']:.3f}\n")

        f.write(f"\n=== Top {TOPK} B much worse (delta most positive) ===\n")
        for _, r in top_B_worse.iterrows():
            f.write(f"{r['id']} | A={r['A_deg']:.3f}  B={r['B_deg']:.3f}  (B-A)={r['delta_B_minus_A']:.3f}\n")

    print("[OK] saved:", out_csv)
    print("[OK] saved:", out_txt)

if __name__ == "__main__":
    main()
