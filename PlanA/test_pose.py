# test_pose.py
import os
import json
import csv
import math
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from pose_dataset import PoseDataset
from models import PoseResNet50
from metrics import quat_loss, quat_angle_deg


# ========= 默认配置（按你现在工程） =========
DEFAULT_DATA_ROOT = r"D:\BaiduNetdiskDownload\data\Aqua_60度"
DEFAULT_JSON_PATH = r"D:\BaiduNetdiskDownload\data\Aqua_60度\test_pose.json"

# 不想每次写参数：默认自动找 ckpt
DEFAULT_CKPT = "auto"  # 也可以改成具体 pt 路径
#DEFAULT_CKPT = r"D:\Redundancy\oneDrive\Desktop\空地探测技术在轨道目标探测中的方法研究\毕业了论文\重要文件\创新1\runs_pose\best_rgbir.pt"
# 建议默认 both：一次跑完 IR + RGBIR，直接对比
DEFAULT_MODE = "both"   # "ir" / "rgbir" / "both"

# 这些如果 ckpt 里有 meta，会优先用 meta；没有才用这里
DEFAULT_OUT_SIZE = 320
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 8
DEFAULT_ROI_EXPAND = 1.30

# 输出根目录（自动在下面再分子目录）
DEFAULT_SAVE_ROOT = r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\test_exports"
# ========================================


def _to_id_str(x):
    try:
        i = int(x)
        return f"{i:04d}"
    except Exception:
        s = str(x)
        return s.zfill(4) if s.isdigit() else s


def quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    n = float(np.linalg.norm(q) + 1e-12)
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def quat_wxyz_to_euler_zyx_deg(q: np.ndarray):
    """q: (4,) wxyz, normalized -> (yaw, pitch, roll) degrees (ZYX)"""
    w, x, y, z = q.tolist()

    # yaw (Z)
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t0, t1)

    # pitch (Y)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    # roll (X)
    t3 = 2.0 * (w * x + y * z)
    t4 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t3, t4)

    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def load_ckpt(path: str):
    # 显式 weights_only=False，避免未来默认改变导致行为不一致（且你 ckpt 里还有 args 等元信息）
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        meta = ckpt.get("args", {})
        return state, meta, ckpt.get("epoch", None), ckpt.get("best_med", None)
    return ckpt, {}, None, None


def _list_candidate_dirs():
    """你工程里常见的权重目录，按优先级搜索"""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [
        os.path.join(here, "runs_pose"),
        os.path.join(here, "runs_pose_fixed"),
        r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\runs_pose",
        r"D:\Redundancy\edgedownload\SeAFusion-main\PlanA\runs_pose_fixed",
    ]
    return [d for d in cand if os.path.isdir(d)]


def _find_best_ckpt(mode: str, hint: str = "auto") -> str:
    """
    mode: 'ir' or 'rgbir'
    hint:
      - 具体文件路径 -> 直接用
      - 目录路径 -> 在该目录里找 best_{mode}.pt
      - 'auto' -> 在候选目录里找 best_{mode}.pt；没有则找最新 pt
    """
    if hint and hint != "auto":
        # 1) 直接是文件
        if os.path.isfile(hint):
            return hint
        # 2) 是目录：优先找 best_{mode}.pt
        if os.path.isdir(hint):
            p = os.path.join(hint, f"best_{mode}.pt")
            if os.path.isfile(p):
                return p
            # 找目录里最新 pt
            pts = [os.path.join(hint, x) for x in os.listdir(hint) if x.lower().endswith(".pt")]
            if pts:
                pts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return pts[0]

    # auto：在常见目录里找 best_{mode}.pt
    for d in _list_candidate_dirs():
        p = os.path.join(d, f"best_{mode}.pt")
        if os.path.isfile(p):
            return p

    # 再兜底：找最新 pt
    pts_all = []
    for d in _list_candidate_dirs():
        pts_all += [os.path.join(d, x) for x in os.listdir(d) if x.lower().endswith(".pt")]
    if pts_all:
        pts_all.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return pts_all[0]

    raise FileNotFoundError(f"找不到 ckpt：mode={mode}, hint={hint}，请把 DEFAULT_CKPT 改成具体 pt 路径。")


@torch.no_grad()
def evaluate_and_export(model, loader, device, out_csv_path: str, meta_info: dict):
    model.eval()
    rows = []
    losses = []
    angles = []

    for x, q_gt, ids in loader:
        x = x.to(device, non_blocking=True)
        q_gt = q_gt.to(device, non_blocking=True)

        q_pred = model(x)
        loss = quat_loss(q_pred, q_gt)
        ang = quat_angle_deg(q_pred, q_gt)  # (B,)

        losses.append(loss.item())
        angles.append(ang.detach().cpu())

        q_pred_np = q_pred.detach().cpu().numpy()
        q_gt_np = q_gt.detach().cpu().numpy()
        ang_np = ang.detach().cpu().numpy()

        for i in range(len(ids)):
            sid = _to_id_str(ids[i])
            qg = quat_wxyz_normalize(q_gt_np[i])
            qp = quat_wxyz_normalize(q_pred_np[i])
            yg, pg, rg = quat_wxyz_to_euler_zyx_deg(qg)
            yp, pp, rp = quat_wxyz_to_euler_zyx_deg(qp)

            rows.append({
                "id": sid,
                "angle_deg": float(ang_np[i]),
                "q_gt_wxyz": f"[{qg[0]:.6f},{qg[1]:.6f},{qg[2]:.6f},{qg[3]:.6f}]",
                "q_pred_wxyz": f"[{qp[0]:.6f},{qp[1]:.6f},{qp[2]:.6f},{qp[3]:.6f}]",
                "euler_gt_zyx_deg": f"[{yg:.3f},{pg:.3f},{rg:.3f}]",
                "euler_pred_zyx_deg": f"[{yp:.3f},{pp:.3f},{rp:.3f}]",
            })

    angles = torch.cat(angles, dim=0).numpy()
    mean = float(np.mean(angles))
    med  = float(np.median(angles))
    p90  = float(np.percentile(angles, 90))
    loss_mean = float(np.mean(losses))

    rows.sort(key=lambda r: r["angle_deg"])

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys()) if rows else ["id", "angle_deg"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary_path = os.path.join(os.path.dirname(out_csv_path), "summary_test.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== META ===\n")
        for k, v in meta_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== METRICS ===\n")
        f.write(f"loss: {loss_mean:.5f}\n")
        f.write(f"ang(mean/median/p90): {mean:.3f}/{med:.3f}/{p90:.3f}\n")

    return loss_mean, mean, med, p90, out_csv_path, summary_path


def _make_out_dir(save_root: str, mode: str, out_size: int, roi_expand: float, ckpt_path: str):
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    tag = f"{mode}_out{out_size}_ex{roi_expand:.2f}_{ckpt_name}"
    return os.path.join(save_root, tag)


def run_one(mode: str, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = _find_best_ckpt(mode, args.ckpt)

    state, meta, epoch, best_med = load_ckpt(ckpt_path)

    # 优先用 ckpt 里的 meta，避免“参数/权重不匹配”
    mode_use = str(meta.get("mode", mode))
    out_size = int(meta.get("out_size", args.out_size))
    roi_expand = float(meta.get("roi_expand", args.roi_expand))

    # dataset
    ds = PoseDataset(
        data_root=args.data_root,
        json_path=args.json_path,
        mode=mode_use,
        out_size=out_size,
        train=False,
        roi_expand=roi_expand,
        roi_jitter=0.0,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # model
    in_ch = 1 if mode_use == "ir" else 4
    model = PoseResNet50(in_ch=in_ch, pretrained=False, dropout=0.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 通道自检：防止再出现 1/4 通道混乱
    x0, _, _ = next(iter(loader))
    if x0.shape[1] != in_ch:
        raise RuntimeError(f"[FATAL] Channel mismatch: dataset gives C={x0.shape[1]} but model expects C={in_ch} (mode={mode_use})")

    out_dir = _make_out_dir(args.save_root, mode_use, out_size, roi_expand, ckpt_path)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pred_test_full.csv")

    meta_info = {
        "mode": mode_use,
        "out_size": out_size,
        "roi_expand": roi_expand,
        "ckpt": ckpt_path,
        "epoch": epoch,
        "best_med(val)": best_med,
        "data_root": args.data_root,
        "json_path": args.json_path,
    }

    loss_mean, mean, med, p90, out_csv_path, summary_path = evaluate_and_export(
        model, loader, device, out_csv, meta_info
    )

    print(f"[TEST] mode={mode_use}  ckpt={ckpt_path}")
    print(f"[TEST] loss {loss_mean:.5f} | ang(mean/med/p90) {mean:.3f}/{med:.3f}/{p90:.3f}")
    print(f"[OK] CSV saved: {out_csv_path}")
    print(f"[OK] summary saved: {summary_path}")
    return (mode_use, mean, med, p90, out_csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)  # "auto" / 文件 / 目录
    parser.add_argument("--mode", type=str, choices=["ir", "rgbir", "both"], default=DEFAULT_MODE)

    parser.add_argument("--out_size", type=int, default=DEFAULT_OUT_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--roi_expand", type=float, default=DEFAULT_ROI_EXPAND)

    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    args = parser.parse_args()

    modes = ["ir", "rgbir"] if args.mode == "both" else [args.mode]

    results = []
    for m in modes:
        try:
            results.append(run_one(m, args))
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    if results:
        print("\n=== SUMMARY (for paper) ===")
        for (mode, mean, med, p90, csv_path) in results:
            print(f"{mode}: mean/median/p90 = {mean:.3f}/{med:.3f}/{p90:.3f} | csv={csv_path}")


if __name__ == "__main__":
    main()
