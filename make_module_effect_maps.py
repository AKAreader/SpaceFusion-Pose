import os
import cv2
import numpy as np

A_DIR = r"...\routeA_v2\...\fused_val"
B_DIR = r"...\routeB_gate\...\fused_val"
C_DIR = r"...\routeC_v2_ssim\...\fused_val"
OUT_DIR = r".\effect_report"

def safe_imread(path, flag):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flag)

def to_gray(img):
    if img.ndim == 2: return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def norm_0_255(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255.0).clip(0,255).astype(np.uint8)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(B_DIR) if f.lower().endswith(".png")]
    files.sort()

    for f in files:
        pa, pb, pc = os.path.join(A_DIR, f), os.path.join(B_DIR, f), os.path.join(C_DIR, f)
        if not (os.path.exists(pa) and os.path.exists(pb) and os.path.exists(pc)):
            continue

        A = safe_imread(pa, cv2.IMREAD_UNCHANGED)
        B = safe_imread(pb, cv2.IMREAD_UNCHANGED)
        C = safe_imread(pc, cv2.IMREAD_UNCHANGED)
        if A is None or B is None or C is None:
            continue

        Ag, Bg, Cg = to_gray(A), to_gray(B), to_gray(C)

        # Gate effect proxy
        gate_eff = cv2.absdiff(Bg, Ag)
        gate_eff = cv2.applyColorMap(norm_0_255(gate_eff), cv2.COLORMAP_JET)

        # Rectify/CA effect proxy
        rect_eff = cv2.absdiff(Cg, Bg)
        rect_eff = cv2.applyColorMap(norm_0_255(rect_eff), cv2.COLORMAP_JET)

        out1 = os.path.join(OUT_DIR, f.replace(".png", "_AtoB_gateEffect.png"))
        out2 = os.path.join(OUT_DIR, f.replace(".png", "_BtoC_rectifyEffect.png"))
        cv2.imencode(".png", gate_eff)[1].tofile(out1)
        cv2.imencode(".png", rect_eff)[1].tofile(out2)

    print("[OK] done:", OUT_DIR)

if __name__ == "__main__":
    main()
