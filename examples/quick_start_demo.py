"""Generate a deterministic zero-data RGB/IR quick-start demonstration.

This script creates a synthetic CPU baseline preview. It is not SoPD-Net,
SeAFusion, or trained-model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DEFAULT_CONFIG = "configs/quick_start_demo.yaml"


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    first = current if current.is_dir() else current.parent
    for candidate in [first, *first.parents]:
        if (candidate / "README.md").exists() and (candidate / "tools" / "validate_config.py").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from the current path.")


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse YAML config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return data


def validate_size(value: int) -> int:
    if value < 64:
        raise ValueError("--size must be at least 64 pixels.")
    return value


def validate_weights(visible_weight: float, infrared_weight: float) -> None:
    if not 0.0 <= visible_weight <= 1.0:
        raise ValueError("visible_luminance_weight must be between 0 and 1.")
    if not 0.0 <= infrared_weight <= 1.0:
        raise ValueError("infrared_weight must be between 0 and 1.")
    if abs((visible_weight + infrared_weight) - 1.0) > 1e-6:
        raise ValueError("Fusion weights must sum to 1.0.")


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def parse_args(argv: Optional[Iterable[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic zero-data RGB/IR quick-start demo.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to the quick-start YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--size", type=int, default=None, help="Override square image size.")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic seed.")
    return parser.parse_args(argv)


def rect_from_center(size: int, cx: float, cy: float, w: float, h: float) -> Tuple[int, int, int, int]:
    return (
        int(round((cx - w / 2) * size)),
        int(round((cy - h / 2) * size)),
        int(round((cx + w / 2) * size)),
        int(round((cy + h / 2) * size)),
    )


def draw_spacecraft_geometry(visible_rgb: np.ndarray, infrared: np.ndarray, size: int) -> None:
    body = rect_from_center(size, 0.50, 0.50, 0.20, 0.24)
    left_panel = rect_from_center(size, 0.29, 0.50, 0.28, 0.12)
    right_panel = rect_from_center(size, 0.71, 0.50, 0.28, 0.12)
    antenna_start = (int(0.50 * size), int(0.38 * size))
    antenna_end = (int(0.50 * size), int(0.23 * size))

    cv2.rectangle(visible_rgb, left_panel[:2], left_panel[2:], (54, 93, 142), -1)
    cv2.rectangle(visible_rgb, right_panel[:2], right_panel[2:], (54, 93, 142), -1)
    cv2.rectangle(visible_rgb, body[:2], body[2:], (174, 178, 184), -1)
    cv2.rectangle(visible_rgb, body[:2], body[2:], (72, 82, 96), max(1, size // 96))
    cv2.line(visible_rgb, antenna_start, antenna_end, (222, 226, 230), max(1, size // 128))
    cv2.circle(visible_rgb, antenna_end, max(2, size // 64), (240, 224, 120), -1)

    panel_step = max(4, size // 16)
    for x in range(left_panel[0] + panel_step, left_panel[2], panel_step):
        cv2.line(visible_rgb, (x, left_panel[1]), (x, left_panel[3]), (82, 129, 184), 1)
    for x in range(right_panel[0] + panel_step, right_panel[2], panel_step):
        cv2.line(visible_rgb, (x, right_panel[1]), (x, right_panel[3]), (82, 129, 184), 1)
    cv2.line(visible_rgb, (left_panel[2], int(0.50 * size)), (body[0], int(0.50 * size)), (210, 214, 216), max(1, size // 100))
    cv2.line(visible_rgb, (body[2], int(0.50 * size)), (right_panel[0], int(0.50 * size)), (210, 214, 216), max(1, size // 100))

    cv2.rectangle(infrared, left_panel[:2], left_panel[2:], 118, -1)
    cv2.rectangle(infrared, right_panel[:2], right_panel[2:], 118, -1)
    cv2.rectangle(infrared, body[:2], body[2:], 205, -1)
    cv2.ellipse(infrared, (int(0.54 * size), int(0.47 * size)), (max(2, size // 24), max(2, size // 32)), 0, 0, 360, 242, -1)
    cv2.line(infrared, antenna_start, antenna_end, 176, max(1, size // 96))
    cv2.circle(infrared, antenna_end, max(2, size // 64), 225, -1)


def generate_synthetic_pair(size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = x / max(1, size - 1)
    yn = y / max(1, size - 1)

    visible = np.zeros((size, size, 3), dtype=np.float32)
    visible[..., 0] = 26 + 45 * xn + 12 * yn
    visible[..., 1] = 33 + 38 * yn
    visible[..., 2] = 52 + 55 * (1.0 - xn) + 8 * np.sin(yn * np.pi)
    visible += rng.normal(0.0, 1.2, visible.shape).astype(np.float32)

    infrared = 24 + 28 * (1.0 - yn) + 9 * np.sin((xn + yn) * np.pi)
    infrared += rng.normal(0.0, 0.9, infrared.shape).astype(np.float32)

    visible_u8 = np.clip(visible, 0, 255).astype(np.uint8)
    infrared_u8 = np.clip(infrared, 0, 255).astype(np.uint8)
    draw_spacecraft_geometry(visible_u8, infrared_u8, size)
    return visible_u8, infrared_u8


def make_baseline_preview(visible_rgb: np.ndarray, infrared: np.ndarray, visible_weight: float, infrared_weight: float) -> np.ndarray:
    ycrcb = cv2.cvtColor(visible_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[..., 0].astype(np.float32)
    ir_channel = infrared.astype(np.float32)
    fused_y = visible_weight * y_channel + infrared_weight * ir_channel
    ycrcb[..., 0] = np.clip(fused_y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def with_label(rgb: np.ndarray, label: str) -> np.ndarray:
    output = rgb.copy()
    font_scale = max(0.45, rgb.shape[1] / 512.0)
    thickness = max(1, rgb.shape[1] // 180)
    cv2.rectangle(output, (0, 0), (rgb.shape[1], max(22, rgb.shape[0] // 9)), (18, 24, 32), -1)
    cv2.putText(output, label, (8, max(16, rgb.shape[0] // 14)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 247, 250), thickness, cv2.LINE_AA)
    return output


def write_png(path: Path, image_rgb_or_gray: np.ndarray, is_rgb: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(image_rgb_or_gray, cv2.COLOR_RGB2BGR) if is_rgb else image_rgb_or_gray
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resolved_config(output_dir: Path, size: int, seed: int) -> Dict[str, Any]:
    return {
        "dataset": {
            "name": "synthetic_quick_start",
            "root": str(output_dir.resolve()),
            "visible_dir": str((output_dir / "visible").resolve()),
            "infrared_dir": str((output_dir / "infrared").resolve()),
            "annotation_file": str((output_dir / "pose_annotations.csv").resolve()),
            "manifest_file": str((output_dir / "manifest.json").resolve()),
        },
        "image": {"height": size, "width": size, "visible_channels": 3, "infrared_channels": 1},
        "pose": {"representation": "quaternion", "fields": {"w": "qw", "x": "qx", "y": "qy", "z": "qz"}},
        "split": {"train_ratio": 0.9, "val_ratio": 0.1, "random_seed": seed},
        "notes": [
            "Generated by the public synthetic quick-start demo.",
            "This configuration does not represent a full research dataset.",
        ],
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = parse_args(argv)
        repo_root = find_repo_root(Path(__file__))
        config_path = resolve_path(repo_root, args.config)
        config = load_yaml(config_path)
        demo = config.get("demo", {})
        baseline = config.get("baseline", {})
        sample = config.get("sample", {})
        pose = sample.get("pose", {})

        output_dir = resolve_path(repo_root, args.output_dir or demo.get("output_dir", "outputs/quick_start"))
        size = validate_size(int(args.size if args.size is not None else demo.get("image_size", 256)))
        seed = int(args.seed if args.seed is not None else demo.get("seed", 42))
        visible_weight = float(baseline.get("visible_luminance_weight", 0.60))
        infrared_weight = float(baseline.get("infrared_weight", 0.40))
        validate_weights(visible_weight, infrared_weight)

        pair_id = str(sample.get("pair_id", "0001"))
        object_class = str(sample.get("object_class", "SyntheticDemo"))
        phase_angle = int(sample.get("phase_angle_deg", 45))
        output_dir.mkdir(parents=True, exist_ok=True)

        visible_path = output_dir / "visible" / f"{pair_id}.png"
        infrared_path = output_dir / "infrared" / f"{pair_id}.png"
        preview_path = output_dir / "baseline_fused_preview.png"
        grid_path = output_dir / "preview_grid.png"
        annotation_path = output_dir / "pose_annotations.csv"
        manifest_path = output_dir / "manifest.json"
        resolved_config_path = output_dir / "resolved_dataset.yaml"
        summary_path = output_dir / "summary.json"

        print("[INFO] Generating deterministic synthetic RGB/IR sample.")
        print("[INFO] This is a baseline preview, not trained-model inference.")
        visible_rgb, infrared = generate_synthetic_pair(size=size, seed=seed)
        baseline_rgb = make_baseline_preview(visible_rgb, infrared, visible_weight, infrared_weight)

        write_png(visible_path, visible_rgb, is_rgb=True)
        print("[PASS] Generated visible input.")
        write_png(infrared_path, infrared, is_rgb=False)
        print("[PASS] Generated infrared input.")
        write_png(preview_path, baseline_rgb, is_rgb=True)
        print("[PASS] Generated baseline fused preview.")
        write_png(grid_path, np.hstack([with_label(visible_rgb, "Visible RGB"), with_label(cv2.cvtColor(infrared, cv2.COLOR_GRAY2RGB), "Infrared"), with_label(baseline_rgb, "Baseline Preview")]), is_rgb=True)

        with annotation_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair_id", "qw", "qx", "qy", "qz", "object_class", "phase_angle_deg"])
            writer.writeheader()
            writer.writerow({"pair_id": pair_id, "qw": float(pose.get("qw", 1.0)), "qx": float(pose.get("qx", 0.0)), "qy": float(pose.get("qy", 0.0)), "qz": float(pose.get("qz", 0.0)), "object_class": object_class, "phase_angle_deg": phase_angle})

        manifest = {"description": "Deterministic synthetic sample generated by the public quick-start demo.", "samples": [{"pair_id": pair_id, "visible": f"visible/{pair_id}.png", "infrared": f"infrared/{pair_id}.png", "pose": {"qw": float(pose.get("qw", 1.0)), "qx": float(pose.get("qx", 0.0)), "qy": float(pose.get("qy", 0.0)), "qz": float(pose.get("qz", 0.0))}, "metadata": {"object_class": object_class, "phase_angle_deg": phase_angle}}]}
        write_json(manifest_path, manifest)
        with resolved_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(build_resolved_config(output_dir, size, seed), handle, sort_keys=False, allow_unicode=True)
        print("[PASS] Generated manifest and resolved dataset config.")

        validator_path = repo_root / "tools" / "validate_config.py"
        subprocess.run([sys.executable, str(validator_path), "--config", str(resolved_config_path), "--check-paths"], cwd=str(repo_root), check=True)
        print("[PASS] Existing configuration validator completed successfully.")

        summary = {"status": "pass", "demo_type": "synthetic_cpu_baseline", "model_inference": False, "checkpoint_used": False, "gpu_required": False, "network_access_required": False, "seed": seed, "image_size": size, "outputs": {"visible": f"visible/{pair_id}.png", "infrared": f"infrared/{pair_id}.png", "baseline_preview": "baseline_fused_preview.png", "preview_grid": "preview_grid.png", "manifest": "manifest.json", "resolved_config": "resolved_dataset.yaml"}, "sha256": {"visible": sha256_file(visible_path), "infrared": sha256_file(infrared_path), "baseline_preview": sha256_file(preview_path), "preview_grid": sha256_file(grid_path)}}
        write_json(summary_path, summary)
        print("[PASS] Quick start demo completed.")
        print(f"[INFO] Result directory: {output_dir.resolve()}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
