from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "quick_start_demo.py"
CONFIG = REPO_ROOT / "configs" / "quick_start_demo.yaml"
VALIDATOR = REPO_ROOT / "tools" / "validate_config.py"


def run_demo(output_dir: Path, size: int = 128, seed: int = 42) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--config", str(CONFIG), "--output-dir", str(output_dir), "--size", str(size), "--seed", str(seed)], cwd=str(REPO_ROOT), check=True)


class QuickStartDemoTests(unittest.TestCase):
    def test_generated_files_exist_and_are_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            run_demo(out)
            expected = [out / "visible" / "0001.png", out / "infrared" / "0001.png", out / "baseline_fused_preview.png", out / "preview_grid.png", out / "pose_annotations.csv", out / "manifest.json", out / "resolved_dataset.yaml", out / "summary.json"]
            for path in expected:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_image_shapes_match_requested_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            run_demo(out, size=128)
            visible = cv2.imread(str(out / "visible" / "0001.png"), cv2.IMREAD_COLOR)
            infrared = cv2.imread(str(out / "infrared" / "0001.png"), cv2.IMREAD_UNCHANGED)
            baseline = cv2.imread(str(out / "baseline_fused_preview.png"), cv2.IMREAD_COLOR)
            grid = cv2.imread(str(out / "preview_grid.png"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(visible)
            self.assertIsNotNone(infrared)
            self.assertIsNotNone(baseline)
            self.assertIsNotNone(grid)
            self.assertEqual(visible.shape, (128, 128, 3))
            self.assertEqual(infrared.shape, (128, 128))
            self.assertEqual(baseline.shape, (128, 128, 3))
            self.assertEqual(grid.shape[0], 128)
            self.assertEqual(grid.shape[1], 128 * 3)
            self.assertEqual(grid.shape[2], 3)

    def test_deterministic_outputs_for_same_seed_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_a = Path(tmp) / "a"
            out_b = Path(tmp) / "b"
            run_demo(out_a, size=128, seed=42)
            run_demo(out_b, size=128, seed=42)
            for rel in [Path("visible") / "0001.png", Path("infrared") / "0001.png", Path("baseline_fused_preview.png")]:
                img_a = cv2.imread(str(out_a / rel), cv2.IMREAD_UNCHANGED)
                img_b = cv2.imread(str(out_b / rel), cv2.IMREAD_UNCHANGED)
                self.assertTrue(np.array_equal(img_a, img_b), rel)

    def test_manifest_summary_and_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "demo"
            run_demo(out, size=128, seed=42)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["model_inference"])
            self.assertFalse(summary["gpu_required"])
            self.assertFalse(summary["checkpoint_used"])
            self.assertEqual(len(manifest["samples"]), 1)
            sample = manifest["samples"][0]
            self.assertTrue((out / sample["visible"]).exists())
            self.assertTrue((out / sample["infrared"]).exists())
            subprocess.run([sys.executable, str(VALIDATOR), "--config", str(out / "resolved_dataset.yaml"), "--check-paths"], cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    unittest.main()
