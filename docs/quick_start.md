# Public Quick Start

## Purpose

This quick start is a zero-data CPU demonstration of repository structure, manifest generation, configuration validation, and a transparent fusion baseline.

It is not SoPD-Net inference, SeAFusion inference, or a research benchmark. It does not download datasets, read model checkpoints, require PyTorch, or require a GPU.

## Installation

```powershell
conda create -n spacefusion-quickstart python=3.8
conda activate spacefusion-quickstart
pip install -r requirements.txt
```

The public quick start does not require PyTorch. Install PyTorch separately only when you are preparing to run research training or model inference scripts.

## Run

Default run:

```powershell
python examples/quick_start_demo.py
```

Custom Windows / PowerShell run:

```powershell
python examples/quick_start_demo.py `
  --output-dir outputs/my_demo `
  --size 256 `
  --seed 42
```

Linux/macOS run:

```bash
python examples/quick_start_demo.py \
  --output-dir outputs/my_demo \
  --size 256 \
  --seed 42
```

## Generated Files

The default command writes files under `outputs/quick_start/`, which is ignored by Git:

```text
outputs/quick_start/
|-- visible/
|   `-- 0001.png
|-- infrared/
|   `-- 0001.png
|-- baseline_fused_preview.png
|-- preview_grid.png
|-- pose_annotations.csv
|-- manifest.json
|-- resolved_dataset.yaml
`-- summary.json
```

`visible/0001.png` is a deterministic synthetic RGB spacecraft-style input.

`infrared/0001.png` is a spatially aligned synthetic single-channel infrared input.

`baseline_fused_preview.png` is a transparent CPU baseline that blends visible luminance and infrared intensity. It is not trained-model inference.

`preview_grid.png` places the visible input, infrared input, and baseline preview side by side.

`pose_annotations.csv` stores one synthetic quaternion pose row for testing the data contract.

`manifest.json` follows the example paired-sample schema used by the repository.

`resolved_dataset.yaml` is generated from the demo outputs and validated with `tools/validate_config.py --check-paths`.

`summary.json` records the run status and confirms that no checkpoint, GPU, network access, or trained model inference was used.

## Next Steps

Use this zero-data public demo first to confirm the repository entry points and data contract.

For local real-data configuration, review:

- `configs/example_dataset.yaml`
- `docs/configuration.md`
- `docs/entrypoints.md`
- `docs/reproducibility.md`

For research training and evaluation, configure local paired visible/infrared data, pose annotations, and checkpoints before running the experimental scripts listed in `docs/entrypoints.md`.

## Limitations

- The synthetic images are not research data.
- The baseline preview is not a trained model.
- The demo does not represent paper metrics or benchmark performance.
- The demo does not validate real-scene or flight behavior.
- The purpose is to validate the public repository entrance and expected data contract.
