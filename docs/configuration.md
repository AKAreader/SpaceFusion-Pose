# Configuration Guide

## Why Configuration Templates Are Needed

This repository is a research codebase assembled from fusion, pose-estimation, and simulation-oriented experiments. Some legacy scripts still contain local absolute paths. The configuration templates provide a recommended direction for future cleanup, but not every script has been fully refactored yet.

The goal of the `configs/` directory is to make local paths and runtime assumptions visible before running training or evaluation.

The current templates are a recommended direction for new work. They do not mean that every historical script has already been converted to YAML-based configuration. For a map of current entry points, see `docs/entrypoints.md`. For a path-specific audit, see `docs/path_audit.md`.

## `configs/example_dataset.yaml`

This file is a dataset template. It should be copied or edited for a local machine before running experiments.

Important fields:

| Field | Meaning |
| --- | --- |
| `dataset.root` | Local dataset root directory. |
| `dataset.visible_dir` | Directory containing visible RGB images. |
| `dataset.infrared_dir` | Directory containing infrared images. |
| `dataset.annotation_file` | Pose annotation table, often Excel or CSV. |
| `dataset.manifest_file` | Optional manifest that maps visible/infrared pairs and labels. |
| `image.height`, `image.width` | Expected image size for a local experiment. |
| `pose.representation` | Pose representation, such as quaternion. |
| `pose.fields` | Column names used by local annotation files. |
| `split.*` | Suggested train/validation split settings. |

The full dataset is not bundled in this repository.

## `configs/runtime_safe.yaml`

This file documents conservative local runtime settings.

Important fields:

| Field | Meaning |
| --- | --- |
| `runtime.num_workers: 0` | Safer for Windows debugging and notebooks. |
| `runtime.pin_memory: false` | Avoids some host-memory issues during smoke testing. |
| `runtime.deterministic: true` | Encourages reproducibility when supported. |
| `training.batch_size: 1` | Useful for small-GPU or path-debugging runs. |
| `training.accum_steps: 8` | Example gradient accumulation setting. |
| `training.grad_clip: 1.0` | Conservative clipping value for unstable experiments. |

These settings are recommendations and are not automatically consumed by every legacy script.

## Migrating Local Absolute Paths

Several scripts may still contain paths such as `E:\...` or `D:\...`. A future cleanup pass should:

1. identify local constants in each script,
2. move dataset paths into YAML config files,
3. add command-line arguments for config selection,
4. keep defaults safe and non-destructive,
5. document required annotation columns.

Avoid committing private local dataset paths.

## Windows / PyCharm / PowerShell Notes

- Use PowerShell from the repository root when following command examples.
- Prefer `num_workers: 0` for first-run debugging on Windows.
- Configure interpreter and working directory explicitly in PyCharm.
- Keep generated outputs under ignored directories such as `outputs/`, `test_outputs/`, or `runs_*`.

## Git LFS Checkpoints

Model checkpoints under `model/` are tracked by Git LFS. After cloning:

```powershell
git lfs install
git lfs pull
```

Do not manually replace Git LFS pointer files with private large files unless you intend to publish them.

## Fields Users Should Usually Modify

Most users should review and update:

- `dataset.root`,
- `dataset.visible_dir`,
- `dataset.infrared_dir`,
- `dataset.annotation_file`,
- `dataset.manifest_file`,
- `pose.fields`,
- image size settings,
- runtime worker and batch-size settings.

Configuration templates are intended to make experiments easier to reproduce, but they do not guarantee full reproduction of all research runs.

Before selecting a script to run, review `docs/entrypoints.md` to understand which scripts are primary routes and which are historical or analysis utilities.
