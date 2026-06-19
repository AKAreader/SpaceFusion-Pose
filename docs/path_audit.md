# Path Audit

## Purpose

This audit documents path assumptions in the current research codebase. It does not mean all scripts are broken. Some paths refer to ignored local outputs or historical experiments.

The goal is to make local path assumptions visible before future refactoring. This document is informational only; it does not modify the existing training, testing, or evaluation scripts.

## Summary

The repository contains several kinds of path references:

- direct Windows absolute paths used by training, testing, evaluation, and analysis scripts;
- local checkpoint and report paths from previous experiments;
- ignored output directory names such as `runs_*`, `outputs/`, and `test_outputs/`;
- documentation and template paths that intentionally show placeholders or examples.

The highest-risk items are scripts that use machine-specific defaults such as `E:\ALL`, `E:\Test`, `D:\BaiduNetdiskDownload`, or `D:\Redundancy\edgedownload\SeAFusion-main`. These paths should not be treated as portable defaults for external users.

## Risk Levels

| Level | Meaning |
| --- | --- |
| High | A local absolute path directly affects whether a script can run on another machine. |
| Medium | A default output, checkpoint, run, or historical experiment path may be local-specific but usually does not block understanding the repository. |
| Low | Documentation examples, `.gitignore` patterns, configuration templates, or directory names that intentionally describe local-only files. |

## High-risk Local Path Assumptions

Representative high-risk files include:

| Area | Example files | Typical assumptions |
| --- | --- | --- |
| Root fusion training | `train.py`, `train_A.py`, `train_B_gate.py`, `train_C.py`, `train_Cv2.py` | Defaults such as `E:\ALL\RGB`, `E:\ALL\IR`, and local Excel annotation paths. |
| Analysis / reporting | `doctor.py`, `evaluate.py`, `make_result.py`, `make_map.py`, `pose_unified_table.py`, `hard_subset_pose_table.py` | Local data roots, local comparison folders, and previous run checkpoints. |
| PlanA pose route | `PlanA/train_pose.py`, `PlanA/test_pose.py`, `PlanA/K_train_pose.py`, `PlanA/analyze_pose_errors.py` | `D:\BaiduNetdiskDownload\data\...` dataset defaults and local checkpoint paths. |
| PlanA_RGB route | `PlanA_RGB/train_pose.py`, `PlanA_RGB/test_pose.py`, `PlanA_RGB/make.py` | Local RGB-only data and comparison output paths. |
| PlanD route | `PlanD/config_pose.py`, `PlanD/train_pose_B.py`, `PlanD/test_pose.py`, `PlanD/build_map.py`, `PlanD/generate_dataset_saliency.py` | `D:\BaiduNetdiskDownload\All` and related local annotation paths. |
| MATLAB evaluation | `Evaluation/*.m` | Original local folders such as `D:\Graduate\...`, `D:\Github\SeAFusion\...`, and related dataset paths. |

These references are the first candidates for future config-driven wrappers or command-line arguments.

## Medium-risk Output or Experiment Paths

Representative medium-risk references include:

- `runs_fusion/...`
- `runs_pose/...`
- `runs_pose_fixed/...`
- `runs_pose_unified/...`
- `test_exports/...`
- `analysis_ir_test_*`
- `ablation_compare1/`
- `ab_crop_compare_*`
- generated report folders such as `qualitative_report/` and `residual_report/`

These usually point to ignored local experiment outputs. They are useful for preserving research context, but they should not be required for a clean first run.

## Low-risk Documentation or Template Paths

Low-risk references include:

- placeholders in `configs/example_dataset.yaml`;
- explanatory examples in `docs/configuration.md`;
- ignored directory patterns in README and `docs/directory_guide.md`;
- `examples/example_manifest.json` schema paths such as `visible/0001.png`.

These are intentional documentation or template references and are not considered portability bugs.

## Recommended Cleanup Strategy

1. Keep legacy scripts unchanged for reproducibility.
2. Add config-driven wrappers for recommended routes.
3. Move new experiments toward `configs/example_dataset.yaml` and `configs/runtime_safe.yaml`.
4. Document dataset paths instead of committing local data.
5. Prefer environment variables or explicit command-line arguments over local absolute defaults.
6. Introduce small smoke tests before attempting full training reproducibility.

## Do Not Change Yet

Do not rewrite every historical script at once. Some scripts encode specific experiment context, and changing them without a migration plan can make old results harder to interpret.

Recommended next steps are:

- select one public quick-start route;
- add a wrapper that reads `configs/example_dataset.yaml`;
- keep the original script available as a legacy reference;
- document any behavior changes before replacing defaults.
