# Entry Points

## Recommended Reading Order

1. `README.md`
2. `docs/project_overview.md`
3. `docs/configuration.md`
4. `docs/reproducibility.md`
5. `docs/directory_guide.md`
6. `docs/path_audit.md`

## Main Research Routes

This repository contains several research routes. Not every script is a polished command-line product, and many scripts require local dataset paths, annotation files, or checkpoints.

### Fusion-oriented scripts

- `train.py`
- `train_A.py`
- `train_B_gate.py`
- `train_C.py`
- `train_Cv2.py`

These scripts are the main root-level fusion training routes. Some contain local path defaults and should be reviewed before running.

### Pose-oriented routes

- `PlanA/train_pose.py`
- `PlanD/train_pose.py`
- `PlanA_RGB/train_pose.py`

These scripts explore pose-oriented training variants. `PlanA` and `PlanD` are experimental routes for infrared-visible or dual-route work, while `PlanA_RGB` is useful as an RGB-only comparison route.

### Testing / inference

- `test.py`
- `infer_routeA.py`
- `PlanA/test_pose.py`
- `PlanD/test_pose.py`
- `PlanA_RGB/test_pose.py`

`test.py` is the simplest original-style fusion inference entry point, but it still expects local test images and checkpoints.

### Evaluation

- `evaluate.py`
- `pose_unified_table.py`
- `fire_domain_pose_train_eval.py`
- `hard_subset_pose_table.py`
- `Evaluation/test_evaluation.m`

Evaluation scripts may depend on local result folders, local annotation tables, or MATLAB paths. Review `docs/path_audit.md` before treating these as portable commands.

### Data preparation

- `build_fire_manifests.py`
- `make_map.py`
- `PlanD/build_map.py`
- `PlanD/build_roi.py`
- `PlanD/generate_dataset_saliency.py`
- `PlanD/merge_csv_to_json.py`

These scripts help construct manifests, maps, ROI metadata, or derived inputs for local experiments.

## Recommended Public Quick Start

For external users, start with:

- `configs/example_dataset.yaml`
- `configs/runtime_safe.yaml`
- `examples/example_manifest.json`
- `examples/quick_smoke_check.py`

Then read:

- `docs/configuration.md`
- `docs/reproducibility.md`
- `docs/path_audit.md`

Run the smoke check from the repository root:

```powershell
python examples/quick_smoke_check.py
```

This check does not train models, download data, require a GPU, or verify scientific results.

## Notes

Many scripts were developed for local research experiments and may require path adaptation. The safest next engineering step is to add config-driven wrappers for a small number of recommended routes instead of rewriting all legacy scripts at once.
