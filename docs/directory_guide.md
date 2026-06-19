# Directory Guide

## Top-Level Directories

| Path | Purpose | Notes |
| --- | --- | --- |
| `Evaluation/` | Fusion metric evaluation utilities, including MATLAB-style scripts. | Useful for quantitative image-fusion analysis. |
| `Figure/` | Static figures used by documentation and examples. | Keep curated figures small and relevant. |
| `PlanA/` | Experimental pose-oriented fusion route. | Contains training, testing, metrics, and comparison scripts. |
| `PlanA_RGB/` | RGB-only pose-related route. | Useful as a comparison route for modality studies. |
| `PlanD/` | Additional pose-oriented experiments, ROI tools, saliency utilities, and dataset processing scripts. | Contains several exploratory scripts and configuration assumptions. |
| `docs/` | Project documentation and technical notes. | Start with `project_overview.md` and this guide. |
| `model/` | Model checkpoints. | Checkpoints are tracked with Git LFS. |

## Main Script Families

| Category | Scripts |
| --- | --- |
| Fusion training | `train.py`, `train_A.py`, `train_B_gate.py`, `train_C.py`, `train_Cv2.py` |
| Pose training | `PlanA/train_pose.py`, `PlanD/train_pose.py`, `PlanA_RGB/train_pose.py` |
| Inference / testing | `test.py`, `infer_routeA.py`, `PlanA/test_pose.py`, `PlanD/test_pose.py`, `PlanA_RGB/test_pose.py` |
| Evaluation | `evaluate.py`, `pose_unified_table.py`, `fire_domain_pose_train_eval.py`, `hard_subset_pose_table.py`, `Evaluation/test_evaluation.m` |
| Data preparation | `build_fire_manifests.py`, `make_map.py`, `PlanD/build_map.py`, `PlanD/build_roi.py`, `PlanD/generate_dataset_saliency.py`, `PlanD/merge_csv_to_json.py` |
| Visualization and reporting | `viz_report.py`, `make_result.py`, `make_module_effect_maps.py`, `PlanA/make_ab_crop_compare.py`, `PlanD/make_ab_crop_compare.py` |

## Ignored Local Directories

The following directories are normally local-only and ignored by Git:

| Pattern | Typical content |
| --- | --- |
| `datasets/` | Local datasets and manifests. |
| `outputs/`, `test_outputs/` | Generated outputs. |
| `test_imgs/` | Local test inputs. |
| `SeAFusion/` | Generated fusion results from `test.py`. |
| `runs_*`, `runs_compare/`, `runs_fusion/`, `runs_pose_unified/` | Training and experiment runs. |
| `PlanA/runs_pose*/`, `PlanD/runs_pose*/`, `PlanA_RGB/runs_pose_rgb/` | Pose experiment checkpoints and outputs. |
| `qualitative_report/`, `residual_report/`, `paper_figs_qualitative/` | Generated analysis figures. |
| `__pycache__/`, `.idea/` | Local cache and editor metadata. |

Do not add these directories to Git unless there is a deliberate release decision.

## Git LFS Managed Files

The repository tracks model checkpoint extensions with Git LFS:

```text
*.pt
*.pth
```

After cloning, use:

```powershell
git lfs install
git lfs pull
```

## Where New Contributors Should Start

1. Read `README.md` for the project scope and quick start.
2. Read `docs/project_overview.md` for the research motivation.
3. Inspect `test.py` for the simplest fusion inference entry point.
4. Inspect `train_Cv2.py` and `docs/SoPD-Net_technical_reference.md` for the current main fusion research route.
5. Inspect `PlanD/config_pose.py` before running PlanD scripts, because local dataset paths may need to be changed.
