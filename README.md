# Spacecraft Infrared-Visible Fusion and Pose Estimation Research Framework

## Overview

This repository provides a research-oriented framework for visible/infrared spacecraft perception, image fusion, and pose-oriented evaluation.

Spacecraft pose estimation is important for non-cooperative target perception, rendezvous, proximity operations, and on-orbit servicing. However, real multi-modal spacecraft data are difficult to acquire and annotate. Visible images provide texture and color structure under favorable illumination, while infrared images provide complementary thermal and silhouette cues under challenging illumination. This project explores a simulation-to-learning workflow that uses visible/infrared synthetic data and deep learning models for fusion and pose-related research.

In short, this repository is intended as:

> A simulation-to-deep-learning research framework for infrared-visible spacecraft perception and pose-oriented fusion.

It should be treated as a research prototype, not as validated spacecraft navigation software.

## Motivation

Infrared-visible image fusion is often evaluated as a visual enhancement problem. In spacecraft perception, fusion can also support downstream pose-aware tasks: maintaining object structure, silhouette information, and illumination-robust cues that are useful for pose estimation and analysis.

This project studies that direction through a practical research codebase:

- synthetic visible/infrared spacecraft-style data organization,
- fusion model training and inference utilities,
- pose-oriented training and evaluation scripts,
- qualitative and quantitative analysis tools.

## Key Features

- Visible/infrared spacecraft image fusion research code.
- Blender-style synthetic data workflow support through local manifests and preprocessing utilities.
- Pose-aware training and evaluation utilities.
- Multiple experimental routes, including `PlanA`, `PlanD`, and `PlanA_RGB`.
- Git LFS managed model checkpoints under `model/`.
- Evaluation utilities for fusion metrics and pose-related analysis.
- Technical notes in `docs/`, including SoPD-Net implementation references.

Some scripts are experimental utilities rather than polished command-line products. Paths, manifests, and dataset locations may need local adaptation before use.

## Research Pipeline

```text
Synthetic spacecraft scene generation
        |
        v
RGB/IR paired data organization
        |
        v
Infrared-visible fusion model training
        |
        v
Pose-oriented supervision / evaluation
        |
        v
Qualitative and quantitative analysis
```

The current repository focuses on the learning, fusion, evaluation, and analysis side of this workflow. Full local datasets and generated experiment outputs are not bundled in the repository.

## Repository Structure

```text
Evaluation/      MATLAB-style fusion metric evaluation utilities.
Figure/          Static figures used by the documentation and examples.
PlanA/           Experimental pose-oriented fusion route.
PlanA_RGB/       RGB-only pose route used for comparison and ablation-style work.
PlanD/           Additional pose-oriented data processing, ROI, and dual-route experiments.
configs/         Example dataset and conservative runtime configuration templates.
docs/            Project notes and technical references.
examples/        Lightweight manifest example and smoke-check script.
.github/workflows/  Lightweight CI smoke check.
model/           Model checkpoints tracked by Git LFS.
```

Useful documentation entry points:

```text
docs/entrypoints.md      Script map and recommended reading order.
docs/path_audit.md       Local path assumption audit.
docs/configuration.md    Configuration template guide.
docs/reproducibility.md  Reproducibility notes and checklists.
```

Ignored local directories include:

```text
datasets/
runs_*/
outputs/
test_outputs/
test_imgs/
SeAFusion/
PlanA/runs_pose*/
PlanD/runs_pose*/
PlanA_RGB/runs_pose_rgb/
```

These directories usually contain local datasets, generated outputs, temporary images, or training results. They are intentionally excluded from normal Git tracking.

## Installation

A conservative local setup is:

```powershell
conda create -n seafusion-pose python=3.8
conda activate seafusion-pose
pip install -r requirements.txt
```

Install PyTorch separately according to your CUDA environment:

```text
https://pytorch.org/get-started/locally/
```

Local development has used Windows / PowerShell and Python 3.8 style environments. PyTorch 2.4.1 has been used in local development, but the exact PyTorch build should be selected for the target CUDA and driver setup.

## Data Preparation

The full simulation dataset is not bundled in this repository.

Most workflows expect paired visible/infrared images and pose-related annotations. Depending on the script, these may be provided through:

- RGB and IR image directories,
- JSON manifest files,
- CSV or Excel annotation tables,
- locally generated ROI or saliency metadata.

Expected local dataset organization should be adapted through manifest files, script arguments, or configuration modules. Some older scripts still contain local absolute path assumptions, and future cleanup should move those paths into a `configs/` directory.

## Model Checkpoints

Model checkpoints under `model/` are tracked with Git LFS.

After cloning the repository, run:

```powershell
git lfs install
git lfs pull
```

The checkpoints are research artifacts. They should not be interpreted as complete validation evidence for a deployment system.

## Quick Start

The following commands show common entry points, but they may require local dataset paths and annotation files to be configured first:

Before running training scripts, review:

```text
configs/example_dataset.yaml
configs/runtime_safe.yaml
docs/configuration.md
docs/entrypoints.md
docs/path_audit.md
```

Run the lightweight smoke check first:

```powershell
python examples/quick_smoke_check.py
```

```powershell
python train_Cv2.py
python pose_unified_table.py
```

For basic fusion inference with the original-style SeAFusion entry point:

```powershell
python test.py --ir_dir ./test_imgs/ir --vi_dir ./test_imgs/vi --save_dir ./SeAFusion
```

The `test_imgs/` and `SeAFusion/` directories are ignored by Git because they are local input/output directories.

## Training and Evaluation Entry Points

See [docs/entrypoints.md](docs/entrypoints.md) for a more detailed entry-point map.

Training scripts:

```text
train.py
train_A.py
train_B_gate.py
train_C.py
train_Cv2.py
PlanA/train_pose.py
PlanD/train_pose.py
PlanA_RGB/train_pose.py
```

Testing and inference scripts:

```text
test.py
PlanA/test_pose.py
PlanD/test_pose.py
PlanA_RGB/test_pose.py
infer_routeA.py
```

Evaluation and reporting scripts:

```text
evaluate.py
pose_unified_table.py
fire_domain_pose_train_eval.py
hard_subset_pose_table.py
Evaluation/test_evaluation.m
```

Data preparation and analysis utilities:

```text
build_fire_manifests.py
make_map.py
PlanD/build_map.py
PlanD/build_roi.py
PlanD/generate_dataset_saliency.py
PlanD/merge_csv_to_json.py
```

## Reproducibility Notes

This repository is currently organized as a research codebase. Some scripts may require local path adaptation, dataset manifests, or pretrained checkpoints.

See [docs/reproducibility.md](docs/reproducibility.md) and [docs/path_audit.md](docs/path_audit.md) for the current reproducibility scope and known path assumptions.

Recommended cleanup work includes:

- moving hard-coded local paths into configuration files,
- adding small sample manifests,
- documenting expected annotation schemas,
- adding smoke tests for imports and lightweight inference,
- separating stable entry points from exploratory scripts.

## Limitations

- The full dataset is not bundled.
- Some scripts may contain local path assumptions.
- Results depend on local simulation settings and annotation consistency.
- Some directories in the working tree may contain ignored local experiments or generated images.
- This is a research prototype and should not be interpreted as a validated spacecraft navigation system.

## Citation

If you use this repository, please cite the repository or related papers when available. A formal `CITATION.cff` file should be added or maintained for citation metadata.

## License and Acknowledgements

This repository currently includes an MIT License file.

Parts of this codebase are adapted from or inspired by prior infrared-visible image fusion research codebases, including SeAFusion-style training and evaluation components. Please also respect the licenses and citations of the upstream projects where applicable.

The original SeAFusion work referenced by the inherited code is:

```bibtex
@article{TANG202228SeAFusion,
  title = {Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
  journal = {Information Fusion},
  volume = {82},
  pages = {28-42},
  year = {2022},
  issn = {1566-2535}
}
```
