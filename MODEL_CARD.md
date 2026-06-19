# Model Card

## Model Purpose

The models in this repository are research artifacts for infrared-visible image fusion and pose-oriented spacecraft perception experiments.

They are intended to support analysis of how visible and infrared cues can be combined for downstream pose-aware perception under synthetic or locally prepared data conditions.

## Intended Use

Intended uses include:

- research on visible/infrared image fusion,
- synthetic spacecraft perception experiments,
- pose-oriented fusion training and evaluation,
- ablation studies across experimental routes such as `PlanA`, `PlanD`, and `PlanA_RGB`,
- local qualitative and quantitative analysis.

## Out-of-Scope Use

These models are not intended for:

- flight software,
- safety-critical spacecraft navigation,
- autonomous operational decision making,
- claims of validated real-world deployment performance,
- use without checking dataset assumptions and preprocessing conventions.

The checkpoints are research artifacts, not validated flight software.

## Inputs / Outputs

Typical inputs:

- visible RGB images,
- infrared or grayscale thermal-style images,
- paired image identifiers,
- optional pose annotations, ROI metadata, or manifests depending on the script.

Typical outputs:

- fused visible/infrared images,
- intermediate training outputs,
- pose-related predictions or evaluation reports for specific experimental scripts.

Exact input and output formats depend on the selected script and local configuration.

## Checkpoints

Model checkpoints are stored under `model/` and tracked with Git LFS. The repository uses:

```text
*.pt
*.pth
```

for Git LFS tracking.

After cloning:

```powershell
git lfs install
git lfs pull
```

## Training Data Description

Training and evaluation workflows are expected to use paired RGB/IR images and pose-related annotations. The intended data source is primarily local Blender-style simulation or locally prepared synthetic datasets.

The full dataset is not bundled in this repository.

## Evaluation Status

The repository includes evaluation utilities and reporting scripts, but this model card does not claim a fixed benchmark score. Any performance statement should be tied to a specific dataset version, script, configuration, checkpoint, and reproducible result table.

## Limitations

- Some scripts contain local path assumptions.
- Dataset organization may need adaptation.
- Model behavior depends on simulation settings and annotation consistency.
- Checkpoints may not cover all experimental routes.
- The repository is a research prototype, not a production package.

## Ethical / Safety Notes

Spacecraft perception research can inform high-stakes autonomy. Results from this repository should be interpreted carefully and should not be used as operational evidence without independent validation, stress testing, and domain-specific safety review.
