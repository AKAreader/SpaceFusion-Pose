# Reproducibility Notes

## Scope

This repository is a research codebase. Reproduction may require adapting local dataset paths, annotation files, and hardware-specific settings.

The current Phase 2 additions provide non-invasive reproducibility scaffolding: configuration templates, example manifests, a smoke check, and a lightweight CI workflow.

## What Is Currently Reproducible

Without the full dataset, users can currently reproduce repository-level checks:

- file and directory structure checks,
- lightweight dependency import checks,
- Git LFS pointer visibility checks,
- documentation and configuration review.

The smoke check does not train models and does not validate scientific results.

## What Requires Local Data

Training, evaluation, and pose-oriented experiments require local data such as:

- paired visible/infrared images,
- pose annotations,
- manifests mapping RGB/IR image pairs,
- optional ROI or saliency metadata,
- Git LFS checkpoints where required by the selected script.

The full simulation dataset is not bundled in this repository.

## Suggested Environment

A conservative local environment is:

```powershell
conda create -n seafusion-pose python=3.8
conda activate seafusion-pose
pip install -r requirements.txt
```

Install PyTorch and torchvision separately according to the target CUDA environment.

## Git LFS Setup

After cloning:

```powershell
git lfs install
git lfs pull
```

If checkpoint files are still small text pointer files after cloning, run `git lfs pull` again and check network access.

## Dataset Preparation Checklist

- Prepare visible RGB images.
- Prepare infrared images.
- Verify pair identifiers.
- Prepare pose annotation files.
- Confirm quaternion or pose field names.
- Create or update a manifest file.
- Copy `configs/example_dataset.yaml` and replace placeholder paths.
- Keep raw datasets outside Git.

## Training Checklist

- Confirm local dataset paths.
- Confirm checkpoint availability if resuming or evaluating.
- Start with conservative runtime settings.
- Use small batch sizes for first-run debugging.
- Avoid committing generated runs or private data.
- Record script, config, checkpoint, and dataset version for each experiment.

## Evaluation Checklist

- Confirm evaluation input directories.
- Confirm pose annotation consistency.
- Record metric scripts and versions.
- Keep generated reports in ignored output directories.
- Do not present local exploratory outputs as validated benchmark results without review.

## Known Blockers

- Some legacy scripts contain local absolute paths.
- Dataset schema is not yet standardized.
- No full public dataset is bundled.
- No full benchmark CI is provided.
- Lightweight CI does not install PyTorch or run GPU workflows.

## Future Improvements

- Refactor scripts to consume YAML configs.
- Add minimal sample data or synthetic toy samples.
- Add import-only tests for stable modules.
- Add schema validation for manifests.
- Add a small documented inference path that does not require private data.
- Add formal citation metadata.
