# Dataset Card

## Dataset Purpose

The expected dataset supports research on infrared-visible spacecraft perception, image fusion, and pose-oriented evaluation.

It is designed for workflows where paired visible and infrared images are connected to pose annotations or related metadata.

## Data Modality

Expected modalities include:

- visible RGB images,
- infrared or thermal-style grayscale images,
- pose annotations,
- optional ROI, saliency, or manifest metadata.

## Synthetic Data Source

The intended data source is primarily local Blender-style simulation or locally generated synthetic spacecraft imagery. Simulation can provide controlled variation in viewpoint, pose, illumination, and modality.

The full dataset is not bundled in the repository.

## Annotation Type

Depending on the experiment, annotations may include:

- image pair identifiers,
- pose labels such as quaternion or angle-related values,
- split information for train / validation / test,
- ROI or object region metadata,
- CSV, JSON, or Excel-based manifests.

## Expected Local Format

Local scripts may expect one or more of the following:

```text
RGB image directory
IR image directory
train / val / test JSON files
CSV or Excel pose annotation tables
generated ROI or saliency metadata
```

Some scripts currently contain local path assumptions. Users should inspect the relevant configuration or script before running experiments.

## What Is Included / Not Included

Included in this repository:

- source code for data preparation and evaluation,
- example script structure,
- selected figures,
- Git LFS managed model checkpoints.

Not included:

- the full simulation dataset,
- all local generated outputs,
- all training runs,
- complete external benchmark datasets.

## Known Limitations

- Dataset format is not yet standardized into a single public schema.
- Some scripts use local absolute paths.
- Results can be sensitive to simulation rendering settings.
- Annotation consistency is essential for pose-oriented evaluation.
- Full reproducibility requires local dataset reconstruction or access to matching manifests.

## Recommended Usage

Use the dataset workflow for controlled research experiments. Keep raw datasets and generated outputs outside Git, and use manifests or configuration files to connect scripts to local data.

Recommended cleanup work:

- provide small sample manifests,
- document expected annotation columns,
- move local path assumptions into `configs/`,
- separate public examples from private local datasets.

## Citation / Attribution Notes

If a derived dataset or simulation setup is used in a publication, cite the repository and any upstream datasets, simulation assets, rendering tools, or image-fusion codebases that contributed to the data generation pipeline.
