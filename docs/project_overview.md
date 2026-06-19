# Project Overview

## Research Background

Non-cooperative spacecraft perception is a difficult problem because the target may have unknown attitude, complex geometry, strong illumination changes, and limited observable texture. Pose estimation is especially important for rendezvous, proximity operations, inspection, and on-orbit servicing.

Real paired visible/infrared spacecraft imagery with reliable pose labels is difficult to collect at scale. This repository therefore focuses on a simulation-to-deep-learning workflow for studying infrared-visible spacecraft perception and pose-oriented fusion.

## Why Visible + Infrared

Visible images can preserve color, texture, edges, and surface details when illumination is favorable. Infrared images can provide complementary thermal or silhouette cues when visible imagery is degraded by lighting, shadow, or contrast changes.

The central research question is not only whether fusion improves human visual appearance. The more relevant question for this project is whether fused representations can preserve information useful for downstream pose-aware perception.

## Why Simulation

Simulation allows controlled variation of:

- target geometry,
- pose and viewpoint,
- illumination,
- sensor modality,
- background and rendering conditions,
- annotation consistency.

Blender-style synthetic data workflows are useful because they can provide paired RGB/IR-style images and pose annotations under repeatable conditions. The full local simulation dataset is not bundled in this repository.

## Why Pose-Oriented Fusion

Pose-oriented fusion treats image fusion as part of a perception pipeline. The fused output should preserve structure, contours, and object cues that are relevant to attitude estimation, not only maximize generic visual quality.

The repository explores this idea through several experimental routes:

- `PlanA`: pose-oriented fusion experiments and analysis utilities,
- `PlanD`: additional ROI, saliency, dual-route, and data processing experiments,
- `PlanA_RGB`: RGB-only pose-related comparison utilities,
- root-level training and evaluation scripts for SeAFusion-style and SoPD-Net-style experiments.

## What This Repository Contains

The repository contains:

- visible/infrared fusion model code,
- pose-related training and testing scripts,
- local data preparation utilities,
- evaluation scripts for image fusion and pose-related reporting,
- Git LFS tracked model checkpoints,
- documentation notes for the current implementation.

## What This Repository Does Not Claim

This repository does not claim:

- production readiness,
- flight validation,
- complete public release of the full dataset,
- superior performance claims without separately verifiable experiments,
- universal compatibility across all operating systems and CUDA versions.

It is a research prototype intended for controlled experimentation and further engineering cleanup.

## Roadmap

Short-term documentation work:

- maintain a clear README,
- add model and dataset cards,
- document directory structure and entry points,
- provide dependency guidance.

Engineering cleanup:

- move local absolute paths into configuration files,
- add sample manifests,
- add smoke tests,
- separate stable scripts from exploratory scripts.

Research packaging:

- document simulation assumptions,
- provide reproducibility notes,
- add formal citation metadata,
- prepare concise experiment tables only when results are verified.
