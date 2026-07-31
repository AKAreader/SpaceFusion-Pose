# Changelog

All notable changes to this research repository will be documented here.

## [Unreleased]

## [0.1.0] - 2026-08-01

### Added

- Deterministic zero-data CPU quick-start demo with synthetic aligned RGB/IR inputs, a transparent baseline preview, manifest generation, resolved dataset configuration, pose annotations, run summaries, tests, and CI coverage.
- Research-oriented README, model card, dataset card, reproducibility guidance, configuration documentation, entry-point map, roadmap, and path audit.
- Dataset and runtime configuration templates under `configs/`.
- Example manifest and lightweight repository smoke check under `examples/`.
- GitHub Actions workflow for the public quick start, unit tests, smoke checks, and configuration validation.
- Configuration validation utility under `tools/`.
- Citation, contribution, security, issue-template, pull-request-template, and community maintenance files.
- Git LFS metadata for research checkpoint files.

### Changed

- Positioned the repository as an infrared-visible spacecraft perception, fusion, and pose-oriented research framework.
- Separated the zero-data public demonstration from local research training and inference entry points.
- Improved documentation links among the README, Quick Start, configuration notes, reproducibility guidance, and script entry points.
- Updated GitHub Actions triggers to follow the current default branch, `spacefusion-pose`.

### Fixed

- Replaced stale repository metadata and documentation links that referenced the former repository location.
- Corrected GitHub Actions triggers that still referenced the former default branch.

### Notes

- The full simulation dataset is not bundled with this release.
- The public Quick Start is a deterministic synthetic CPU baseline and is not SoPD-Net or SeAFusion trained-model inference.
- The release does not claim a fixed benchmark score or validated flight performance.
- Some legacy research scripts still contain local path assumptions and may require configuration before use.
- Model checkpoints are research artifacts and should not be interpreted as operational spacecraft software.

[Unreleased]: https://github.com/AKAreader/SpaceFusion-Pose/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AKAreader/SpaceFusion-Pose/releases/tag/v0.1.0
