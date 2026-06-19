# Roadmap

## Current Status

This repository is organized as a research codebase for infrared-visible spacecraft perception, simulation-driven fusion, and pose-oriented evaluation. It now includes project documentation, model and dataset cards, configuration templates, examples, smoke checks, entry-point guidance, path audit notes, and a configuration validator.

## Near-Term Goals

- Connect recommended wrappers to `configs/example_dataset.yaml` and `configs/runtime_safe.yaml`.
- Add a tiny synthetic or toy dataset sample when sharing permissions allow.
- Improve path configuration around the most commonly used scripts.
- Expand smoke checks for documentation links, config files, and lightweight metadata validation.
- Add clearer route-specific notes for PlanA, PlanD, and PlanA_RGB.

## Medium-Term Goals

- Standardize dataset manifest fields across data preparation and evaluation scripts.
- Document the intended use of PlanA, PlanD, and PlanA_RGB research routes.
- Add dry-run training or evaluation entry points that validate inputs without long runs.
- Add small evaluation examples that do not require the full local dataset.
- Improve CI coverage for lint-style checks and documentation consistency.

## Long-Term Goals

- Share cleaned benchmark splits if permissions and storage constraints allow.
- Add reproducible experiment cards for selected routes.
- Improve task-oriented fusion evaluation for pose-aware downstream use.
- Reduce local absolute path assumptions in frequently used scripts.
- Build a clearer bridge between simulation metadata, fusion training, and pose evaluation.

## Non-Goals

- This repository is not flight software.
- This repository is not an operational spacecraft navigation system.
- This repository does not claim effortless end-to-end reproduction without dataset preparation.
- This repository does not replace mission-specific validation, safety analysis, or hardware-in-the-loop testing.

## Contribution Opportunities

- Improve documentation around local dataset preparation.
- Add config-driven wrappers that preserve legacy script behavior.
- Create small, shareable examples for manifest validation and evaluation table generation.
- Add tests for configuration parsing and file structure checks.
- Help separate generated outputs from source-controlled research code.
