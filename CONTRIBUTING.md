# Contributing

Thank you for considering a contribution. This repository is a research prototype for infrared-visible spacecraft perception, simulation-driven fusion, and pose-oriented evaluation. Contributions should keep the project careful, reproducible where possible, and honest about its current limitations.

## Project Scope

The project focuses on research code, documentation, configuration templates, examples, evaluation utilities, and maintainability improvements for visible/infrared spacecraft fusion and pose-related experiments.

## Before You Start

- Read `README.md`, `docs/project_overview.md`, `docs/configuration.md`, and `docs/entrypoints.md`.
- Run the smoke check and configuration validator before proposing code changes.
- Review `docs/path_audit.md` before changing legacy scripts that may contain local path assumptions.

## Development Setup

```powershell
pip install -r requirements.txt
python examples/quick_smoke_check.py
python tools/validate_config.py --config configs/example_dataset.yaml
```

Install PyTorch separately according to your CUDA and operating system environment.

## Recommended Workflow

1. Open an issue or describe the planned change before large edits.
2. Keep changes small and focused.
3. Prefer new wrappers, docs, tests, or config templates before rewriting legacy research scripts.
4. Document dataset assumptions, checkpoint requirements, and hardware assumptions.

## What Contributions Are Welcome

- Documentation improvements.
- Configuration templates and validation checks.
- Small examples that do not require private datasets.
- Bug fixes with clear reproduction steps.
- Evaluation utilities and analysis scripts with transparent assumptions.
- Refactoring that preserves existing research behavior.

## What Contributions Are Out of Scope

- Private datasets, credentials, access tokens, or sensitive mission data.
- Large generated outputs, cache directories, or experiment dumps.
- Untracked model files or large checkpoints that should use Git LFS.
- Claims of new experimental results without scripts, configs, and reproducibility notes.
- Changes that present the repository as operational spacecraft software.

## Reporting Bugs

Use the bug report template when possible. Include the command, environment, config file, relevant logs, and whether `examples/quick_smoke_check.py` passed.

## Suggesting Features

Feature requests should describe the research use case, expected behavior, alternatives considered, and whether the change affects training, evaluation, configuration, or documentation.

## Reproducibility Reports

For reproducibility questions, include the route you tried, dataset status, checkpoint status, local path changes, config file, command, and where the result diverged.

## Pull Request Checklist

- I did not commit private datasets, credentials, or generated outputs.
- I did not modify model checkpoints unless the change is necessary and documented.
- I used Git LFS for large checkpoints when appropriate.
- I updated documentation if behavior changed.
- I ran `python examples/quick_smoke_check.py`.
- I ran `python tools/validate_config.py --config configs/example_dataset.yaml`.
- I described limitations and local assumptions when relevant.

## Style and Documentation Guidelines

Use clear, conservative language. Avoid overstating experimental status. Prefer explicit paths, config examples, and small reproducible checks over broad claims.
