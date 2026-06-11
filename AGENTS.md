# AGENTS.md

## Project goal
This repository implements SoPD-Net for infrared-visible image fusion and downstream spacecraft pose estimation.

## Current paper status
The paper already has:
- fusion metrics table
- qualitative fusion examples
- progressive ablation: Baseline / TrainB / TrainC
- convergence curve

The highest-priority missing evidence is:
1. a unified downstream pose evaluation table comparing Visible only / Infrared only / SeAFusion / SoPD-Net
2. a minimal single-variable ablation, preferably SoPD-Net w/o pose loss

## Hard constraints
- Do not redesign the whole model unless explicitly asked.
- Prefer minimal engineering changes with maximal paper value.
- Keep all existing training and evaluation scripts usable.
- Preserve reproducibility and avoid changing data splits silently.
- When changing code, explain exactly which files changed and why.

## Success criteria
- Reproduce current reported results as closely as possible.
- Add the missing experiments with minimal code changes.
- Export paper-ready tables:
  - unified downstream pose evaluation
  - optional pose-loss ablation
- Report Mean / Median / P95 pose errors.
- Highlight any mismatch between paper claims and actual implemented experiments.

## Preferred workflow
1. inspect repo structure
2. locate training/evaluation entry points
3. identify current experiment pipeline
4. propose the smallest viable changes
5. implement only after confirming the plan in the repository notes/output
6. run evaluation and summarize results