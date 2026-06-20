# Asset Card: Blender Spacecraft Simulation Assets

## Asset Purpose

These Blender files are intended to support visible/infrared spacecraft simulation experiments and material-based rendering exploration.

## Asset Type

- Blender scene/assets files
- Possible embedded material configuration
- Possible embedded scene or scripting components
- Possible rendering or simulation resources

## Intended Use

- Research prototyping
- Visible/infrared rendering experiments
- Spacecraft perception dataset preparation
- Material and illumination exploration
- Pose-oriented synthetic data preparation

## Out-of-Scope Use

- Operational spacecraft navigation
- Production simulation validation
- Safety-critical decision making
- Claims of real-world sensor equivalence without additional validation

## Included Content

The branch stores assets under:

```text
assets/blender/
```

The exact file list should be inspected with:

```bash
git ls-files
```

## Known Limitations

- No standalone rendering Python pipeline is guaranteed yet.
- Material and scene settings may need to be inspected inside Blender.
- The assets are not a complete public benchmark dataset.
- Additional documentation is needed for fully reproducible rendering.

## Data and Licensing Notes

Do not add private, restricted, or third-party assets without verifying redistribution rights.

## Recommended Next Steps

- Record Blender version and add-on requirements.
- Export scene structure notes.
- Add material configuration notes.
- Add camera and lighting setup documentation.
- Add rendering scripts when available.
