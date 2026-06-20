# Blender Simulation Assets

This branch contains Blender-based simulation assets for visible/infrared spacecraft perception research.

## Purpose

These assets support simulation-driven research for paired visible and infrared rendering, material configuration, and pose-oriented perception experiments.

The main research code is maintained in the `seafusion-main` branch. This branch is intended for Blender simulation assets and related documentation.

## Directory Structure

```text
assets/
  blender/   Blender scenes, material files, embedded scripts, and related assets
configs/     Placeholder configuration files for future rendering workflows
docs/        Notes and documentation for simulation assets
```

## Included Assets

The files under `assets/blender/` are copied from the local source directory:

```text
D:\BaiduNetdiskDownload\fire
```

They may include Blender scene files, material settings, embedded scripts, or rendering-related resources.

## Git LFS

Binary 3D assets and large media files are tracked with Git LFS where appropriate.

After cloning this branch, run:

```bash
git lfs install
git lfs pull
```

## Relationship to Other Branches

- `seafusion-main`: main research code, documentation, configs, tools, and CI
- `blender-simulation-assets`: Blender simulation assets
- `blender-bag`: legacy Blender asset branch, temporarily retained
- `main`: legacy web/app branch

## Limitations

- This branch does not yet provide a fully documented rendering pipeline.
- Independent Python rendering scripts may not be included.
- Some material or scripting logic may be embedded inside `.blend` files.
- These assets are research materials, not validated production simulation assets.

## Recommended Next Steps

- Add Blender version information.
- Add rendering configuration examples.
- Document scene structure, camera settings, lighting settings, and material assumptions.
- Add paired visible/infrared rendering scripts when available.
