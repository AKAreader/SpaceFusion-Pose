# Examples

This directory contains lightweight examples for understanding the repository layout and expected local data organization.

## Files

| File | Purpose |
| --- | --- |
| `example_manifest.json` | A small schema example for paired visible/infrared spacecraft samples. |
| `quick_smoke_check.py` | A lightweight environment and repository sanity check. |

## Manifest Example

`example_manifest.json` is a schema example only. It demonstrates how a paired sample may connect:

- a visible RGB image path,
- an infrared image path,
- a pose annotation,
- optional metadata.

The file does not include real images and does not represent the full dataset.

## Dataset Availability

The full dataset is not bundled in this repository. Users should prepare local RGB/IR paired images, pose annotations, and manifest files according to their own simulation or data collection setup.

## Smoke Check

Run from the repository root:

```powershell
python examples/quick_smoke_check.py
```

The smoke check only verifies basic Python dependencies, key repository paths, and whether Git LFS pointer-style checkpoint files are visible. It does not train models, download data, require a GPU, or read large checkpoint tensors.

## Related Documentation

Before running training or evaluation scripts, review:

- `docs/configuration.md`
- `docs/reproducibility.md`
- `docs/entrypoints.md`
- `docs/path_audit.md`

`example_manifest.json` is not a dataset. It is only a compact schema example for understanding how paired visible/infrared samples and pose annotations may be represented.
