# Tools

This directory contains lightweight helper tools for repository validation and future config-driven workflows.

## `validate_config.py`

`validate_config.py` checks the structure of a dataset configuration YAML file. It is intended to help users inspect local dataset settings before adapting training or evaluation scripts.

Recommended commands:

```powershell
python tools/validate_config.py --config configs/example_dataset.yaml
python tools/validate_config.py --config configs/example_dataset.yaml --check-paths
```

The tool only validates configuration structure and optional path existence. It does not train models, download data, import training modules, read images, read model weights, or require a GPU.

Placeholder paths such as `PATH/TO/YOUR/DATASET_ROOT` produce `[WARN]` messages rather than `[FAIL]` messages. Missing local paths also produce warnings when `--check-paths` is enabled, because the full dataset is not bundled in this repository.

This tool is a foundation for a future config-driven workflow. It does not mean that every legacy script already consumes the YAML configuration.
