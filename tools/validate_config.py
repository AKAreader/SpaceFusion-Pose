from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - dependency diagnostic
    yaml = None


REQUIRED_FIELDS = [
    "dataset.name",
    "dataset.root",
    "dataset.visible_dir",
    "dataset.infrared_dir",
    "dataset.annotation_file",
    "dataset.manifest_file",
    "image.height",
    "image.width",
    "image.visible_channels",
    "image.infrared_channels",
    "pose.representation",
    "pose.fields.w",
    "pose.fields.x",
    "pose.fields.y",
    "pose.fields.z",
    "split.train_ratio",
    "split.val_ratio",
    "split.random_seed",
]

PATH_FIELDS = [
    "dataset.root",
    "dataset.visible_dir",
    "dataset.infrared_dir",
    "dataset.annotation_file",
    "dataset.manifest_file",
]


def print_result(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def get_nested(data: Dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def is_placeholder(value: str) -> bool:
    normalized = value.replace("\\", "/").upper()
    return (
        normalized.startswith("PATH/TO/")
        or "PATH/TO/YOUR" in normalized
        or normalized in {"TODO", "TBD", ""}
    )


def load_yaml(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    if yaml is None:
        return {}, ["PyYAML is not installed. Install it with: pip install PyYAML"]
    if not path.exists():
        return {}, [f"Config file does not exist: {path}"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        return {}, [f"Failed to parse YAML: {exc}"]
    if not isinstance(data, dict):
        errors.append("YAML root must be a mapping/object.")
        return {}, errors
    return data, errors


def validate_required_fields(data: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for field in REQUIRED_FIELDS:
        value = get_nested(data, field)
        if value is None:
            missing.append(field)
    return missing


def validate_split(data: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    train_ratio = get_nested(data, "split.train_ratio")
    val_ratio = get_nested(data, "split.val_ratio")
    if isinstance(train_ratio, (int, float)) and isinstance(val_ratio, (int, float)):
        total = float(train_ratio) + float(val_ratio)
        if abs(total - 1.0) > 1e-6:
            warnings.append(f"split.train_ratio + split.val_ratio = {total:.3f}, expected approximately 1.0")
    return warnings


def check_paths(data: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for field in PATH_FIELDS:
        value = get_nested(data, field)
        if value is None:
            continue
        value_str = str(value)
        if is_placeholder(value_str):
            warnings.append(f"{field} is a placeholder: {value_str}")
            continue
        path = Path(value_str).expanduser()
        if not path.exists():
            warnings.append(f"{field} does not exist locally: {value_str}")
    return warnings


def check_manifest_structure(manifest_path: str) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if is_placeholder(manifest_path):
        print_result("INFO", f"Manifest structure check skipped for placeholder path: {manifest_path}")
        return warnings, errors

    path = Path(manifest_path).expanduser()
    if not path.exists():
        warnings.append(f"Manifest file does not exist locally: {manifest_path}")
        return warnings, errors

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "samples" in data:
                samples = data["samples"]
                if not isinstance(samples, list):
                    errors.append("JSON manifest field 'samples' must be a list.")
                else:
                    print_result("PASS", f"JSON manifest contains samples list with {len(samples)} item(s).")
            elif isinstance(data, list):
                print_result("PASS", f"JSON manifest root is a list with {len(data)} item(s).")
            else:
                warnings.append("JSON manifest is readable but does not use a recognized samples schema.")
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    errors.append("CSV manifest has no header row.")
                else:
                    print_result("PASS", f"CSV manifest has columns: {', '.join(reader.fieldnames)}")
        else:
            warnings.append(f"Manifest extension is not JSON or CSV: {suffix or '<none>'}")
    except Exception as exc:
        errors.append(f"Failed to read manifest structure: {exc}")
    return warnings, errors


def emit_list(level: str, messages: Iterable[str]) -> None:
    for message in messages:
        print_result(level, message)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate dataset YAML configuration structure.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML configuration file.")
    parser.add_argument("--check-paths", action="store_true", help="Warn about placeholder or missing local paths.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    print_result("INFO", f"Validating config: {config_path}")
    print_result("INFO", "This tool does not train models, download data, require a GPU, or read model weights.")

    data, load_errors = load_yaml(config_path)
    if load_errors:
        emit_list("FAIL", load_errors)
        return 1
    print_result("PASS", "YAML parsed successfully.")

    missing = validate_required_fields(data)
    if missing:
        emit_list("FAIL", [f"Missing required field: {field}" for field in missing])
        return 1
    print_result("PASS", "All required fields are present.")

    split_warnings = validate_split(data)
    emit_list("WARN", split_warnings)

    if args.check_paths:
        emit_list("WARN", check_paths(data))
        manifest_value = str(get_nested(data, "dataset.manifest_file"))
        manifest_warnings, manifest_errors = check_manifest_structure(manifest_value)
        emit_list("WARN", manifest_warnings)
        if manifest_errors:
            emit_list("FAIL", manifest_errors)
            return 1
    else:
        print_result("INFO", "Path existence checks skipped. Use --check-paths to enable warnings for local paths.")

    print_result("PASS", "Config validation completed without required failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
