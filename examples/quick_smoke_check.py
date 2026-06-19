from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, List, Tuple


DEPENDENCIES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
}

KEY_PATHS = [
    "README.md",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "requirements.txt",
    "configs/example_dataset.yaml",
    "configs/runtime_safe.yaml",
    "docs/entrypoints.md",
    "docs/path_audit.md",
    "tools",
    "tools/validate_config.py",
    "tools/README.md",
    "model",
    "PlanA",
    "PlanD",
    "Evaluation",
]


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / ".git").exists():
            return candidate
        if (candidate / "README.md").exists() and (candidate / "model").exists():
            return candidate
    return current


def print_result(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def check_dependencies() -> Tuple[List[str], List[str]]:
    passed: List[str] = []
    failed: List[str] = []
    for module_name, package_name in DEPENDENCIES.items():
        try:
            importlib.import_module(module_name)
            passed.append(package_name)
            print_result("PASS", f"Dependency import succeeded: {module_name}")
        except Exception as exc:  # pragma: no cover - diagnostic output
            failed.append(package_name)
            print_result("FAIL", f"Dependency import failed: {module_name} ({exc})")
    return passed, failed


def check_paths(repo_root: Path) -> Tuple[List[str], List[str]]:
    passed: List[str] = []
    failed: List[str] = []
    for relative in KEY_PATHS:
        path = repo_root / relative
        if path.exists():
            passed.append(relative)
            print_result("PASS", f"Found required path: {relative}")
        else:
            failed.append(relative)
            print_result("FAIL", f"Missing required path: {relative}")
    return passed, failed


def iter_checkpoint_files(model_dir: Path) -> Iterable[Path]:
    if not model_dir.exists():
        return []
    return list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth"))


def check_lfs_pointers(repo_root: Path) -> Tuple[int, int]:
    checkpoint_files = list(iter_checkpoint_files(repo_root / "model"))
    if not checkpoint_files:
        print_result("WARN", "No checkpoint files found under model/.")
        return 0, 0

    pointer_like = 0
    checked = 0
    for path in checkpoint_files[:20]:
        checked += 1
        try:
            with path.open("rb") as handle:
                header = handle.read(128)
            if b"git-lfs" in header or b"version https://git-lfs.github.com/spec" in header:
                pointer_like += 1
        except Exception as exc:  # pragma: no cover - diagnostic output
            print_result("WARN", f"Could not inspect checkpoint header: {path} ({exc})")

    if pointer_like:
        print_result("PASS", f"Detected Git LFS pointer-style files: {pointer_like}/{checked} inspected.")
    else:
        print_result("WARN", "No Git LFS pointer header detected in inspected checkpoint files. This may be normal after git lfs pull.")
    return checked, pointer_like


def check_optional_data_dirs(repo_root: Path) -> List[str]:
    warnings: List[str] = []
    for relative in ["datasets", "test_imgs"]:
        if not (repo_root / relative).exists():
            warnings.append(relative)
            print_result("WARN", f"Optional local data directory not found: {relative}")
    return warnings


def main() -> int:
    repo_root = find_repo_root(Path.cwd())
    print_result("INFO", f"Repository root: {repo_root}")
    print_result("INFO", "This smoke check does not train models, download data, require a GPU, or read checkpoint tensors.")

    _, dependency_failures = check_dependencies()
    _, path_failures = check_paths(repo_root)
    check_lfs_pointers(repo_root)
    check_optional_data_dirs(repo_root)

    if dependency_failures or path_failures:
        print_result("FAIL", "Smoke check completed with required failures.")
        return 1

    print_result("PASS", "Smoke check completed without required failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
