"""
Repository file discovery: find Python files under a root path with default ignore rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

DEFAULT_IGNORE_DIRS: tuple[str, ...] = (
    ".venv",
    "venv",
    "site-packages",
    "tests",
    ".git",
    "__pycache__",
)


def discover_python_files(
    root_path: Path | str,
    *,
    ignore_dirs: Sequence[str] | None = None,
) -> list[Path]:
    """
    Discover all Python files under root_path, excluding paths that contain
    any of the ignored directory names as a path segment.

    Args:
        root_path: Directory to scan (e.g. repo root).
        ignore_dirs: Directory names to exclude. If None, use DEFAULT_IGNORE_DIRS.

    Returns:
        Sorted list of Paths to .py files (deterministic order).
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        return []

    names_to_ignore = set(ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS)
    result: list[Path] = []

    for path in root.rglob("*.py"):
        if path.is_file() and not path.is_symlink():
            parts = path.relative_to(root).parts
            if not any(part in names_to_ignore for part in parts):
                result.append(path)

    return sorted(result)
