"""Tests for repository file discovery and ignore rules."""

import tempfile
from pathlib import Path

from noctyl_scout.ingestion import DEFAULT_IGNORE_DIRS, discover_python_files


def test_default_ignores():
    """discover_python_files(root) excludes .venv, venv, site-packages, tests, .git, __pycache__."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "lib").mkdir()
        (root / "src" / "app.py").write_text("# app")
        (root / "src" / "lib" / "foo.py").write_text("# foo")

        (root / ".venv").mkdir()
        (root / ".venv" / "ignore.py").write_text("#")
        (root / "venv").mkdir()
        (root / "venv" / "x.py").write_text("#")
        (root / "site-packages" / "x").mkdir(parents=True)
        (root / "site-packages" / "x" / "ignore.py").write_text("#")
        (root / "tests").mkdir()
        (root / "tests" / "test_foo.py").write_text("#")
        (root / ".git").mkdir()
        (root / ".git" / "ignore.py").write_text("#")
        (root / "src" / "__pycache__").mkdir()
        (root / "src" / "__pycache__" / "bar.py").write_text("#")

        result = discover_python_files(root)
        result_rel = sorted(p.relative_to(root) for p in result)
        assert result_rel == [
            Path("src/app.py"),
            Path("src/lib/foo.py"),
        ]


def test_custom_ignore_dirs():
    """Custom ignore_dirs excludes only those directory names."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("#")
        (root / "custom_skip").mkdir()
        (root / "custom_skip" / "bar.py").write_text("#")

        result = discover_python_files(root, ignore_dirs=("custom_skip",))
        result_rel = sorted(p.relative_to(root) for p in result)
        assert result_rel == [Path("src/app.py")]


def test_determinism():
    """Same root yields same sorted list when called twice."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a").mkdir()
        (root / "b").mkdir()
        (root / "a" / "x.py").write_text("#")
        (root / "b" / "y.py").write_text("#")

        r1 = discover_python_files(root)
        r2 = discover_python_files(root)
        assert r1 == r2
        assert r1 == sorted(r1)


def test_empty_root():
    """Root with no .py files or only ignored dirs returns empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert discover_python_files(root) == []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tests").mkdir()
        (root / "tests" / "test_only.py").write_text("#")
        assert discover_python_files(root) == []


def test_default_ignore_dirs_constant():
    """DEFAULT_IGNORE_DIRS includes required names from issue."""
    assert ".venv" in DEFAULT_IGNORE_DIRS
    assert "venv" in DEFAULT_IGNORE_DIRS
    assert "site-packages" in DEFAULT_IGNORE_DIRS
    assert "tests" in DEFAULT_IGNORE_DIRS
    assert ".git" in DEFAULT_IGNORE_DIRS
    assert "__pycache__" in DEFAULT_IGNORE_DIRS
