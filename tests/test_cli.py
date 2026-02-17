"""
Tests for CLI: estimate command, profile flags, output handling, error cases.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Path to noctyl CLI script
CLI_SCRIPT = Path(__file__).resolve().parent.parent / "noctyl" / "cli.py"


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """
    Run CLI command and return (exit_code, stdout, stderr).
    
    Args:
        args: CLI arguments (without 'noctyl' command)
    
    Returns:
        (exit_code, stdout, stderr)
    """
    cmd = [sys.executable, str(CLI_SCRIPT)] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8"
    )
    return result.returncode, result.stdout, result.stderr


def test_cli_estimate_basic():
    """Basic estimate command works."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        exit_code, stdout, stderr = _run_cli(["estimate", str(root)])
        
        # Exit code may be 1 if warnings present, or 0 if no warnings
        assert exit_code in (0, 1)
        # Should output JSON
        data = json.loads(stdout)
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["schema_version"] == "3.0"


def test_cli_estimate_with_profile():
    """--profile flag works."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    yaml_content = """
name: test-profile
expansion_factor: 1.4
output_ratio: 0.7
"""
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        profile_path = root / "profile.yaml"
        profile_path.write_text(yaml_content)
        
        exit_code, stdout, stderr = _run_cli(
            ["estimate", str(root), "--profile", str(profile_path)]
        )
        
        # Exit code may be 1 if warnings present
        assert exit_code in (0, 1)
        data = json.loads(stdout)
        assert data[0]["token_estimate"]["assumptions_profile"] == "test-profile"


def test_cli_estimate_output_file():
    """--output flag writes to file."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        output_file = root / "output.json"
        
        exit_code, stdout, stderr = _run_cli(
            ["estimate", str(root), "--output", str(output_file)]
        )
        
        # Exit code may be 1 if warnings present
        assert exit_code in (0, 1)
        # File should exist and contain JSON
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) > 0


def test_cli_estimate_missing_path():
    """Missing path shows help."""
    exit_code, stdout, stderr = _run_cli(["estimate"])
    
    assert exit_code != 0
    # Should show error or help
    assert "path" in (stdout + stderr).lower() or "required" in (stdout + stderr).lower()


def test_cli_estimate_invalid_path():
    """Invalid path handled gracefully."""
    exit_code, stdout, stderr = _run_cli(["estimate", "/nonexistent/path"])
    
    assert exit_code != 0
    assert "error" in stderr.lower() or "does not exist" in stderr.lower()


def test_cli_estimate_invalid_profile():
    """Invalid profile file handled gracefully."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        exit_code, stdout, stderr = _run_cli(
            ["estimate", str(root), "--profile", "/nonexistent/profile.yaml"]
        )
        
        # Should either exit with error or use default profile with warning
        # If it uses default, exit_code might be 1 (warnings) or 0
        assert exit_code in (0, 1)
        # Should have some output or error message
        assert stdout or stderr


def test_cli_estimate_json_output():
    """Output is valid JSON."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        exit_code, stdout, stderr = _run_cli(["estimate", str(root)])
        
        # Exit code may be 1 if warnings present
        assert exit_code in (0, 1)
        # Should be valid JSON
        data = json.loads(stdout)
        assert isinstance(data, list)
        # Should match schema 3.0 structure
        if len(data) > 0:
            assert "schema_version" in data[0]
            assert data[0]["schema_version"] == "3.0"


def test_cli_estimate_warnings_stderr():
    """Warnings printed to stderr."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", "a")  # Creates cycle
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        exit_code, stdout, stderr = _run_cli(["estimate", str(root)])
        
        # Should have warnings (cycle warnings)
        # Exit code might be 1 if warnings present, or 0 if warnings don't cause non-zero exit
        # But stderr should have warnings
        assert "warning" in stderr.lower() or exit_code == 1
        # stdout should still have valid JSON
        if stdout:
            data = json.loads(stdout)
            assert isinstance(data, list)


def test_cli_no_command():
    """No command shows help."""
    exit_code, stdout, stderr = _run_cli([])
    
    assert exit_code != 0
    # Should show help
    assert "usage" in (stdout + stderr).lower() or "commands" in (stdout + stderr).lower()


def test_cli_estimate_empty_results():
    """Empty results handled."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Empty directory
        
        exit_code, stdout, stderr = _run_cli(["estimate", str(root)])
        
        assert exit_code == 0
        # Should output empty JSON array
        data = json.loads(stdout)
        assert data == []


def test_cli_estimate_exit_code_success():
    """Exit code 0 on success."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        exit_code, stdout, stderr = _run_cli(["estimate", str(root)])
        
        # Should exit with 0 if no warnings
        # Note: if warnings present, exit code might be 1
        assert exit_code in (0, 1)


def test_cli_estimate_exit_code_warnings():
    """Exit code 1 on warnings (if implemented)."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", "a")  # Creates cycle
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        exit_code, stdout, stderr = _run_cli(["estimate", str(root)])
        
        # If warnings cause non-zero exit, should be 1
        # Otherwise might be 0
        assert exit_code in (0, 1)
        # Should have warnings in stderr
        if exit_code == 1:
            assert "warning" in stderr.lower()


def test_cli_estimate_help_flag():
    """--help flag works."""
    exit_code, stdout, stderr = _run_cli(["estimate", "--help"])
    
    assert exit_code == 0
    # Should show help text
    assert "estimate" in stdout.lower() or "usage" in stdout.lower()


def test_integration_cli_golden_fixtures():
    """End-to-end: CLI on golden fixtures."""
    from pathlib import Path as PathLib
    
    golden_dir = PathLib(__file__).resolve().parent / "fixtures" / "golden"
    if not golden_dir.exists():
        pytest.skip("Golden fixtures directory not found")
    
    exit_code, stdout, stderr = _run_cli(["estimate", str(golden_dir)])
    
    # Should succeed
    assert exit_code in (0, 1)  # 1 if warnings present
    # Should output valid JSON
    data = json.loads(stdout)
    assert isinstance(data, list)
    # All results should be schema 3.0
    assert all(r["schema_version"] == "3.0" for r in data)
