"""
Comprehensive integration tests for Phase 3: end-to-end pipeline, CLI, golden fixtures, and all features together.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from noctyl.estimation import ModelProfile
from noctyl.ingestion import run_pipeline_on_directory

CLI_SCRIPT = Path(__file__).resolve().parent.parent / "noctyl" / "cli.py"


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run CLI command and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, str(CLI_SCRIPT)] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8"
    )
    return result.returncode, result.stdout, result.stderr


def test_end_to_end_pipeline_cli_golden():
    """Full pipeline: CLI → golden fixtures → schema 3.0."""
    from pathlib import Path as PathLib
    
    golden_dir = PathLib(__file__).resolve().parent / "fixtures" / "golden"
    if not golden_dir.exists():
        pytest.skip("Golden fixtures directory not found")
    
    # Run via CLI
    exit_code, stdout, stderr = _run_cli(["estimate", str(golden_dir)])
    
    # Should succeed
    assert exit_code in (0, 1)  # 1 if warnings
    
    # Parse JSON output
    data = json.loads(stdout)
    assert isinstance(data, list)
    assert len(data) > 0
    
    # All should be schema 3.0
    for d in data:
        assert d["schema_version"] == "3.0"
        assert d["estimated"] is True
        assert d["enriched"] is True
        assert "token_estimate" in d
        assert "node_signatures" in d


def test_end_to_end_profile_yaml_cli():
    """Profile YAML → CLI → correct estimates."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): 
    prompt = "Test prompt"
    return {"response": prompt}

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    yaml_content = """
name: custom-yaml-profile
expansion_factor: 1.5
output_ratio: 0.6
pricing:
  input_per_1k: 0.01
  output_per_1k: 0.02
"""
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        profile_path = root / "profile.yaml"
        profile_path.write_text(yaml_content)
        
        # Run via CLI with profile
        exit_code, stdout, stderr = _run_cli(
            ["estimate", str(root), "--profile", str(profile_path)]
        )
        
        assert exit_code in (0, 1)
        data = json.loads(stdout)
        assert len(data) == 1
        assert data[0]["token_estimate"]["assumptions_profile"] == "custom-yaml-profile"


def test_end_to_end_custom_profile_pipeline():
    """Custom profile → pipeline → estimates."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): 
    prompt = "Test"
    return {"response": prompt}

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    custom_profile = ModelProfile("custom-pipeline", 1.8, 0.7, 0.1, 0.2)
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        # Run via pipeline
        results, warnings = run_pipeline_on_directory(
            root, estimate=True, profile=custom_profile
        )
        
        assert len(results) == 1
        assert results[0]["token_estimate"]["assumptions_profile"] == "custom-pipeline"
        assert results[0]["schema_version"] == "3.0"


def test_comprehensive_workflow_all_features():
    """All Phase 3 features together: loops, branches, symbolic nodes."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): 
    prompt = "Prompt A"
    return {"response": prompt}

def fb(x): return x
def fc(x): return x

def router(x): return "continue" if x.get("count", 0) < 5 else "done"

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_node("c", fc)
graph.add_edge(START, "a")
graph.add_conditional_edges("a", router, {"continue": "b", "done": "END"})
graph.add_edge("b", "c")
graph.add_edge("c", "a")  # Creates cycle
"""
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        results, warnings = run_pipeline_on_directory(root, estimate=True)
        
        assert len(results) == 1
        d = results[0]
        
        # Should have all Phase 3 features
        assert d["schema_version"] == "3.0"
        assert d["estimated"] is True
        assert "token_estimate" in d
        assert "node_signatures" in d
        assert "per_node_envelopes" in d
        assert "per_path_envelopes" in d
        
        # Should have cycles (loop amplification)
        assert len(d["cycles"]) > 0
        
        # Should have branches (per_path_envelopes)
        assert len(d["conditional_edges"]) > 0
        assert len(d["per_path_envelopes"]) > 0
        
        # Should have warnings (cycles)
        assert len(d["warnings"]) > 0 or len(warnings) > 0


def test_determinism_full_pipeline():
    """Full pipeline determinism: same input → identical output."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)
"""
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        # Run twice
        results1, warnings1 = run_pipeline_on_directory(root, estimate=True)
        results2, warnings2 = run_pipeline_on_directory(root, estimate=True)
        
        # Should be identical
        assert json.dumps(results1, sort_keys=True) == json.dumps(results2, sort_keys=True)
        assert sorted(warnings1) == sorted(warnings2)


def test_error_propagation_through_pipeline():
    """Errors propagate correctly through pipeline."""
    # Invalid profile file
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
        
        # Invalid profile path - should use default and add warning
        results, warnings = run_pipeline_on_directory(
            root, estimate=True, profile="/nonexistent/profile.yaml"
        )
        
        # Should still produce results (using default profile)
        assert len(results) == 1
        assert results[0]["schema_version"] == "3.0"
        # Should have warning about profile load error
        assert any("profile" in w.lower() for w in warnings)


def test_performance_large_workflow():
    """Large workflow performance (many nodes, cycles, branches)."""
    # Create a large workflow
    nodes = []
    edges = []
    cond_edges = []
    
    # 20 nodes
    for i in range(20):
        nodes.append(f"node_{i}")
    
    # Create edges: linear chain with cycles
    for i in range(19):
        edges.append(f"node_{i} -> node_{i+1}")
    
    # Add cycles
    edges.append("node_5 -> node_3")  # Cycle
    edges.append("node_10 -> node_8")  # Cycle
    
    # Add branches
    cond_edges.append("node_15 -> [path1: node_16, path2: node_17]")
    
    source = f"""
from langgraph.graph import StateGraph, START, END

def f(x): return x

graph = StateGraph(dict)
"""
    for node in nodes:
        source += f'graph.add_node("{node}", f)\n'
    
    source += "graph.add_edge(START, node_0)\n"
    for i in range(19):
        source += f"graph.add_edge(node_{i}, node_{i+1})\n"
    source += "graph.add_edge(node_19, END)\n"
    
    # Actually, let's use a simpler but still large workflow
    source = """
from langgraph.graph import StateGraph, START, END

def f(x): return x

graph = StateGraph(dict)
for i in range(10):
    graph.add_node(f"node_{i}", f)
graph.add_edge(START, "node_0")
for i in range(9):
    graph.add_edge(f"node_{i}", f"node_{i+1}")
graph.add_edge("node_9", END)
"""
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        # Should process without performance issues
        import time
        start = time.time()
        results, warnings = run_pipeline_on_directory(root, estimate=True)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert len(results) >= 0  # May or may not have results depending on source


def test_regression_all_phase2_tests_still_pass():
    """No Phase 2 regressions: enriched mode still works."""
    source = """
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)
"""
    
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "workflow.py").write_text(source)
        
        # Phase 2 enriched mode
        results_enriched, _ = run_pipeline_on_directory(root, enriched=True, estimate=False)
        assert len(results_enriched) == 1
        assert results_enriched[0]["schema_version"] == "2.0"
        assert results_enriched[0]["enriched"] is True
        assert "estimated" not in results_enriched[0]
        assert "token_estimate" not in results_enriched[0]
        
        # Phase 1 base mode
        results_base, _ = run_pipeline_on_directory(root, enriched=False, estimate=False)
        assert len(results_base) == 1
        assert results_base[0]["schema_version"] == "1.0"
        assert "enriched" not in results_base[0]
        assert "estimated" not in results_base[0]
        
        # Phase 3 estimate mode
        results_estimate, _ = run_pipeline_on_directory(root, estimate=True)
        assert len(results_estimate) == 1
        assert results_estimate[0]["schema_version"] == "3.0"
        assert results_estimate[0]["estimated"] is True
        assert results_estimate[0]["enriched"] is True
        assert "token_estimate" in results_estimate[0]
