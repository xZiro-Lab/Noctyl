"""
Test Noctyl extraction and Mermaid generation on the example multi-agent system.

Validates that a complex, realistic multi-agent workflow can be extracted
and visualized correctly.
"""

from pathlib import Path

from noctyl_scout.graph import workflow_dict_to_mermaid
from noctyl_scout.ingestion import run_pipeline_on_directory

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "multi_agent_system"


def test_extract_multi_agent_system():
    """Extract workflows from example multi-agent system repo."""
    results, warnings = run_pipeline_on_directory(EXAMPLE_DIR)
    assert len(results) >= 3, "expected at least 3 graphs (workflow, parallel_workflow, simple_agent)"
    assert all("schema_version" in g for g in results)
    assert all("nodes" in g and "edges" in g for g in results)


def test_multi_agent_workflow_structure():
    """Main workflow.py should have coordinator, researcher, writer, reviewer nodes."""
    results, _ = run_pipeline_on_directory(EXAMPLE_DIR)
    # Find the workflow graph (has coordinator node)
    workflow_graph = None
    for g in results:
        node_names = {n["name"] for n in g["nodes"]}
        if "coordinator" in node_names and "researcher" in node_names:
            workflow_graph = g
            break
    assert workflow_graph is not None, "should find workflow with coordinator and researcher"
    node_names = {n["name"] for n in workflow_graph["nodes"]}
    assert "coordinator" in node_names
    assert "researcher" in node_names
    assert "writer" in node_names
    assert "reviewer" in node_names
    assert workflow_graph["entry_point"] == "coordinator"
    assert len(workflow_graph["conditional_edges"]) >= 2  # task_router and quality_check


def test_parallel_workflow_structure():
    """parallel_workflow.py should have parallel paths and conditional edges."""
    results, _ = run_pipeline_on_directory(EXAMPLE_DIR)
    parallel_graph = None
    for g in results:
        node_names = {n["name"] for n in g["nodes"]}
        if "agent_a" in node_names and "agent_b" in node_names and "merger" in node_names:
            parallel_graph = g
            break
    assert parallel_graph is not None
    node_names = {n["name"] for n in parallel_graph["nodes"]}
    assert "agent_a" in node_names
    assert "agent_b" in node_names
    assert "agent_c" in node_names
    assert "merger" in node_names
    # Should have conditional edges from agent_a and agent_b
    cond_sources = {e["source"] for e in parallel_graph["conditional_edges"]}
    assert "agent_a" in cond_sources
    assert "agent_b" in cond_sources


def test_generate_mermaid_for_multi_agent():
    """Generate Mermaid diagrams for all workflows in example repo."""
    results, _ = run_pipeline_on_directory(EXAMPLE_DIR)
    for d in results:
        mermaid = workflow_dict_to_mermaid(d)
        assert mermaid.startswith("flowchart TB")
        assert "Start" in mermaid and "EndNode" in mermaid
        node_names = {n["name"] for n in d["nodes"]}
        for name in node_names:
            assert name in mermaid, f"node {name} should appear in Mermaid"
        # Should have edges
        assert "-->" in mermaid


def test_complex_workflow_has_loops():
    """Main workflow should show loop (reject -> coordinator)."""
    results, _ = run_pipeline_on_directory(EXAMPLE_DIR)
    workflow_graph = None
    for g in results:
        if "coordinator" in {n["name"] for n in g["nodes"]}:
            workflow_graph = g
            break
    assert workflow_graph is not None
    # Reviewer should have conditional edge back to coordinator (loop)
    cond_edges = workflow_graph["conditional_edges"]
    reviewer_edges = [e for e in cond_edges if e["source"] == "reviewer"]
    assert any(e["target"] == "coordinator" for e in reviewer_edges), "reviewer should loop back to coordinator on reject"
