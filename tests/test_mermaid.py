"""Tests for Mermaid diagram generation from workflow dict."""

from noctyl.graph import workflow_dict_to_mermaid, workflow_graph_to_dict
from noctyl.graph import (
    ExtractedConditionalEdge,
    ExtractedEdge,
    ExtractedNode,
    build_workflow_graph,
)


def test_workflow_dict_to_mermaid_linear():
    """Linear workflow produces Mermaid with Start, EndNode, nodes, and edges."""
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [
        ExtractedEdge("START", "a", 3),
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("b", "END", 5),
    ]
    g = build_workflow_graph("id:0", nodes, edges, [], "a")
    d = workflow_graph_to_dict(g)
    m = workflow_dict_to_mermaid(d)
    assert "flowchart TB" in m
    assert "Start" in m
    assert "EndNode" in m
    assert '["a"]' in m or "a[" in m
    assert '["b"]' in m or "b[" in m
    assert "Start -->" in m
    assert "--> EndNode" in m
    assert "a --> b" in m


def test_workflow_dict_to_mermaid_conditional():
    """Conditional edge appears with label in Mermaid."""
    nodes = [ExtractedNode("a", "f", 1)]
    edges = [ExtractedEdge("START", "a", 2)]
    cond = [ExtractedConditionalEdge("a", "done", "END", 3)]
    g = build_workflow_graph("id:0", nodes, edges, cond, "a")
    d = workflow_graph_to_dict(g)
    m = workflow_dict_to_mermaid(d)
    assert "-->|done|" in m or "-->|" in m
    assert "EndNode" in m


def test_workflow_dict_to_mermaid_empty():
    """Empty workflow dict produces minimal valid Mermaid."""
    d = {"nodes": [], "edges": [], "conditional_edges": []}
    m = workflow_dict_to_mermaid(d)
    assert "flowchart TB" in m
    assert "Start" in m
    assert "EndNode" in m


def test_workflow_dict_to_mermaid_from_golden_style():
    """Round-trip: build dict then Mermaid; graph of agents is present."""
    d = {
        "schema_version": "1.0",
        "graph_id": "x:0",
        "entry_point": "agent",
        "terminal_nodes": ["tool"],
        "nodes": [
            {"name": "agent", "callable_ref": "fa", "line": 1},
            {"name": "tool", "callable_ref": "fb", "line": 2},
        ],
        "edges": [
            {"source": "START", "target": "agent", "line": 3},
            {"source": "agent", "target": "tool", "line": 4},
            {"source": "tool", "target": "END", "line": 5},
        ],
        "conditional_edges": [],
    }
    m = workflow_dict_to_mermaid(d)
    assert "agent" in m and "tool" in m
    assert "Start -->" in m
    assert "--> EndNode" in m
