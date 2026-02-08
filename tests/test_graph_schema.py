"""Tests for WorkflowGraph schema and deterministic serialization."""

import json

from noctyl.graph import (
    ExtractedConditionalEdge,
    ExtractedEdge,
    ExtractedNode,
    WorkflowGraph,
    build_workflow_graph,
    workflow_graph_to_dict,
)


def test_workflow_graph_to_dict_structure():
    """Serialized dict has schema_version, graph_id, entry_point, nodes, edges, conditional_edges."""
    g = WorkflowGraph(
        graph_id="file.py:0",
        nodes=(ExtractedNode("a", "f", 10),),
        edges=(ExtractedEdge("a", "b", 11),),
        conditional_edges=(),
        entry_point="a",
    )
    d = workflow_graph_to_dict(g)
    assert d["schema_version"] == "1.0"
    assert d["graph_id"] == "file.py:0"
    assert d["entry_point"] == "a"
    assert "nodes" in d and isinstance(d["nodes"], list)
    assert "edges" in d and isinstance(d["edges"], list)
    assert "conditional_edges" in d and isinstance(d["conditional_edges"], list)
    assert len(d["nodes"]) == 1
    assert d["nodes"][0] == {"name": "a", "callable_ref": "f", "line": 10}
    assert len(d["edges"]) == 1
    assert d["edges"][0] == {"source": "a", "target": "b", "line": 11}


def test_deterministic_serialization():
    """Same WorkflowGraph -> same JSON (serialize twice, assert equal)."""
    nodes = [
        ExtractedNode("b", "fb", 2),
        ExtractedNode("a", "fa", 1),
    ]
    edges = [
        ExtractedEdge("a", "b", 3),
    ]
    cond_edges = [
        ExtractedConditionalEdge("b", "x", "a", 4),
    ]
    g = WorkflowGraph(
        graph_id="x:0",
        nodes=tuple(nodes),
        edges=tuple(edges),
        conditional_edges=tuple(cond_edges),
        entry_point="a",
    )
    d1 = workflow_graph_to_dict(g)
    d2 = workflow_graph_to_dict(g)
    json1 = json.dumps(d1, sort_keys=True)
    json2 = json.dumps(d2, sort_keys=True)
    assert json1 == json2
    assert d1 == d2


def test_nodes_sorted_by_name_then_line():
    """Nodes in serialized output are sorted by (name, line)."""
    g = WorkflowGraph(
        graph_id="x:0",
        nodes=(
            ExtractedNode("z", "fz", 1),
            ExtractedNode("a", "fa", 2),
            ExtractedNode("a", "fa2", 1),
        ),
        edges=(),
        conditional_edges=(),
        entry_point=None,
    )
    d = workflow_graph_to_dict(g)
    names = [n["name"] for n in d["nodes"]]
    assert names == ["a", "a", "z"]
    assert d["nodes"][0]["line"] == 1 and d["nodes"][1]["line"] == 2


def test_edges_sorted_by_source_target_line():
    """Edges in serialized output are sorted by (source, target, line)."""
    g = WorkflowGraph(
        graph_id="x:0",
        nodes=(),
        edges=(
            ExtractedEdge("b", "c", 2),
            ExtractedEdge("a", "b", 1),
        ),
        conditional_edges=(),
        entry_point=None,
    )
    d = workflow_graph_to_dict(g)
    assert d["edges"][0]["source"] == "a"
    assert d["edges"][1]["source"] == "b"


def test_conditional_edges_sorted():
    """Conditional edges in serialized output are sorted."""
    g = WorkflowGraph(
        graph_id="x:0",
        nodes=(),
        edges=(),
        conditional_edges=(
            ExtractedConditionalEdge("n", "y", "b", 2),
            ExtractedConditionalEdge("n", "x", "a", 1),
        ),
        entry_point=None,
    )
    d = workflow_graph_to_dict(g)
    labels = [e["condition_label"] for e in d["conditional_edges"]]
    assert labels == ["x", "y"]


def test_build_workflow_graph():
    """build_workflow_graph produces WorkflowGraph from lists."""
    nodes = [ExtractedNode("a", "f", 1)]
    edges = [ExtractedEdge("a", "b", 2)]
    cond = [ExtractedConditionalEdge("b", "ok", "a", 3)]
    g = build_workflow_graph("id:0", nodes, edges, cond, "a")
    assert g.graph_id == "id:0"
    assert g.entry_point == "a"
    assert len(g.nodes) == 1 and g.nodes[0].name == "a"
    assert len(g.edges) == 1 and g.edges[0].target == "b"
    assert len(g.conditional_edges) == 1 and g.conditional_edges[0].condition_label == "ok"


def test_entry_point_null_serialized():
    """entry_point None -> JSON null."""
    g = WorkflowGraph(
        graph_id="x:0",
        nodes=(),
        edges=(),
        conditional_edges=(),
        entry_point=None,
    )
    d = workflow_graph_to_dict(g)
    assert d["entry_point"] is None
    s = json.dumps(d, sort_keys=True)
    assert '"entry_point": null' in s


def test_empty_graph_roundtrip():
    """Empty graph serializes to valid JSON with empty arrays."""
    g = WorkflowGraph(
        graph_id="empty:0",
        nodes=(),
        edges=(),
        conditional_edges=(),
        entry_point=None,
    )
    d = workflow_graph_to_dict(g)
    assert d["nodes"] == []
    assert d["edges"] == []
    assert d["conditional_edges"] == []
    parsed = json.loads(json.dumps(d, sort_keys=True))
    assert parsed == d
