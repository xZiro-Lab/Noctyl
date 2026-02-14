"""Tests for Phase 2 ExecutionModel and execution_model_to_dict."""

import json

import pytest

from noctyl.graph import (
    DetectedCycle,
    ExecutionModel,
    ExtractedConditionalEdge,
    ExtractedEdge,
    ExtractedNode,
    NodeAnnotation,
    StructuralMetrics,
    StructuralRisk,
    WorkflowGraph,
    build_workflow_graph,
    execution_model_to_dict,
)


def _minimal_workflow_graph() -> WorkflowGraph:
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [ExtractedEdge("a", "b", 3)]
    return build_workflow_graph("file.py:0", nodes, edges, [], "a")


def _minimal_execution_model() -> ExecutionModel:
    wg = _minimal_workflow_graph()
    metrics = StructuralMetrics(
        node_count=2,
        edge_count=1,
        entry_node="a",
        terminal_nodes=(),
        unreachable_nodes=(),
        longest_acyclic_path=2,
        avg_branching_factor=1.0,
        max_depth_before_cycle=None,
    )
    risks = StructuralRisk(
        unreachable_node_ids=(),
        dead_end_ids=(),
        non_terminating_cycle_ids=(),
        multiple_entry_points=False,
    )
    return ExecutionModel(
        graph=wg,
        entry_point=wg.entry_point,
        terminal_nodes=wg.terminal_nodes,
        shape="linear",
        cycles=(),
        metrics=metrics,
        node_annotations=(),
        risks=risks,
    )


def test_execution_model_to_dict_structure():
    """Serialized dict has schema_version 2.0, enriched True, and all enriched keys."""
    model = _minimal_execution_model()
    d = execution_model_to_dict(model)
    assert d["schema_version"] == "2.0"
    assert d["enriched"] is True
    assert d["shape"] == "linear"
    assert "cycles" in d and isinstance(d["cycles"], list)
    assert "metrics" in d and isinstance(d["metrics"], dict)
    assert "node_annotations" in d and isinstance(d["node_annotations"], list)
    assert "risks" in d and isinstance(d["risks"], dict)


def test_execution_model_base_graph_preserved():
    """Base graph fields from workflow_graph_to_dict are present and correct."""
    model = _minimal_execution_model()
    d = execution_model_to_dict(model)
    assert d["graph_id"] == "file.py:0"
    assert d["entry_point"] == "a"
    assert d["terminal_nodes"] == []
    assert len(d["nodes"]) == 2
    assert [n["name"] for n in d["nodes"]] == ["a", "b"]
    assert len(d["edges"]) == 1
    assert d["edges"][0]["source"] == "a" and d["edges"][0]["target"] == "b"
    assert d["conditional_edges"] == []


def test_execution_model_deterministic_serialization():
    """Same ExecutionModel -> same JSON (serialize twice, assert equal)."""
    model = _minimal_execution_model()
    d1 = execution_model_to_dict(model)
    d2 = execution_model_to_dict(model)
    assert d1 == d2
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_cycles_sorted_by_cycle_type_then_nodes():
    """Cycles in output are sorted by (cycle_type, nodes)."""
    wg = _minimal_workflow_graph()
    metrics = StructuralMetrics(
        node_count=2, edge_count=1, entry_node="a", terminal_nodes=(),
        unreachable_nodes=(), longest_acyclic_path=2, avg_branching_factor=1.0,
        max_depth_before_cycle=None,
    )
    risks = StructuralRisk((), (), (), False)
    cycles = (
        DetectedCycle("multi_node", ("b", "a"), True),
        DetectedCycle("self_loop", ("a",), False),
    )
    model = ExecutionModel(
        graph=wg, entry_point="a", terminal_nodes=(), shape="cyclic",
        cycles=cycles, metrics=metrics, node_annotations=(), risks=risks,
    )
    d = execution_model_to_dict(model)
    cycle_types = [c["cycle_type"] for c in d["cycles"]]
    assert cycle_types == ["multi_node", "self_loop"]  # sorted by (cycle_type, nodes)
    assert d["cycles"][0]["nodes"] == ["b", "a"]  # order preserved as in tuple
    assert d["cycles"][0]["reaches_terminal"] is True
    assert d["cycles"][1]["nodes"] == ["a"]
    assert d["cycles"][1]["reaches_terminal"] is False


def test_node_annotations_sorted_by_node_name():
    """Node annotations in output are sorted by node_name."""
    wg = _minimal_workflow_graph()
    metrics = StructuralMetrics(
        node_count=2, edge_count=1, entry_node="a", terminal_nodes=(),
        unreachable_nodes=(), longest_acyclic_path=2, avg_branching_factor=1.0,
        max_depth_before_cycle=None,
    )
    risks = StructuralRisk((), (), (), False)
    annotations = (
        NodeAnnotation("b", "imported_function", "read_only", "tool_like"),
        NodeAnnotation("a", "local_function", "mutates_state", "llm_like"),
    )
    model = ExecutionModel(
        graph=wg, entry_point="a", terminal_nodes=(), shape="linear",
        cycles=(), metrics=metrics, node_annotations=annotations, risks=risks,
    )
    d = execution_model_to_dict(model)
    names = [a["node_name"] for a in d["node_annotations"]]
    assert names == ["a", "b"]
    assert d["node_annotations"][0]["origin"] == "local_function"
    assert d["node_annotations"][0]["state_interaction"] == "mutates_state"
    assert d["node_annotations"][0]["role"] == "llm_like"
    assert d["node_annotations"][1]["origin"] == "imported_function"
    assert d["node_annotations"][1]["role"] == "tool_like"


def test_metrics_and_risks_serialized():
    """Metrics and risks dicts have expected keys and JSON-serializable values."""
    wg = _minimal_workflow_graph()
    metrics = StructuralMetrics(
        node_count=3,
        edge_count=4,
        entry_node="start",
        terminal_nodes=("end",),
        unreachable_nodes=("orphan",),
        longest_acyclic_path=5,
        avg_branching_factor=1.33,
        max_depth_before_cycle=2,
    )
    risks = StructuralRisk(
        unreachable_node_ids=("orphan",),
        dead_end_ids=("dead",),
        non_terminating_cycle_ids=("c0",),
        multiple_entry_points=True,
    )
    model = ExecutionModel(
        graph=wg, entry_point="start", terminal_nodes=("end",), shape="branching",
        cycles=(), metrics=metrics, node_annotations=(), risks=risks,
    )
    d = execution_model_to_dict(model)
    m = d["metrics"]
    assert m["node_count"] == 3 and m["edge_count"] == 4
    assert m["entry_node"] == "start"
    assert m["terminal_nodes"] == ["end"]
    assert m["unreachable_nodes"] == ["orphan"]
    assert m["longest_acyclic_path"] == 5
    assert m["avg_branching_factor"] == 1.33
    assert m["max_depth_before_cycle"] == 2
    r = d["risks"]
    assert r["unreachable_node_ids"] == ["orphan"]
    assert r["dead_end_ids"] == ["dead"]
    assert r["non_terminating_cycle_ids"] == ["c0"]
    assert r["multiple_entry_points"] is True


def test_no_token_or_cost_fields():
    """Output dict does not contain token or cost related keys."""
    model = _minimal_execution_model()
    d = execution_model_to_dict(model)
    keys_lower = [k.lower() for k in d.keys()]
    assert "token" not in " ".join(keys_lower)
    assert "cost" not in " ".join(keys_lower)


def test_empty_execution_model_json_roundtrip():
    """Minimal ExecutionModel serializes to valid JSON and roundtrips."""
    model = _minimal_execution_model()
    d = execution_model_to_dict(model)
    s = json.dumps(d, sort_keys=True)
    parsed = json.loads(s)
    assert parsed["schema_version"] == "2.0"
    assert parsed["enriched"] is True
    assert parsed["cycles"] == []
    assert parsed["node_annotations"] == []


def test_execution_model_with_conditional_edges():
    """ExecutionModel with conditional edges preserves them in base output."""
    nodes = [
        ExtractedNode("n", "f", 1),
    ]
    edges = []
    cond = [ExtractedConditionalEdge("n", "yes", "END", 2)]
    wg = build_workflow_graph("id:0", nodes, edges, cond, "n")
    metrics = StructuralMetrics(
        node_count=1, edge_count=0, entry_node="n", terminal_nodes=("n",),
        unreachable_nodes=(), longest_acyclic_path=1, avg_branching_factor=1.0,
        max_depth_before_cycle=None,
    )
    risks = StructuralRisk((), (), (), False)
    model = ExecutionModel(
        graph=wg, entry_point="n", terminal_nodes=("n",), shape="linear",
        cycles=(), metrics=metrics, node_annotations=(), risks=risks,
    )
    d = execution_model_to_dict(model)
    assert d["terminal_nodes"] == ["n"]
    assert len(d["conditional_edges"]) == 1
    assert d["conditional_edges"][0]["condition_label"] == "yes"
    assert d["conditional_edges"][0]["target"] == "END"


def test_schema_version_is_forced_to_v2():
    """Phase 2 serializer always emits schema_version 2.0."""
    g = WorkflowGraph(
        graph_id="x:0",
        nodes=(ExtractedNode("a", "f", 1),),
        edges=(),
        conditional_edges=(),
        entry_point="a",
        terminal_nodes=(),
        schema_version="1.0",
    )
    metrics = StructuralMetrics(
        node_count=1,
        edge_count=0,
        entry_node="a",
        terminal_nodes=(),
        unreachable_nodes=(),
        longest_acyclic_path=1,
        avg_branching_factor=0.0,
        max_depth_before_cycle=None,
    )
    model = ExecutionModel(
        graph=g,
        entry_point="a",
        terminal_nodes=(),
        shape="linear",
        cycles=(),
        metrics=metrics,
        node_annotations=(),
        risks=StructuralRisk((), (), (), False),
    )
    d = execution_model_to_dict(model)
    assert d["schema_version"] == "2.0"


def test_cycle_sorting_tie_breaks_on_nodes_tuple():
    """When cycle_type is equal, tuple order breaks ties deterministically."""
    wg = _minimal_workflow_graph()
    metrics = StructuralMetrics(
        node_count=2,
        edge_count=1,
        entry_node="a",
        terminal_nodes=(),
        unreachable_nodes=(),
        longest_acyclic_path=2,
        avg_branching_factor=1.0,
        max_depth_before_cycle=None,
    )
    model = ExecutionModel(
        graph=wg,
        entry_point="a",
        terminal_nodes=(),
        shape="cyclic",
        cycles=(
            DetectedCycle("multi_node", ("z", "a"), True),
            DetectedCycle("multi_node", ("a", "z"), True),
        ),
        metrics=metrics,
        node_annotations=(),
        risks=StructuralRisk((), (), (), False),
    )
    d = execution_model_to_dict(model)
    assert d["cycles"][0]["nodes"] == ["a", "z"]
    assert d["cycles"][1]["nodes"] == ["z", "a"]


@pytest.mark.parametrize(
    ("origin", "state_interaction", "role"),
    [
        ("local_function", "pure", "unknown"),
        ("imported_function", "read_only", "tool_like"),
        ("class_method", "mutates_state", "control_node"),
        ("lambda", "pure", "llm_like"),
        ("unknown", "read_only", "unknown"),
    ],
)
def test_node_annotation_literal_values_roundtrip(origin, state_interaction, role):
    """Serializer preserves allowed node annotation literal values."""
    wg = _minimal_workflow_graph()
    metrics = StructuralMetrics(
        node_count=2,
        edge_count=1,
        entry_node="a",
        terminal_nodes=(),
        unreachable_nodes=(),
        longest_acyclic_path=2,
        avg_branching_factor=1.0,
        max_depth_before_cycle=None,
    )
    model = ExecutionModel(
        graph=wg,
        entry_point="a",
        terminal_nodes=(),
        shape="linear",
        cycles=(),
        metrics=metrics,
        node_annotations=(NodeAnnotation("a", origin, state_interaction, role),),
        risks=StructuralRisk((), (), (), False),
    )
    d = execution_model_to_dict(model)
    assert d["node_annotations"] == [
        {
            "node_name": "a",
            "origin": origin,
            "state_interaction": state_interaction,
            "role": role,
        }
    ]


@pytest.mark.parametrize(
    "shape",
    ["linear", "branching", "cyclic", "disconnected", "invalid"],
)
def test_shape_values_are_serialized(shape):
    """All expected shape labels are serialized as-is."""
    model = _minimal_execution_model()
    shaped = ExecutionModel(
        graph=model.graph,
        entry_point=model.entry_point,
        terminal_nodes=model.terminal_nodes,
        shape=shape,
        cycles=model.cycles,
        metrics=model.metrics,
        node_annotations=model.node_annotations,
        risks=model.risks,
    )
    d = execution_model_to_dict(shaped)
    assert d["shape"] == shape
