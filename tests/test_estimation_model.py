"""Tests for Phase 3 estimation data model and workflow_estimate_to_dict."""

import json

import pytest

from noctyl.estimation import (
    CostEnvelope,
    ESTIMATED_SCHEMA_VERSION,
    ModelProfile,
    NodeTokenSignature,
    WorkflowEstimate,
    workflow_estimate_to_dict,
)
from noctyl.graph import (
    ExecutionModel,
    ExtractedEdge,
    ExtractedNode,
    StructuralMetrics,
    StructuralRisk,
    build_workflow_graph,
    execution_model_to_dict,
)


def _minimal_workflow_graph():
    """Helper: minimal WorkflowGraph for testing."""
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [ExtractedEdge("a", "b", 3)]
    return build_workflow_graph("file.py:0", nodes, edges, [], "a")


def _minimal_execution_model():
    """Helper: minimal ExecutionModel for testing."""
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
    from noctyl.graph import ExecutionModel

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


def _minimal_workflow_estimate():
    """Helper: minimal WorkflowEstimate for testing."""
    em = _minimal_execution_model()
    sigs = (
        NodeTokenSignature("a", 100, 1.2, True, False),
        NodeTokenSignature("b", 150, 1.3, False, False),
    )
    envelope = CostEnvelope(250, 300, 350, True, "structural-static")
    return WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="default",
        per_node_envelopes={
            "a": CostEnvelope(100, 120, 140, True, "structural-static"),
            "b": CostEnvelope(150, 195, 210, True, "structural-static"),
        },
        per_path_envelopes={},
        warnings=(),
    )


# ── Structure and schema version ────────────────────────────────────────────


def test_workflow_estimate_to_dict_structure():
    """Serialized dict has schema_version 3.0, estimated True, enriched True."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    assert d["schema_version"] == "3.0"
    assert d["estimated"] is True
    assert d["enriched"] is True
    assert "token_estimate" in d and isinstance(d["token_estimate"], dict)
    assert "node_signatures" in d and isinstance(d["node_signatures"], list)
    assert "per_node_envelopes" in d and isinstance(d["per_node_envelopes"], dict)
    assert "per_path_envelopes" in d and isinstance(d["per_path_envelopes"], dict)
    assert "warnings" in d and isinstance(d["warnings"], list)


def test_phase2_data_preserved():
    """Phase 2 enriched fields are preserved in Phase 3 output."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    # Phase 2 fields should be present
    assert d["graph_id"] == "file.py:0"
    assert d["entry_point"] == "a"
    assert d["shape"] == "linear"
    assert "cycles" in d
    assert "metrics" in d
    assert "node_annotations" in d
    assert "risks" in d
    # Base graph structure
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1


def test_token_estimate_structure():
    """Token estimate dict has all required fields."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    te = d["token_estimate"]
    assert te["assumptions_profile"] == "default"
    assert te["min_tokens"] == 250
    assert te["expected_tokens"] == 300
    assert te["max_tokens"] == 350
    assert te["bounded"] is True
    assert te["confidence"] == "structural-static"


# ── Deterministic serialization ──────────────────────────────────────────────


def test_workflow_estimate_deterministic_serialization():
    """Same WorkflowEstimate -> same JSON (serialize twice, assert equal)."""
    estimate = _minimal_workflow_estimate()
    d1 = workflow_estimate_to_dict(estimate)
    d2 = workflow_estimate_to_dict(estimate)
    assert d1 == d2
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_node_signatures_sorted_by_node_name():
    """Node signatures in output are sorted by node_name."""
    em = _minimal_execution_model()
    sigs = (
        NodeTokenSignature("z", 200, 1.5, False, True),
        NodeTokenSignature("a", 100, 1.2, True, False),
        NodeTokenSignature("m", 150, 1.3, True, False),
    )
    envelope = CostEnvelope(450, 500, 550, True, "structural-static")
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    names = [s["node_name"] for s in d["node_signatures"]]
    assert names == ["a", "m", "z"]
    assert d["node_signatures"][0]["base_prompt_tokens"] == 100
    assert d["node_signatures"][1]["base_prompt_tokens"] == 150
    assert d["node_signatures"][2]["base_prompt_tokens"] == 200


def test_per_node_envelopes_sorted_keys():
    """Per-node envelopes dict keys are sorted."""
    em = _minimal_execution_model()
    sigs = (
        NodeTokenSignature("a", 100, 1.2, True, False),
        NodeTokenSignature("b", 150, 1.3, False, False),
    )
    envelope = CostEnvelope(250, 300, 350, True, "structural-static")
    per_node = {
        "z": CostEnvelope(200, 250, 300, True, "structural-static"),
        "a": CostEnvelope(100, 120, 140, True, "structural-static"),
        "m": CostEnvelope(150, 180, 200, True, "structural-static"),
    }
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes=per_node,
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    keys = list(d["per_node_envelopes"].keys())
    assert keys == ["a", "m", "z"]
    assert d["per_node_envelopes"]["a"]["min_tokens"] == 100
    assert d["per_node_envelopes"]["z"]["min_tokens"] == 200


def test_per_path_envelopes_sorted_keys():
    """Per-path envelopes dict keys are sorted."""
    em = _minimal_execution_model()
    sigs = (NodeTokenSignature("a", 100, 1.2, True, False),)
    envelope = CostEnvelope(100, 120, 140, True, "structural-static")
    per_path = {
        "path_z": CostEnvelope(200, 250, 300, True, "structural-static"),
        "path_a": CostEnvelope(100, 120, 140, True, "structural-static"),
    }
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes=per_path,
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    keys = list(d["per_path_envelopes"].keys())
    assert keys == ["path_a", "path_z"]


def test_warnings_sorted():
    """Warnings list is sorted alphabetically."""
    em = _minimal_execution_model()
    sigs = (NodeTokenSignature("a", 100, 1.2, True, False),)
    envelope = CostEnvelope(100, 120, 140, True, "structural-static")
    warnings = ("zebra warning", "alpha warning", "beta warning")
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=warnings,
    )
    d = workflow_estimate_to_dict(estimate)
    assert d["warnings"] == ["alpha warning", "beta warning", "zebra warning"]


# ── CostEnvelope invariant validation ────────────────────────────────────────


def test_cost_envelope_valid():
    """Valid CostEnvelope (min <= expected <= max) constructs successfully."""
    e = CostEnvelope(100, 150, 200, True, "structural-static")
    assert e.min_tokens == 100
    assert e.expected_tokens == 150
    assert e.max_tokens == 200


def test_cost_envelope_invariant_violation_min_gt_expected():
    """CostEnvelope raises ValueError when min > expected."""
    with pytest.raises(ValueError, match="CostEnvelope invariant violated"):
        CostEnvelope(200, 150, 100, True, "structural-static")


def test_cost_envelope_invariant_violation_expected_gt_max():
    """CostEnvelope raises ValueError when expected > max."""
    with pytest.raises(ValueError, match="CostEnvelope invariant violated"):
        CostEnvelope(100, 250, 200, True, "structural-static")


def test_cost_envelope_equal_bounds():
    """CostEnvelope with equal min/expected/max is valid."""
    e = CostEnvelope(100, 100, 100, True, "structural-static")
    assert e.min_tokens == e.expected_tokens == e.max_tokens == 100


# ── Frozen dataclass immutability ────────────────────────────────────────────


def test_node_token_signature_frozen():
    """NodeTokenSignature is immutable."""
    sig = NodeTokenSignature("a", 100, 1.2, True, False)
    with pytest.raises(AttributeError):
        sig.node_name = "b"


def test_model_profile_frozen():
    """ModelProfile is immutable."""
    profile = ModelProfile("gpt-4o", 1.4, 0.6, 0.005, 0.015)
    with pytest.raises(AttributeError):
        profile.name = "default"


def test_cost_envelope_frozen():
    """CostEnvelope is immutable."""
    e = CostEnvelope(100, 150, 200, True, "structural-static")
    with pytest.raises(AttributeError):
        e.min_tokens = 200


def test_workflow_estimate_frozen():
    """WorkflowEstimate is immutable."""
    estimate = _minimal_workflow_estimate()
    with pytest.raises(AttributeError):
        estimate.graph_id = "new_id"


# ── JSON roundtrip ──────────────────────────────────────────────────────────


def test_workflow_estimate_json_roundtrip():
    """WorkflowEstimate serializes to valid JSON and roundtrips."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    s = json.dumps(d, sort_keys=True)
    parsed = json.loads(s)
    assert parsed["schema_version"] == "3.0"
    assert parsed["estimated"] is True
    assert parsed["enriched"] is True
    assert parsed["token_estimate"]["min_tokens"] == 250
    assert len(parsed["node_signatures"]) == 2


# ── Node signature serialization ──────────────────────────────────────────────


def test_node_signature_all_fields_serialized():
    """All NodeTokenSignature fields are serialized correctly."""
    sig = NodeTokenSignature("planner", 120, 1.35, True, False)
    em = _minimal_execution_model()
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=(sig,),
        envelope=CostEnvelope(120, 162, 200, True, "structural-static"),
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    sig_dict = d["node_signatures"][0]
    assert sig_dict["node_name"] == "planner"
    assert sig_dict["base_prompt_tokens"] == 120
    assert sig_dict["expansion_factor"] == 1.35
    assert sig_dict["input_dependency"] is True
    assert sig_dict["symbolic"] is False


def test_node_signature_symbolic():
    """Symbolic node signatures are serialized correctly."""
    sig = NodeTokenSignature("unknown", 0, 1.3, False, True)
    em = _minimal_execution_model()
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=(sig,),
        envelope=CostEnvelope(0, 100, 200, False, "structural-static"),
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    assert d["node_signatures"][0]["symbolic"] is True
    assert d["node_signatures"][0]["base_prompt_tokens"] == 0


# ── Cost envelope serialization ───────────────────────────────────────────────


def test_cost_envelope_serialization():
    """CostEnvelope serializes with all fields."""
    envelope = CostEnvelope(2400, 5100, 9100, True, "structural-static")
    em = _minimal_execution_model()
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=(),
        envelope=envelope,
        assumptions_profile="gpt-4o",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    te = d["token_estimate"]
    assert te["min_tokens"] == 2400
    assert te["expected_tokens"] == 5100
    assert te["max_tokens"] == 9100
    assert te["bounded"] is True
    assert te["confidence"] == "structural-static"


def test_cost_envelope_unbounded():
    """Unbounded CostEnvelope serializes correctly."""
    envelope = CostEnvelope(100, 500, 10000, False, "structural-static")
    em = _minimal_execution_model()
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=(),
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    assert d["token_estimate"]["bounded"] is False


# ── Schema version enforcement ────────────────────────────────────────────────


def test_schema_version_is_forced_to_v3():
    """Phase 3 serializer always emits schema_version 3.0."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    assert d["schema_version"] == "3.0"
    assert d["schema_version"] == ESTIMATED_SCHEMA_VERSION


def test_estimated_flag_always_true():
    """Estimated dict always has estimated=True."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    assert d["estimated"] is True
    assert d["estimated"] is not False


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_empty_node_signatures():
    """WorkflowEstimate with no node signatures."""
    em = _minimal_execution_model()
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=(),
        envelope=CostEnvelope(0, 0, 0, True, "structural-static"),
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    assert d["node_signatures"] == []


def test_empty_per_node_envelopes():
    """WorkflowEstimate with empty per-node envelopes."""
    estimate = _minimal_workflow_estimate()
    estimate = WorkflowEstimate(
        graph_id=estimate.graph_id,
        execution_model=estimate.execution_model,
        node_signatures=estimate.node_signatures,
        envelope=estimate.envelope,
        assumptions_profile=estimate.assumptions_profile,
        per_node_envelopes={},
        per_path_envelopes=estimate.per_path_envelopes,
        warnings=estimate.warnings,
    )
    d = workflow_estimate_to_dict(estimate)
    assert d["per_node_envelopes"] == {}


def test_empty_warnings():
    """WorkflowEstimate with no warnings."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    assert d["warnings"] == []


def test_all_symbolic_nodes():
    """WorkflowEstimate with all symbolic node signatures."""
    em = _minimal_execution_model()
    sigs = (
        NodeTokenSignature("a", 0, 1.3, False, True),
        NodeTokenSignature("b", 0, 1.3, False, True),
    )
    envelope = CostEnvelope(0, 100, 500, False, "structural-static")
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=("All nodes are symbolic",),
    )
    d = workflow_estimate_to_dict(estimate)
    assert all(s["symbolic"] for s in d["node_signatures"])
    assert d["warnings"] == ["All nodes are symbolic"]


# ── Large estimate serialization ─────────────────────────────────────────────


def test_large_workflow_estimate_serialization():
    """Serializing estimate with many node signatures is deterministic."""
    em = _minimal_execution_model()
    n = 50
    sigs = tuple(
        NodeTokenSignature(f"n{i:03d}", 100 + i, 1.2 + i * 0.01, i % 2 == 0, False)
        for i in range(n)
    )
    per_node = {
        f"n{i:03d}": CostEnvelope(100 + i, 120 + i, 140 + i, True, "structural-static")
        for i in range(n)
    }
    envelope = CostEnvelope(5000, 6000, 7000, True, "structural-static")
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes=per_node,
        per_path_envelopes={},
        warnings=(),
    )
    d1 = workflow_estimate_to_dict(estimate)
    d2 = workflow_estimate_to_dict(estimate)
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
    assert len(d1["node_signatures"]) == n
    assert len(d1["per_node_envelopes"]) == n
    # Signatures sorted by name
    names = [s["node_name"] for s in d1["node_signatures"]]
    assert names == sorted(names)


# ── Per-path envelopes ───────────────────────────────────────────────────────


def test_per_path_envelopes_serialization():
    """Per-path envelopes serialize correctly."""
    em = _minimal_execution_model()
    sigs = (NodeTokenSignature("a", 100, 1.2, True, False),)
    envelope = CostEnvelope(100, 120, 140, True, "structural-static")
    per_path = {
        "path_yes": CostEnvelope(100, 120, 140, True, "structural-static"),
        "path_no": CostEnvelope(150, 180, 200, True, "structural-static"),
    }
    estimate = WorkflowEstimate(
        graph_id=em.graph.graph_id,
        execution_model=em,
        node_signatures=sigs,
        envelope=envelope,
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes=per_path,
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    assert len(d["per_path_envelopes"]) == 2
    assert d["per_path_envelopes"]["path_no"]["min_tokens"] == 150
    assert d["per_path_envelopes"]["path_yes"]["min_tokens"] == 100


# ── Integration with Phase 2 ────────────────────────────────────────────────


def test_phase2_enriched_fields_present():
    """All Phase 2 enriched fields are present in Phase 3 output."""
    estimate = _minimal_workflow_estimate()
    d = workflow_estimate_to_dict(estimate)
    # Phase 2 fields
    assert "shape" in d
    assert "cycles" in d
    assert "metrics" in d
    assert "node_annotations" in d
    assert "risks" in d
    # Phase 3 fields
    assert "token_estimate" in d
    assert "node_signatures" in d
    assert "per_node_envelopes" in d
    assert "per_path_envelopes" in d
    assert "warnings" in d


def test_phase3_extends_phase2_not_replaces():
    """Phase 3 output includes Phase 2 data, doesn't replace it."""
    em = _minimal_execution_model()
    # Add some Phase 2 data
    from noctyl.graph import DetectedCycle, NodeAnnotation

    em_with_data = ExecutionModel(
        graph=em.graph,
        entry_point=em.entry_point,
        terminal_nodes=em.terminal_nodes,
        shape="cyclic",
        cycles=(DetectedCycle("self_loop", ("a",), True),),
        metrics=em.metrics,
        node_annotations=(NodeAnnotation("a", "local_function", "pure", "llm_like"),),
        risks=em.risks,
    )
    estimate = WorkflowEstimate(
        graph_id=em_with_data.graph.graph_id,
        execution_model=em_with_data,
        node_signatures=(NodeTokenSignature("a", 100, 1.2, True, False),),
        envelope=CostEnvelope(100, 120, 140, True, "structural-static"),
        assumptions_profile="test",
        per_node_envelopes={},
        per_path_envelopes={},
        warnings=(),
    )
    d = workflow_estimate_to_dict(estimate)
    # Phase 2 data preserved
    assert d["shape"] == "cyclic"
    assert len(d["cycles"]) == 1
    assert len(d["node_annotations"]) == 1
    # Phase 3 data added
    assert d["estimated"] is True
    assert len(d["node_signatures"]) == 1
