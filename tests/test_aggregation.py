"""Tests for Phase 3 workflow aggregation module."""

import pytest

from noctyl.analysis.digraph import DirectedGraph, build_digraph
from noctyl.estimation import CostEnvelope, aggregate_workflow_envelope
from noctyl.graph import ExtractedEdge, ExtractedNode, build_workflow_graph


def _create_linear_wg() -> tuple[DirectedGraph, dict[str, CostEnvelope]]:
    """Helper: linear workflow graph."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("b", "c", 5),
        ExtractedEdge("c", "END", 6),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, [], "a")
    digraph = build_digraph(wg)
    
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
    }
    
    return digraph, per_node


def test_aggregate_single_terminal():
    """Single terminal node."""
    digraph, per_node = _create_linear_wg()
    
    envelope = aggregate_workflow_envelope(
        per_node, "a", ("c",), digraph
    )
    
    # Should sum envelopes along path a → b → c
    # Path includes all nodes: a (20) + b (25) + c (30) = 75
    # But aggregation sums from entry to terminal, so includes all nodes
    assert envelope.expected_tokens >= 20 + 25 + 30  # At least sum of all nodes
    assert envelope.min_tokens >= min(10, 15, 20)
    assert envelope.max_tokens >= max(30, 35, 40)


def test_aggregate_multiple_terminals():
    """Multiple terminal nodes."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("a", "c", 5),
        ExtractedEdge("b", "END", 6),
        ExtractedEdge("c", "END", 7),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, [], "a")
    digraph = build_digraph(wg)
    
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
    }
    
    envelope = aggregate_workflow_envelope(
        per_node, "a", ("b", "c"), digraph
    )
    
    # Path a → b: expected includes a + b
    # Path a → c: expected includes a + c
    # Aggregate: sum across terminals (or max, depending on implementation)
    assert envelope.expected_tokens > 0  # Should have some tokens
    assert envelope.min_tokens >= min(10, 15, 20)
    assert envelope.max_tokens >= max(30, 35, 40)
    # Ensure invariant holds
    assert envelope.min_tokens <= envelope.expected_tokens <= envelope.max_tokens


def test_aggregate_no_entry_point():
    """Missing entry point."""
    digraph, per_node = _create_linear_wg()
    
    envelope = aggregate_workflow_envelope(
        per_node, None, ("c",), digraph
    )
    
    # Should return zero envelope
    assert envelope.min_tokens == 0
    assert envelope.expected_tokens == 0
    assert envelope.max_tokens == 0


def test_aggregate_no_terminals():
    """No terminal nodes."""
    digraph, per_node = _create_linear_wg()
    
    envelope = aggregate_workflow_envelope(
        per_node, "a", (), digraph
    )
    
    # Should return zero envelope
    assert envelope.min_tokens == 0
    assert envelope.expected_tokens == 0
    assert envelope.max_tokens == 0


def test_aggregate_multiple_paths_to_terminal():
    """Multiple paths to same terminal."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
        ExtractedNode("d", "i", 4),
    ]
    edges = [
        ExtractedEdge("a", "b", 5),
        ExtractedEdge("a", "c", 6),
        ExtractedEdge("b", "d", 7),
        ExtractedEdge("c", "d", 8),
        ExtractedEdge("d", "END", 9),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, [], "a")
    digraph = build_digraph(wg)
    
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
        "d": CostEnvelope(25, 35, 45, True, "structural-static"),
    }
    
    envelope = aggregate_workflow_envelope(
        per_node, "a", ("d",), digraph
    )
    
    # Path a → b → d: expected = 20 + 25 + 35 = 80
    # Path a → c → d: expected = 20 + 30 + 35 = 85
    # Should take max (worst case) = 85
    assert envelope.expected_tokens == max(80, 85)


def test_aggregate_bounded_flag():
    """Bounded flag aggregation."""
    digraph, per_node = _create_linear_wg()
    
    # Make one node unbounded
    per_node["b"] = CostEnvelope(15, 25, 35, False, "structural-static")
    
    envelope = aggregate_workflow_envelope(
        per_node, "a", ("c",), digraph
    )
    
    # bounded should be False if any node is unbounded
    assert envelope.bounded is False


def test_aggregate_confidence_always_structural_static():
    """Confidence field always structural-static."""
    digraph, per_node = _create_linear_wg()
    
    envelope = aggregate_workflow_envelope(
        per_node, "a", ("c",), digraph
    )
    
    assert envelope.confidence == "structural-static"


def test_aggregate_deterministic():
    """Deterministic output."""
    digraph, per_node = _create_linear_wg()
    
    envelope1 = aggregate_workflow_envelope(
        per_node, "a", ("c",), digraph
    )
    envelope2 = aggregate_workflow_envelope(
        per_node, "a", ("c",), digraph
    )
    
    # Same inputs should produce identical outputs
    assert envelope1 == envelope2
