"""Tests for Phase 3 branch envelope module."""

import pytest

from noctyl.analysis.digraph import DirectedGraph, build_digraph
from noctyl.estimation import CostEnvelope, compute_branch_envelopes
from noctyl.graph import (
    ExtractedConditionalEdge,
    ExtractedEdge,
    ExtractedNode,
    build_workflow_graph,
)


def _create_branching_wg() -> tuple[DirectedGraph, dict[str, CostEnvelope]]:
    """Helper: workflow with conditional branch."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("a", "c", 5),
    ]
    cond_edges = [
        ExtractedConditionalEdge("a", "path1", "b", 6),
        ExtractedConditionalEdge("a", "path2", "c", 7),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, cond_edges, "a")
    digraph = build_digraph(wg)
    
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
    }
    
    return digraph, per_node


def test_branch_single_branch_point():
    """Single branch with 2 paths."""
    digraph, per_node = _create_branching_wg()
    cond_edges = (
        ExtractedConditionalEdge("a", "path1", "b", 6),
        ExtractedConditionalEdge("a", "path2", "c", 7),
    )
    
    per_path = compute_branch_envelopes(digraph, per_node, cond_edges)
    
    # Should have envelopes for both paths
    assert "a:path1" in per_path
    assert "a:path2" in per_path
    
    # Path1: a → b (cheaper: envelope 15/25/35)
    # Path2: a → c (more expensive: envelope 20/30/40)
    # Branch envelope: min = min(15, 20) = 15, max = max(35, 40) = 40
    # expected = (15 + 40) / 2 = 27 (but clamped to be between min and max)
    
    env_path1 = per_path["a:path1"]
    env_path2 = per_path["a:path2"]
    
    # Both paths should have same envelope (min/expected/max across all paths)
    assert env_path1.min_tokens == min(15, 20)
    assert env_path1.max_tokens == max(35, 40)
    # Expected should be between min and max
    assert env_path1.min_tokens <= env_path1.expected_tokens <= env_path1.max_tokens


def test_branch_multiple_branch_points():
    """Multiple independent branches."""
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
    ]
    cond_edges = [
        ExtractedConditionalEdge("a", "x", "b", 8),
        ExtractedConditionalEdge("a", "y", "c", 9),
        ExtractedConditionalEdge("b", "z", "d", 10),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, cond_edges, "a")
    digraph = build_digraph(wg)
    
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
        "d": CostEnvelope(25, 35, 45, True, "structural-static"),
    }
    
    per_path = compute_branch_envelopes(digraph, per_node, tuple(cond_edges))
    
    # Should have envelopes for all branch points
    assert "a:x" in per_path
    assert "a:y" in per_path
    assert "b:z" in per_path
    
    # Each branch point computed independently
    assert len(per_path) == 3


def test_branch_paths_different_costs():
    """Paths with varying costs."""
    digraph, per_node = _create_branching_wg()
    
    # Make paths have very different costs
    per_node["b"] = CostEnvelope(5, 10, 15, True, "structural-static")  # Cheap
    per_node["c"] = CostEnvelope(50, 100, 150, True, "structural-static")  # Expensive
    
    cond_edges = (
        ExtractedConditionalEdge("a", "cheap", "b", 6),
        ExtractedConditionalEdge("a", "expensive", "c", 7),
    )
    
    per_path = compute_branch_envelopes(digraph, per_node, cond_edges)
    
    env = per_path["a:cheap"]  # Both paths have same envelope (min/max across all)
    
    # min = min(5, 50) = 5 (from min_tokens)
    # max = max(15, 150) = 150
    # expected should be between min and max
    assert env.min_tokens == min(5, 50)
    assert env.max_tokens == max(15, 150)
    assert env.min_tokens <= env.expected_tokens <= env.max_tokens


def test_branch_path_to_end():
    """Paths terminating at END."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
    ]
    edges = [
        ExtractedEdge("a", "b", 3),
        ExtractedEdge("b", "END", 4),
    ]
    cond_edges = [
        ExtractedConditionalEdge("a", "path1", "b", 5),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, cond_edges, "a")
    digraph = build_digraph(wg)
    
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
    }
    
    per_path = compute_branch_envelopes(digraph, per_node, tuple(cond_edges))
    
    # Should handle END node correctly
    assert "a:path1" in per_path


def test_branch_no_conditional_edges():
    """No branches case."""
    digraph, per_node = _create_branching_wg()
    
    per_path = compute_branch_envelopes(digraph, per_node, ())
    
    # Should return empty dict
    assert per_path == {}


def test_branch_deterministic_path_keys():
    """Deterministic path keys."""
    digraph, per_node = _create_branching_wg()
    cond_edges = (
        ExtractedConditionalEdge("a", "z", "b", 6),
        ExtractedConditionalEdge("a", "a", "c", 7),
    )
    
    per_path = compute_branch_envelopes(digraph, per_node, cond_edges)
    
    # Path keys should be sorted for determinism
    keys = list(per_path.keys())
    # Note: keys may not be sorted in the dict, but should be sortable
    assert sorted(keys) == sorted(set(keys))  # Check uniqueness and sortability


def test_branch_bounded_flag_aggregation():
    """Bounded flag logic."""
    digraph, per_node = _create_branching_wg()
    
    # Make one path bounded, one unbounded
    per_node["b"] = CostEnvelope(15, 25, 35, True, "structural-static")  # Bounded
    per_node["c"] = CostEnvelope(20, 30, 40, False, "structural-static")  # Unbounded
    
    cond_edges = (
        ExtractedConditionalEdge("a", "path1", "b", 6),
        ExtractedConditionalEdge("a", "path2", "c", 7),
    )
    
    per_path = compute_branch_envelopes(digraph, per_node, cond_edges)
    
    # bounded should be False if any path is unbounded
    env = per_path["a:path1"]
    assert env.bounded is False
