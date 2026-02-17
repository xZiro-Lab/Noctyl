"""Tests for Phase 3 token propagation module."""

import pytest

from noctyl.analysis.digraph import DirectedGraph, build_digraph
from noctyl.estimation import CostEnvelope, ModelProfile, NodeTokenSignature, propagate_tokens
from noctyl.graph import ExtractedEdge, ExtractedNode, build_workflow_graph


def _default_model_profile() -> ModelProfile:
    """Helper: default ModelProfile for testing."""
    return ModelProfile("default", 1.3, 0.5, 0.0, 0.0)


def _create_linear_wg() -> tuple[DirectedGraph, tuple[NodeTokenSignature, ...]]:
    """Helper: linear workflow graph (a → b → c)."""
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
    
    signatures = (
        NodeTokenSignature("a", 100, 1.3, False, False),
        NodeTokenSignature("b", 50, 1.3, False, False),
        NodeTokenSignature("c", 75, 1.3, False, False),
    )
    
    return digraph, signatures


def test_propagate_linear_workflow():
    """Simple linear chain (a → b → c)."""
    digraph, signatures = _create_linear_wg()
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    assert "a" in envelopes
    assert "b" in envelopes
    assert "c" in envelopes
    
    # Entry point 'a' starts with base_prompt_tokens
    env_a = envelopes["a"]
    assert env_a.min_tokens == 100
    assert env_a.expected_tokens == 100
    assert env_a.max_tokens == 100
    
    # Node 'b': T_out = (T_in + base_prompt) × expansion_factor
    # T_in = 100, base_prompt = 50, expansion = 1.3
    # T_out = (100 + 50) × 1.3 = 195
    env_b = envelopes["b"]
    assert env_b.expected_tokens == int((100 + 50) * 1.3)
    
    # Node 'c': T_out = (195 + 75) × 1.3 = 351
    env_c = envelopes["c"]
    assert env_c.expected_tokens == int((env_b.expected_tokens + 75) * 1.3)


def test_propagate_branching_workflow():
    """Branching workflow (a → b, a → c)."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("a", "c", 5),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, [], "a")
    digraph = build_digraph(wg)
    
    signatures = (
        NodeTokenSignature("a", 100, 1.3, False, False),
        NodeTokenSignature("b", 50, 1.3, False, False),
        NodeTokenSignature("c", 75, 1.3, False, False),
    )
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    # Both branches receive tokens from 'a'
    env_b = envelopes["b"]
    env_c = envelopes["c"]
    
    # Both should have same input (from 'a')
    assert env_b.expected_tokens == int((100 + 50) * 1.3)
    assert env_c.expected_tokens == int((100 + 75) * 1.3)


def test_propagate_cyclic_workflow():
    """Simple cycle (a → b → a)."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
    ]
    edges = [
        ExtractedEdge("a", "b", 3),
        ExtractedEdge("b", "a", 4),
    ]
    wg = build_workflow_graph("test:0", nodes, edges, [], "a")
    digraph = build_digraph(wg)
    
    signatures = (
        NodeTokenSignature("a", 100, 1.3, False, False),
        NodeTokenSignature("b", 50, 1.3, False, False),
    )
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    # Cycle nodes should propagate normally (before amplification)
    # Entry point 'a' starts with base_prompt_tokens
    env_a = envelopes["a"]
    assert env_a.expected_tokens == 100
    
    # Node 'b' receives from 'a'
    env_b = envelopes["b"]
    assert env_b.expected_tokens == int((100 + 50) * 1.3)
    
    # Cycle nodes should be marked bounded=False initially
    # (loop amplification will handle bounds)


def test_propagate_convergence():
    """Multiple incoming edges (b → d, c → d)."""
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
    ]
    wg = build_workflow_graph("test:0", nodes, edges, [], "a")
    digraph = build_digraph(wg)
    
    signatures = (
        NodeTokenSignature("a", 100, 1.3, False, False),
        NodeTokenSignature("b", 50, 1.3, False, False),
        NodeTokenSignature("c", 75, 1.3, False, False),
        NodeTokenSignature("d", 25, 1.3, False, False),
    )
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    # Node 'd' should sum tokens from both 'b' and 'c'
    env_d = envelopes["d"]
    
    # Path a → b → d
    path_bd = int((int((100 + 50) * 1.3) + 25) * 1.3)
    # Path a → c → d
    path_cd = int((int((100 + 75) * 1.3) + 25) * 1.3)
    
    # Sum of both paths
    expected_sum = path_bd + path_cd
    assert env_d.expected_tokens == expected_sum


def test_propagate_symbolic_nodes():
    """Nodes with symbolic=True."""
    digraph, _ = _create_linear_wg()
    
    signatures = (
        NodeTokenSignature("a", 0, 1.3, False, True),  # Symbolic
        NodeTokenSignature("b", 0, 1.3, False, True),  # Symbolic
        NodeTokenSignature("c", 75, 1.3, False, False),  # Not symbolic, base=75
    )
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    # Symbolic nodes use base_prompt_tokens=0 but expansion_factor still applied
    env_a = envelopes["a"]
    assert env_a.expected_tokens == 0  # Entry point with base=0
    
    env_b = envelopes["b"]
    # T_out = (0 + 0) × 1.3 = 0
    assert env_b.expected_tokens == 0
    
    env_c = envelopes["c"]
    # T_out = (0 + 75) × 1.3 = 97.5 ≈ 97
    assert env_c.expected_tokens == int((0 + 75) * 1.3)


def test_propagate_no_predecessors():
    """Entry point handling."""
    digraph, signatures = _create_linear_wg()
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    # Entry point 'a' has no predecessors, starts with base_prompt_tokens
    env_a = envelopes["a"]
    assert env_a.min_tokens == 100
    assert env_a.expected_tokens == 100
    assert env_a.max_tokens == 100


def test_propagate_expansion_factor_from_signature():
    """Custom expansion factors from signatures."""
    digraph, _ = _create_linear_wg()
    
    signatures = (
        NodeTokenSignature("a", 100, 1.5, False, False),  # Custom expansion
        NodeTokenSignature("b", 50, 1.2, False, False),  # Custom expansion
        NodeTokenSignature("c", 75, 1.3, False, False),
    )
    profile = ModelProfile("default", 1.3, 0.5, 0.0, 0.0)  # Default 1.3
    
    envelopes = propagate_tokens(digraph, signatures, profile)
    
    # Node 'a' uses its own expansion_factor (1.5)
    env_a = envelopes["a"]
    assert env_a.expected_tokens == 100  # Entry point
    
    # Node 'b' uses its own expansion_factor (1.2), not profile default
    env_b = envelopes["b"]
    assert env_b.expected_tokens == int((100 + 50) * 1.2)  # Uses 1.2, not 1.3


def test_propagate_empty_graph():
    """Empty graph edge case."""
    wg = build_workflow_graph("test:0", [], [], [], None)
    digraph = build_digraph(wg)
    profile = _default_model_profile()
    
    envelopes = propagate_tokens(digraph, (), profile)
    
    # Should only have START and END
    assert len(envelopes) <= 2
    if "START" in envelopes:
        assert envelopes["START"].expected_tokens == 0


def test_propagate_deterministic_output():
    """Deterministic output (same inputs → identical outputs)."""
    digraph, signatures = _create_linear_wg()
    profile = _default_model_profile()
    
    envelopes1 = propagate_tokens(digraph, signatures, profile)
    envelopes2 = propagate_tokens(digraph, signatures, profile)
    
    # Same inputs should produce identical outputs
    assert envelopes1 == envelopes2
    
    # Check sorted order (keys should be sorted)
    keys1 = list(envelopes1.keys())
    keys2 = list(envelopes2.keys())
    assert keys1 == keys2
