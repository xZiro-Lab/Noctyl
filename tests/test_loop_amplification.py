"""Tests for Phase 3 loop amplification module."""

import pytest

from noctyl.estimation import (
    CostEnvelope,
    ModelProfile,
    apply_loop_amplification,
)
from noctyl.graph.execution_model import DetectedCycle


def _default_model_profile() -> ModelProfile:
    """Helper: default ModelProfile for testing."""
    return ModelProfile("default", 1.3, 0.5, 0.0, 0.0)


def test_amplify_self_loop():
    """Single node cycle."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("self_loop", ("a",), True),
    )
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile, assumed_iterations=5)
    
    # Self-loop node should be multiplied by iterations
    env_a = result["a"]
    assert env_a.min_tokens == 10 * 5
    assert env_a.expected_tokens == 20 * 5
    assert env_a.max_tokens == 30 * 5
    assert env_a.bounded is False  # Assumed iterations -> unbounded


def test_amplify_multi_node_cycle():
    """Multi-node cycle (a → b → c → a)."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("multi_node", ("a", "b", "c"), True),
    )
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile, assumed_iterations=3)
    
    # All cycle nodes should be amplified equally
    assert result["a"].expected_tokens == 20 * 3
    assert result["b"].expected_tokens == 25 * 3
    assert result["c"].expected_tokens == 30 * 3
    
    # All should be marked unbounded
    assert result["a"].bounded is False
    assert result["b"].bounded is False
    assert result["c"].bounded is False


def test_amplify_non_terminating_cycle():
    """Non-terminating cycle."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("non_terminating", ("a", "b"), False),
    )
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile, assumed_iterations=5)
    
    # Non-terminating cycles use assumed_iterations
    assert result["a"].expected_tokens == 20 * 5
    assert result["b"].expected_tokens == 25 * 5
    assert result["a"].bounded is False
    assert result["b"].bounded is False


def test_amplify_multiple_cycles():
    """Multiple independent cycles."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
        "d": CostEnvelope(25, 35, 45, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("self_loop", ("a",), True),
        DetectedCycle("multi_node", ("c", "d"), True),
    )
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile, assumed_iterations=4)
    
    # First cycle: 'a' amplified
    assert result["a"].expected_tokens == 20 * 4
    
    # Second cycle: 'c' and 'd' amplified
    assert result["c"].expected_tokens == 30 * 4
    assert result["d"].expected_tokens == 35 * 4
    
    # 'b' not in any cycle, should be unchanged
    assert result["b"].expected_tokens == 25
    assert result["b"].bounded is True


def test_amplify_empty_cycles():
    """No cycles case."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
    }
    cycles = ()
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile)
    
    # Should be unchanged
    assert result == per_node


def test_amplify_custom_iterations():
    """Custom assumed_iterations."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("self_loop", ("a",), True),
    )
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile, assumed_iterations=10)
    
    # Should use custom iterations
    assert result["a"].expected_tokens == 20 * 10


def test_amplify_preserves_non_cycle_nodes():
    """Non-cycle nodes unchanged."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
        "c": CostEnvelope(20, 30, 40, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("self_loop", ("a",), True),
    )
    profile = _default_model_profile()
    
    result = apply_loop_amplification(per_node, cycles, profile)
    
    # 'b' and 'c' not in cycle, should be unchanged
    assert result["b"] == per_node["b"]
    assert result["c"] == per_node["c"]


def test_amplify_deterministic():
    """Deterministic output."""
    per_node = {
        "a": CostEnvelope(10, 20, 30, True, "structural-static"),
        "b": CostEnvelope(15, 25, 35, True, "structural-static"),
    }
    cycles = (
        DetectedCycle("multi_node", ("a", "b"), True),
    )
    profile = _default_model_profile()
    
    result1 = apply_loop_amplification(per_node, cycles, profile)
    result2 = apply_loop_amplification(per_node, cycles, profile)
    
    # Same inputs should produce identical outputs
    assert result1 == result2
