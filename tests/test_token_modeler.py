"""Integration tests for Phase 3 TokenModeler."""

import pytest

from noctyl.analysis import analyze
from noctyl.estimation import ModelProfile, TokenModeler
from noctyl.graph import (
    ExtractedConditionalEdge,
    ExtractedEdge,
    ExtractedNode,
    build_workflow_graph,
)


def _create_model_profile() -> ModelProfile:
    """Helper: create ModelProfile for testing."""
    return ModelProfile("test-profile", 1.3, 0.5, 0.0, 0.0)


def _create_linear_wg():
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
    return build_workflow_graph("test:0", nodes, edges, [], "a")


def _create_cyclic_wg():
    """Helper: cyclic workflow graph."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
    ]
    edges = [
        ExtractedEdge("a", "b", 3),
        ExtractedEdge("b", "a", 4),
    ]
    return build_workflow_graph("test:0", nodes, edges, [], "a")


def _create_branching_wg():
    """Helper: branching workflow graph."""
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
    return build_workflow_graph("test:0", nodes, edges, cond_edges, "a")


def test_estimate_linear_workflow():
    """End-to-end linear workflow."""
    wg = _create_linear_wg()
    source = '''def f(): return "prompt1"
def g(): return "prompt2"
def h(): return "prompt3"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Verify WorkflowEstimate structure
    assert estimate.graph_id == "test:0"
    assert estimate.execution_model == em
    assert len(estimate.node_signatures) == 3
    assert estimate.envelope is not None
    assert estimate.assumptions_profile == "test-profile"
    assert len(estimate.per_node_envelopes) > 0
    assert estimate.per_path_envelopes == {}  # No conditional edges
    assert isinstance(estimate.warnings, tuple)


def test_estimate_cyclic_workflow():
    """End-to-end cyclic workflow."""
    wg = _create_cyclic_wg()
    source = '''def f(): return "prompt1"
def g(): return "prompt2"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Verify loop amplification applied
    assert len(estimate.node_signatures) == 2
    assert estimate.envelope is not None
    
    # Should have warnings for unbounded loops
    assert len(estimate.warnings) > 0
    # Check for cycle-related warnings (format may vary)
    assert any("Cycle" in w or "cycle" in w or "unbounded" in w for w in estimate.warnings)


def test_estimate_branching_workflow():
    """End-to-end branching workflow."""
    wg = _create_branching_wg()
    source = '''def f(): return "prompt1"
def g(): return "prompt2"
def h(): return "prompt3"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Verify branch envelopes computed
    assert len(estimate.per_path_envelopes) > 0
    assert "a:path1" in estimate.per_path_envelopes
    assert "a:path2" in estimate.per_path_envelopes


def test_estimate_with_source_code():
    """Source code provided."""
    wg = _create_linear_wg()
    source = '''def f(): return "Hello world"
def g(): return "Goodbye"
def h(): return "Test"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Verify prompt detection ran
    assert len(estimate.node_signatures) == 3
    # At least some nodes should have base_prompt_tokens > 0
    assert any(sig.base_prompt_tokens > 0 for sig in estimate.node_signatures)


def test_estimate_without_source_code():
    """No source code."""
    wg = _create_linear_wg()
    em = analyze(wg)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile)
    
    # Verify all nodes marked symbolic
    assert all(sig.symbolic for sig in estimate.node_signatures)
    assert all(sig.base_prompt_tokens == 0 for sig in estimate.node_signatures)


def test_estimate_warnings_collection():
    """Warnings collection."""
    wg = _create_cyclic_wg()
    source = '''def f(): return "prompt"
def g(): return "prompt"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Should have warnings for unbounded loops
    assert len(estimate.warnings) > 0
    assert isinstance(estimate.warnings, tuple)
    # Warnings should be sorted
    assert list(estimate.warnings) == sorted(estimate.warnings)


def test_estimate_deterministic_output():
    """Deterministic output."""
    wg = _create_linear_wg()
    source = '''def f(): return "prompt"
def g(): return "prompt"
def h(): return "prompt"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate1 = modeler.estimate(em, profile, source=source)
    estimate2 = modeler.estimate(em, profile, source=source)
    
    # Same inputs should produce identical outputs
    assert estimate1.graph_id == estimate2.graph_id
    assert estimate1.envelope == estimate2.envelope
    assert estimate1.node_signatures == estimate2.node_signatures
    assert estimate1.per_node_envelopes == estimate2.per_node_envelopes
    assert estimate1.per_path_envelopes == estimate2.per_path_envelopes
    assert estimate1.warnings == estimate2.warnings


def test_estimate_empty_graph():
    """Empty graph edge case."""
    wg = build_workflow_graph("test:0", [], [], [], None)
    em = analyze(wg)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile)
    
    # Should handle gracefully
    assert estimate.graph_id == "test:0"
    assert len(estimate.node_signatures) == 0
    assert estimate.envelope.min_tokens == 0
    assert estimate.envelope.expected_tokens == 0
    assert estimate.envelope.max_tokens == 0


def test_estimate_model_profile_integration():
    """ModelProfile usage."""
    wg = _create_linear_wg()
    source = '''def f(): return "prompt"
def g(): return "prompt"
def h(): return "prompt"
'''
    em = analyze(wg, source=source)
    
    profile1 = ModelProfile("profile1", 1.5, 0.6, 0.0, 0.0)
    profile2 = ModelProfile("profile2", 2.0, 0.7, 0.0, 0.0)
    
    modeler = TokenModeler()
    estimate1 = modeler.estimate(em, profile1, source=source)
    estimate2 = modeler.estimate(em, profile2, source=source)
    
    # Different profiles should produce different estimates
    assert estimate1.assumptions_profile == "profile1"
    assert estimate2.assumptions_profile == "profile2"
    assert estimate1.envelope.expected_tokens != estimate2.envelope.expected_tokens


def test_estimate_workflow_estimate_structure():
    """WorkflowEstimate validation."""
    wg = _create_linear_wg()
    source = '''def f(): return "prompt"
def g(): return "prompt"
def h(): return "prompt"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Verify all required fields present
    assert hasattr(estimate, "graph_id")
    assert hasattr(estimate, "execution_model")
    assert hasattr(estimate, "node_signatures")
    assert hasattr(estimate, "envelope")
    assert hasattr(estimate, "assumptions_profile")
    assert hasattr(estimate, "per_node_envelopes")
    assert hasattr(estimate, "per_path_envelopes")
    assert hasattr(estimate, "warnings")
    
    # Verify execution_model reference preserved
    assert estimate.execution_model == em


def test_estimate_per_node_envelopes_sorted():
    """Dict sorting."""
    wg = _create_linear_wg()
    source = '''def f(): return "prompt"
def g(): return "prompt"
def h(): return "prompt"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Keys should be sorted
    per_node_keys = list(estimate.per_node_envelopes.keys())
    assert per_node_keys == sorted(per_node_keys)
    
    per_path_keys = list(estimate.per_path_envelopes.keys())
    assert per_path_keys == sorted(per_path_keys)


def test_estimate_integration_with_phase2():
    """Phase 2 integration."""
    wg = _create_linear_wg()
    source = '''def f(): return "prompt"
def g(): return "prompt"
def h(): return "prompt"
'''
    em = analyze(wg, source=source)
    profile = _create_model_profile()
    modeler = TokenModeler()
    
    estimate = modeler.estimate(em, profile, source=source)
    
    # Verify all Phase 2 data preserved
    assert estimate.execution_model.shape == em.shape
    assert estimate.execution_model.cycles == em.cycles
    assert estimate.execution_model.metrics == em.metrics
    assert estimate.execution_model.node_annotations == em.node_annotations
    assert estimate.execution_model.risks == em.risks
