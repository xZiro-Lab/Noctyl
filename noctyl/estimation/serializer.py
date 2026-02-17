"""
Phase 3 estimation serializer: deterministic JSON serialization for WorkflowEstimate.
"""

from __future__ import annotations

from noctyl.graph.execution_model import execution_model_to_dict

from noctyl.estimation.data_model import (
    CostEnvelope,
    NodeTokenSignature,
    WorkflowEstimate,
)

ESTIMATED_SCHEMA_VERSION = "3.0"


def _cost_envelope_to_dict(envelope: CostEnvelope) -> dict:
    """Convert CostEnvelope to JSON-serializable dict."""
    return {
        "min_tokens": envelope.min_tokens,
        "expected_tokens": envelope.expected_tokens,
        "max_tokens": envelope.max_tokens,
        "bounded": envelope.bounded,
        "confidence": envelope.confidence,
    }


def workflow_estimate_to_dict(estimate: WorkflowEstimate) -> dict:
    """
    Return a JSON-serializable dict with deterministic ordering.
    
    Extends Phase 2 enriched dict (via execution_model_to_dict) with Phase 3 estimation fields.
    schema_version 3.0, estimated: true, enriched: true (from Phase 2).
    """
    # Start with Phase 2 enriched dict
    base = execution_model_to_dict(estimate.execution_model)
    
    # Override schema version and add estimated flag
    base["schema_version"] = ESTIMATED_SCHEMA_VERSION
    base["estimated"] = True
    # enriched: true is already set by execution_model_to_dict
    
    # Token estimate (workflow-level envelope)
    base["token_estimate"] = {
        "assumptions_profile": estimate.assumptions_profile,
        "min_tokens": estimate.envelope.min_tokens,
        "expected_tokens": estimate.envelope.expected_tokens,
        "max_tokens": estimate.envelope.max_tokens,
        "bounded": estimate.envelope.bounded,
        "confidence": estimate.envelope.confidence,
    }
    
    # Node signatures: sorted by node_name for determinism
    sigs_sorted = sorted(estimate.node_signatures, key=lambda s: s.node_name)
    base["node_signatures"] = [
        {
            "node_name": sig.node_name,
            "base_prompt_tokens": sig.base_prompt_tokens,
            "expansion_factor": sig.expansion_factor,
            "input_dependency": sig.input_dependency,
            "symbolic": sig.symbolic,
        }
        for sig in sigs_sorted
    ]
    
    # Per-node envelopes: sorted dict keys for determinism
    base["per_node_envelopes"] = {
        node_name: _cost_envelope_to_dict(envelope)
        for node_name, envelope in sorted(estimate.per_node_envelopes.items())
    }
    
    # Per-path envelopes: sorted dict keys for determinism
    base["per_path_envelopes"] = {
        path_id: _cost_envelope_to_dict(envelope)
        for path_id, envelope in sorted(estimate.per_path_envelopes.items())
    }
    
    # Warnings: sorted alphabetically for determinism
    base["warnings"] = sorted(estimate.warnings)
    
    return base
