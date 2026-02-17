"""
Phase 3 estimation data model: token signatures, model profiles, cost envelopes, and workflow estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from noctyl.graph.execution_model import ExecutionModel

ConfidenceType = Literal["structural-static"]


@dataclass(frozen=True)
class NodeTokenSignature:
    """Per-node static token signature for Phase 3 estimation."""

    node_name: str
    base_prompt_tokens: int  # Statically detected prompt size
    expansion_factor: float  # Heuristic growth factor (typically 1.0-2.0)
    input_dependency: bool  # Whether output scales with input tokens
    symbolic: bool  # True if prompt size couldn't be statically determined


@dataclass(frozen=True)
class ModelProfile:
    """User-declared model assumptions for token estimation."""

    name: str  # e.g., "gpt-4o", "default"
    expansion_factor: float  # Default model expansion factor
    output_ratio: float  # Ratio of output tokens to input tokens (typically 0.0-1.0)
    pricing_input_per_1k: float  # Optional pricing per 1k input tokens (defaults to 0.0)
    pricing_output_per_1k: float  # Optional pricing per 1k output tokens (defaults to 0.0)


@dataclass(frozen=True)
class CostEnvelope:
    """
    Token cost envelope: min/expected/max range for a path or workflow.
    
    Invariant: min_tokens <= expected_tokens <= max_tokens
    """

    min_tokens: int  # Minimum token count
    expected_tokens: int  # Expected/midpoint token count
    max_tokens: int  # Maximum token count
    bounded: bool  # True if all loops have known bounds
    confidence: ConfidenceType  # Always "structural-static" for Phase 3

    def __post_init__(self) -> None:
        """Validate invariant: min <= expected <= max."""
        if not (self.min_tokens <= self.expected_tokens <= self.max_tokens):
            raise ValueError(
                f"CostEnvelope invariant violated: "
                f"min_tokens ({self.min_tokens}) <= expected_tokens ({self.expected_tokens}) <= "
                f"max_tokens ({self.max_tokens})"
            )


@dataclass(frozen=True)
class WorkflowEstimate:
    """
    Full Phase 3 estimation result per graph.
    
    Holds ExecutionModel reference plus all estimation artifacts:
    node signatures, workflow-level envelope, per-node/per-path envelopes, warnings.
    """

    graph_id: str  # From ExecutionModel.graph.graph_id
    execution_model: ExecutionModel  # Phase 2 enriched model
    node_signatures: tuple[NodeTokenSignature, ...]  # Sorted by node_name for determinism
    envelope: CostEnvelope  # Workflow-level token envelope
    assumptions_profile: str  # ModelProfile.name used
    per_node_envelopes: dict[str, CostEnvelope]  # Per-node token envelopes
    per_path_envelopes: dict[str, CostEnvelope]  # Per-path envelopes for conditional branches
    warnings: tuple[str, ...]  # Estimation warnings (e.g., unbounded loops, symbolic nodes)
