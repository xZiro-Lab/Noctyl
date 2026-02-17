"""Phase 3 estimation: token signatures, model profiles, cost envelopes, and workflow estimates."""

from noctyl.estimation.aggregation import aggregate_workflow_envelope
from noctyl.estimation.branch_envelope import compute_branch_envelopes
from noctyl.estimation.data_model import (
    CostEnvelope,
    ModelProfile,
    NodeTokenSignature,
    WorkflowEstimate,
)
from noctyl.estimation.loop_amplification import apply_loop_amplification
from noctyl.estimation.propagation import propagate_tokens
from noctyl.estimation.prompt_detection import (
    PromptFragment,
    compute_node_token_signatures,
    detect_prompt_strings,
    estimate_tokens_from_string,
)
from noctyl.estimation.serializer import (
    ESTIMATED_SCHEMA_VERSION,
    workflow_estimate_to_dict,
)
from noctyl.estimation.token_modeler import TokenModeler

__all__ = [
    "CostEnvelope",
    "ESTIMATED_SCHEMA_VERSION",
    "ModelProfile",
    "NodeTokenSignature",
    "PromptFragment",
    "TokenModeler",
    "WorkflowEstimate",
    "aggregate_workflow_envelope",
    "apply_loop_amplification",
    "compute_branch_envelopes",
    "compute_node_token_signatures",
    "detect_prompt_strings",
    "estimate_tokens_from_string",
    "propagate_tokens",
    "workflow_estimate_to_dict",
]
