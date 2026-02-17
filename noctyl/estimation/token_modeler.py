"""
TokenModeler: orchestrates token estimation pipeline.
"""

from __future__ import annotations

from noctyl.analysis.digraph import build_digraph
from noctyl.estimation.aggregation import aggregate_workflow_envelope
from noctyl.estimation.branch_envelope import compute_branch_envelopes
from noctyl.estimation.data_model import ModelProfile, WorkflowEstimate
from noctyl.estimation.loop_amplification import apply_loop_amplification
from noctyl.estimation.propagation import propagate_tokens
from noctyl.estimation.prompt_detection import compute_node_token_signatures
from noctyl.graph.execution_model import ExecutionModel


class TokenModeler:
    """Phase 3 token estimation: orchestrates propagation, amplification, and aggregation."""

    def estimate(
        self,
        execution_model: ExecutionModel,
        model_profile: ModelProfile,
        *,
        source: str | None = None,
        file_path: str | None = None,
    ) -> WorkflowEstimate:
        """
        Estimate token usage for a workflow.
        
        Pipeline:
        1. Build DirectedGraph from WorkflowGraph
        2. Compute node token signatures (prompt detection)
        3. Propagate tokens through graph
        4. Apply loop amplification
        5. Compute branch envelopes
        6. Aggregate workflow envelope
        7. Collect warnings
        8. Build WorkflowEstimate
        
        Returns WorkflowEstimate with all estimation artifacts.
        """
        # Get WorkflowGraph from ExecutionModel
        wg = execution_model.graph
        
        # Build DirectedGraph
        digraph = build_digraph(wg)
        
        # Compute node token signatures
        node_signatures = compute_node_token_signatures(wg, source, model_profile)
        
        # Propagate tokens
        per_node = propagate_tokens(digraph, node_signatures, model_profile)
        
        # Apply loop amplification
        per_node = apply_loop_amplification(
            per_node, execution_model.cycles, model_profile
        )
        
        # Compute branch envelopes
        per_path = compute_branch_envelopes(
            digraph, per_node, wg.conditional_edges
        )
        
        # Aggregate workflow envelope
        workflow_envelope = aggregate_workflow_envelope(
            per_node, wg.entry_point, wg.terminal_nodes, digraph
        )
        
        # Collect warnings
        warnings: list[str] = []
        
        # Symbolic nodes
        for sig in node_signatures:
            if sig.symbolic:
                warnings.append(
                    f"Node {sig.node_name} marked as symbolic (unresolvable callable)"
                )
        
        # Unbounded loops
        for cycle in execution_model.cycles:
            if cycle.cycle_type != "non_terminating":
                # All cycles are unbounded for now (no static bounds in Phase 2)
                cycle_nodes_str = ", ".join(sorted(cycle.nodes))
                warnings.append(
                    f"Cycle [{cycle_nodes_str}] has unbounded iterations (assumed 5)"
                )
        
        # Non-terminating cycles
        for cycle in execution_model.cycles:
            if cycle.cycle_type == "non_terminating":
                cycle_nodes_str = ", ".join(sorted(cycle.nodes))
                warnings.append(f"Non-terminating cycle detected: [{cycle_nodes_str}]")
        
        # Sort warnings for determinism
        warnings_sorted = tuple(sorted(warnings))
        
        # Build WorkflowEstimate
        # Sort dict keys for determinism
        per_node_sorted = dict(sorted(per_node.items()))
        per_path_sorted = dict(sorted(per_path.items()))
        
        return WorkflowEstimate(
            graph_id=wg.graph_id,
            execution_model=execution_model,
            node_signatures=node_signatures,  # Already sorted by compute_node_token_signatures
            envelope=workflow_envelope,
            assumptions_profile=model_profile.name,
            per_node_envelopes=per_node_sorted,
            per_path_envelopes=per_path_sorted,
            warnings=warnings_sorted,
        )
