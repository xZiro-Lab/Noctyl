"""
GraphAnalyzer: orchestrate digraph, control_flow, metrics, node_annotation, structural_risk -> ExecutionModel.
"""

from __future__ import annotations

from noctyl.graph.execution_model import ExecutionModel
from noctyl.graph.graph import WorkflowGraph

from noctyl.analysis.control_flow import compute_control_flow
from noctyl.analysis.digraph import build_digraph
from noctyl.analysis.metrics import compute_metrics
from noctyl.analysis.node_annotation import compute_node_annotations
from noctyl.analysis.structural_risk import compute_structural_risk


class GraphAnalyzer:
    """Phase 2 analyzer: enrich a WorkflowGraph into an ExecutionModel."""

    def analyze(
        self,
        workflow_graph: WorkflowGraph,
        *,
        source: str | None = None,
        file_path: str | None = None,
    ) -> ExecutionModel:
        """
        Build an ExecutionModel from a WorkflowGraph.
        When source (and optionally file_path) is provided, node annotations use the source for origin/state/role.
        """
        dg = build_digraph(workflow_graph)
        shape, cycles = compute_control_flow(dg, workflow_graph)
        metrics = compute_metrics(dg, workflow_graph, cycles)
        node_annotations = compute_node_annotations(workflow_graph, source=source, file_path=file_path)
        risks = compute_structural_risk(
            workflow_graph, metrics, shape, cycles, dg
        )

        return ExecutionModel(
            graph=workflow_graph,
            entry_point=workflow_graph.entry_point,
            terminal_nodes=workflow_graph.terminal_nodes,
            shape=shape,
            cycles=tuple(cycles),
            metrics=metrics,
            node_annotations=node_annotations,
            risks=risks,
        )


def analyze(
    workflow_graph: WorkflowGraph,
    *,
    source: str | None = None,
    file_path: str | None = None,
) -> ExecutionModel:
    """Convenience: run GraphAnalyzer().analyze(workflow_graph, source=..., file_path=...)."""
    return GraphAnalyzer().analyze(workflow_graph, source=source, file_path=file_path)
