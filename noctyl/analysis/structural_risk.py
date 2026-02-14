"""
Structural risk: unreachable nodes, dead ends, non-terminating cycles, multiple entry points.
"""

from __future__ import annotations

from noctyl.graph.execution_model import DetectedCycle, StructuralRisk, StructuralMetrics
from noctyl.graph.graph import WorkflowGraph

from noctyl.analysis.digraph import DirectedGraph


def compute_structural_risk(
    wg: WorkflowGraph,
    metrics: StructuralMetrics,
    shape: str,
    cycles: list[DetectedCycle],
    dg: DirectedGraph,
) -> StructuralRisk:
    """
    Compute StructuralRisk from metrics, shape, cycles, and digraph.
    """
    unreachable_node_ids = tuple(sorted(metrics.unreachable_nodes))

    workflow_node_names = {n.name for n in wg.nodes}
    terminal_set = set(wg.terminal_nodes)
    dead_end_ids = tuple(
        sorted(
            n
            for n in workflow_node_names
            if dg.out_degree(n) == 0 and n not in terminal_set
        )
    )

    non_terminating_cycle_ids = tuple(
        sorted(
            "|".join(c.nodes)  # stable id per cycle
            for c in cycles
            if not c.reaches_terminal
        )
    )

    multiple_entry_points = shape == "invalid"

    return StructuralRisk(
        unreachable_node_ids=unreachable_node_ids,
        dead_end_ids=dead_end_ids,
        non_terminating_cycle_ids=non_terminating_cycle_ids,
        multiple_entry_points=multiple_entry_points,
    )
