"""
Structural metrics: basic counts, unreachable nodes, longest acyclic path, branching factor, max depth before cycle.
"""

from __future__ import annotations

from noctyl.graph.execution_model import StructuralMetrics
from noctyl.graph.graph import WorkflowGraph

from noctyl.analysis.digraph import DirectedGraph


def _reachable_from(dg: DirectedGraph, start: str) -> set[str]:
    """BFS from start."""
    from collections import deque

    q: deque[str] = deque([start])
    seen: set[str] = set()
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        for succ in dg.successors(n):
            q.append(succ)
    return seen


def _longest_simple_path_from(dg: DirectedGraph, start: str) -> int:
    """Longest path from start without revisiting a node (DFS)."""
    best = [0]

    def dfs(node: str, visited: set[str], depth: int) -> None:
        if depth > best[0]:
            best[0] = depth
        for succ in dg.successors(node):
            if succ not in visited:
                visited.add(succ)
                dfs(succ, visited, depth + 1)
                visited.discard(succ)

    visited: set[str] = {start}
    dfs(start, visited, 0)
    return best[0]


def _max_depth_before_cycle(
    dg: DirectedGraph,
    start: str,
    cycle_nodes: set[str],
) -> int | None:
    """From start, DFS; when first entering a cycle node, record depth; return max such depth."""
    if not cycle_nodes:
        return None
    best = [None]

    def dfs(node: str, visited: set[str], depth: int) -> None:
        if node in cycle_nodes:
            if best[0] is None or depth > best[0]:
                best[0] = depth
        for succ in dg.successors(node):
            if succ not in visited:
                visited.add(succ)
                dfs(succ, visited, depth + 1)
                visited.discard(succ)

    visited: set[str] = {start}
    dfs(start, visited, 0)
    return best[0]


def compute_metrics(
    dg: DirectedGraph,
    wg: WorkflowGraph,
    cycles: list,
) -> StructuralMetrics:
    """
    Compute StructuralMetrics from digraph and workflow graph.
    cycles: list of DetectedCycle (used to get cycle node set for max_depth_before_cycle).
    """
    workflow_node_names = {n.name for n in wg.nodes}
    node_count = len(wg.nodes)
    edge_count = len(wg.edges) + len(wg.conditional_edges)
    entry_node = wg.entry_point
    terminal_nodes = wg.terminal_nodes

    reachable = _reachable_from(dg, "START")
    unreachable_nodes = tuple(
        sorted(workflow_node_names - reachable)
    )

    # Longest simple path from START (number of edges = path length in "steps")
    start_node = "START"
    if wg.entry_point is not None:
        start_node = wg.entry_point
    longest_acyclic_path = _longest_simple_path_from(dg, start_node)

    # Average branching factor: sum of out-degrees / num workflow nodes
    if workflow_node_names:
        total_out = sum(dg.out_degree(n) for n in workflow_node_names)
        avg_branching_factor = total_out / len(workflow_node_names)
    else:
        avg_branching_factor = 0.0

    cycle_nodes: set[str] = set()
    for c in cycles:
        cycle_nodes.update(c.nodes)
    max_depth_before_cycle = _max_depth_before_cycle(dg, "START", cycle_nodes)

    return StructuralMetrics(
        node_count=node_count,
        edge_count=edge_count,
        entry_node=entry_node,
        terminal_nodes=terminal_nodes,
        unreachable_nodes=unreachable_nodes,
        longest_acyclic_path=longest_acyclic_path,
        avg_branching_factor=round(avg_branching_factor, 2),
        max_depth_before_cycle=max_depth_before_cycle,
    )
