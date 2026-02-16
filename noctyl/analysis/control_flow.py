"""
Control-flow analysis: SCC, cycle detection, cycle classification, termination reachability, graph shape.
"""

from __future__ import annotations

from collections import deque

from noctyl.graph.execution_model import DetectedCycle
from noctyl.graph.graph import WorkflowGraph

from noctyl.analysis.digraph import DirectedGraph

# Cycle types per phase2 §5.1
CYCLE_SELF_LOOP = "self_loop"
CYCLE_MULTI_NODE = "multi_node"
CYCLE_CONDITIONAL = "conditional"
CYCLE_NON_TERMINATING = "non_terminating"

SHAPE_INVALID = "invalid"
SHAPE_DISCONNECTED = "disconnected"
SHAPE_CYCLIC = "cyclic"
SHAPE_BRANCHING = "branching"
SHAPE_LINEAR = "linear"


def _tarjan_scc(dg: DirectedGraph) -> list[set[str]]:
    """Tarjan's algorithm: return list of strongly connected components (each a set of node names)."""
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    sccs: list[set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in dg.successors(v):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: set[str] = set()
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.add(w)
                if w == v:
                    break
            sccs.append(component)

    for node in sorted(dg.nodes()):
        if node not in index:
            strongconnect(node)

    return sccs


def _has_self_loop(dg: DirectedGraph, node: str) -> bool:
    return node in dg.successors(node)


def _cycle_has_conditional_edge(dg: DirectedGraph, cycle_nodes: set[str]) -> bool:
    for u in cycle_nodes:
        for v in dg.successors(u):
            if v in cycle_nodes and dg.is_conditional_edge(u, v):
                return True
    return False


def _can_reach_terminal(
    dg: DirectedGraph,
    from_nodes: set[str],
    terminal_nodes: tuple[str, ...],
) -> bool:
    """BFS from from_nodes; return True if any path reaches a terminal node or END."""
    terminal_set = set(terminal_nodes) | {"END"}
    q: deque[str] = deque(from_nodes)
    seen = set(from_nodes)
    while q:
        n = q.popleft()
        if n in terminal_set:
            return True
        for succ in dg.successors(n):
            if succ not in seen:
                seen.add(succ)
                q.append(succ)
    return False


def _nodes_reachable_from(dg: DirectedGraph, start: str) -> set[str]:
    """BFS from start; return set of reachable nodes."""
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


def _infer_entry_nodes(dg: DirectedGraph) -> set[str]:
    """Nodes with no incoming edge from any other node (excluding START)."""
    entries: set[str] = set()
    for n in dg.nodes():
        if n == "START" or n == "END":
            continue
        preds = dg.predecessors(n)
        # Only START or no predecessors (disconnected node)
        internal_preds = [p for p in preds if p != "START"]
        if not internal_preds:
            entries.add(n)
    return entries


def compute_control_flow(
    dg: DirectedGraph,
    wg: WorkflowGraph,
) -> tuple[str, list[DetectedCycle]]:
    """
    Compute graph shape and list of detected cycles (each with type and reaches_terminal).
    Returns (shape, cycles) with cycles having sorted node tuples for determinism.
    """
    terminal_nodes = wg.terminal_nodes
    workflow_nodes = {n.name for n in wg.nodes}
    # Exclude START/END from "internal" node set for shape
    internal_nodes = workflow_nodes | {"START", "END"}

    # --- Shape: invalid (multiple entry points) ---
    entries = _infer_entry_nodes(dg)
    if wg.entry_point is None and workflow_nodes:
        shape = SHAPE_INVALID
    elif len(entries) > 1:
        shape = SHAPE_INVALID
    else:
        shape = None  # compute below

    # --- SCC and cycles ---
    sccs = _tarjan_scc(dg)
    cycles: list[DetectedCycle] = []
    nodes_in_cycle: set[str] = set()

    for scc in sccs:
        # Only consider SCCs that are actually cycles: size 1 with self-loop, or size > 1
        if len(scc) == 1:
            node = next(iter(scc))
            if not _has_self_loop(dg, node):
                continue
            cycle_nodes = tuple(sorted(scc))
            nodes_in_cycle |= scc
            has_cond = dg.is_conditional_edge(node, node)
            reaches = _can_reach_terminal(dg, scc, terminal_nodes)
            cycle_type = CYCLE_CONDITIONAL if has_cond else CYCLE_SELF_LOOP
            if not reaches:
                cycle_type = CYCLE_NON_TERMINATING
            cycles.append(
                DetectedCycle(cycle_type=cycle_type, nodes=cycle_nodes, reaches_terminal=reaches)
            )
        else:
            cycle_nodes = tuple(sorted(scc))
            nodes_in_cycle |= scc
            has_cond = _cycle_has_conditional_edge(dg, scc)
            reaches = _can_reach_terminal(dg, scc, terminal_nodes)
            cycle_type = CYCLE_CONDITIONAL if has_cond else CYCLE_MULTI_NODE
            if not reaches:
                cycle_type = CYCLE_NON_TERMINATING
            cycles.append(
                DetectedCycle(cycle_type=cycle_type, nodes=cycle_nodes, reaches_terminal=reaches)
            )

    # Sort cycles for determinism
    cycles.sort(key=lambda c: (c.cycle_type, c.nodes))

    # --- Shape (if not already invalid) ---
    if shape is None:
        reachable = _nodes_reachable_from(dg, "START")
        internal_reachable = reachable & workflow_nodes
        if workflow_nodes and (workflow_nodes - internal_reachable):
            shape = SHAPE_DISCONNECTED
        elif cycles:
            shape = SHAPE_CYCLIC
        else:
            # Check branching: any node with out_degree > 1
            branching = any(dg.out_degree(n) > 1 for n in dg.nodes() if n != "END")
            shape = SHAPE_BRANCHING if branching else SHAPE_LINEAR

    return (shape, cycles)
