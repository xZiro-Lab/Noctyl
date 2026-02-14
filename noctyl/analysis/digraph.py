"""
Build a directed graph from a WorkflowGraph for control-flow and metrics.
Nodes = workflow node names + START + END. Edges from edges, conditional_edges, and START -> entry_point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noctyl.graph.graph import WorkflowGraph


@dataclass
class DirectedGraph:
    """
    Adjacency-list directed graph with optional conditional-edge set.
    All node names are strings; START and END are included when used.
    """

    _successors: dict[str, list[str]] = field(default_factory=dict)
    _predecessors: dict[str, list[str]] = field(default_factory=dict)
    _conditional_edges: set[tuple[str, str]] = field(default_factory=set)

    def nodes(self) -> set[str]:
        """All node names."""
        return set(self._successors.keys()) | set(self._predecessors.keys())

    def successors(self, node: str) -> list[str]:
        """List of targets of edges from node (order preserved)."""
        return list(self._successors.get(node, []))

    def predecessors(self, node: str) -> list[str]:
        """List of sources of edges into node."""
        return list(self._predecessors.get(node, []))

    def out_degree(self, node: str) -> int:
        return len(self._successors.get(node, []))

    def is_conditional_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._conditional_edges

    def add_edge(self, source: str, target: str, conditional: bool = False) -> None:
        if source not in self._successors:
            self._successors[source] = []
        self._successors[source].append(target)
        if target not in self._predecessors:
            self._predecessors[target] = []
        self._predecessors[target].append(source)
        if conditional:
            self._conditional_edges.add((source, target))


def build_digraph(wg: WorkflowGraph) -> DirectedGraph:
    """
    Build a DirectedGraph from a WorkflowGraph.
    Nodes: all node names from wg.nodes plus START and END when they appear.
    Edges: each ExtractedEdge and ExtractedConditionalEdge; plus START -> entry_point when set.
    """
    dg = DirectedGraph()
    node_names = {n.name for n in wg.nodes}

    for e in wg.edges:
        dg.add_edge(e.source, e.target, conditional=False)
    for e in wg.conditional_edges:
        dg.add_edge(e.source, e.target, conditional=True)

    if wg.entry_point is not None:
        dg.add_edge("START", wg.entry_point, conditional=False)

    # Ensure START and END are in the node set if they appear in edges
    for n in node_names:
        if n not in dg._successors:
            dg._successors[n] = []
        if n not in dg._predecessors:
            dg._predecessors[n] = []
    if "START" not in dg._successors and "START" not in dg._predecessors:
        dg._successors["START"] = []
        dg._predecessors["START"] = []
    if "END" not in dg._successors and "END" not in dg._predecessors:
        dg._successors["END"] = []
        dg._predecessors["END"] = []

    return dg
