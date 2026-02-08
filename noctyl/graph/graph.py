"""
Workflow graph data model and deterministic JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass

from noctyl.graph.edges import ExtractedConditionalEdge, ExtractedEdge
from noctyl.graph.nodes import ExtractedNode

DEFAULT_SCHEMA_VERSION = "1.0"


def _compute_terminal_nodes(
    edges: list[ExtractedEdge],
    conditional_edges: list[ExtractedConditionalEdge],
) -> tuple[str, ...]:
    """Node names that have an outgoing edge or conditional edge to END."""
    sources: set[str] = set()
    for e in edges:
        if e.target == "END":
            sources.add(e.source)
    for e in conditional_edges:
        if e.target == "END":
            sources.add(e.source)
    return tuple(sorted(sources))


@dataclass(frozen=True)
class WorkflowGraph:
    """Aggregated workflow graph: nodes, directed edges, entry point, terminal nodes."""

    graph_id: str
    nodes: tuple[ExtractedNode, ...]
    edges: tuple[ExtractedEdge, ...]
    conditional_edges: tuple[ExtractedConditionalEdge, ...]
    entry_point: str | None
    terminal_nodes: tuple[str, ...]
    schema_version: str = DEFAULT_SCHEMA_VERSION


def build_workflow_graph(
    graph_id: str,
    nodes: list[ExtractedNode],
    edges: list[ExtractedEdge],
    conditional_edges: list[ExtractedConditionalEdge],
    entry_point: str | None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> WorkflowGraph:
    """Build a WorkflowGraph from ingestion outputs."""
    terminal_nodes = _compute_terminal_nodes(edges, conditional_edges)
    return WorkflowGraph(
        graph_id=graph_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        conditional_edges=tuple(conditional_edges),
        entry_point=entry_point,
        terminal_nodes=terminal_nodes,
        schema_version=schema_version,
    )


def workflow_graph_to_dict(g: WorkflowGraph) -> dict:
    """
    Return a JSON-serializable dict with deterministic ordering.
    Same WorkflowGraph -> same dict (and same JSON with sort_keys=True).
    """
    nodes_sorted = sorted(g.nodes, key=lambda n: (n.name, n.line))
    edges_sorted = sorted(g.edges, key=lambda e: (e.source, e.target, e.line))
    cond_sorted = sorted(
        g.conditional_edges,
        key=lambda e: (e.source, e.condition_label, e.target, e.line),
    )
    return {
        "schema_version": g.schema_version,
        "graph_id": g.graph_id,
        "entry_point": g.entry_point,
        "terminal_nodes": list(g.terminal_nodes),
        "nodes": [
            {"name": n.name, "callable_ref": n.callable_ref, "line": n.line}
            for n in nodes_sorted
        ],
        "edges": [
            {"source": e.source, "target": e.target, "line": e.line}
            for e in edges_sorted
        ],
        "conditional_edges": [
            {
                "source": e.source,
                "condition_label": e.condition_label,
                "target": e.target,
                "line": e.line,
            }
            for e in cond_sorted
        ],
    }
