"""
Extract LangGraph sequential edges from add_edge(source, target) calls.

Uses AST and tracked StateGraph instances to attribute each add_edge call
to a graph. Source/target are string refs (literal, START/END, or best-effort).
"""

from __future__ import annotations

import ast

from noctyl.graph.edges import ExtractedEdge
from noctyl.ingestion.receiver_resolution import build_alias_map, resolve_receiver
from noctyl.ingestion.stategraph_tracker import TrackedStateGraph


def _edge_arg_to_string(node: ast.expr) -> str:
    """
    Extract source or target as string. Handles literal, START/END, non-literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ("START", "END"):
            return node.id
        return node.id
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


def extract_add_edge_calls(
    source: str,
    file_path: str,
    tracked_graphs: list[TrackedStateGraph],
) -> dict[str, list[ExtractedEdge]]:
    """
    Extract all add_edge(source, target) calls and attribute them to tracked graphs.

    - source: file contents
    - file_path: path for this file (unused for extraction; for future use)
    - tracked_graphs: list of TrackedStateGraph for this file (from track_stategraph_instances)

    Returns dict mapping graph_id to list of ExtractedEdge (source, target, line).
    Only add_edge calls whose receiver resolves to a tracked graph variable are included.
    Non-literal source/target are stringified; missing nodes do not prevent extraction.
    On SyntaxError returns {}.
    """
    result: dict[str, list[ExtractedEdge]] = {
        t.graph_id: [] for t in tracked_graphs
    }

    if not tracked_graphs:
        return result

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    roots: set[str] = set()
    graph_id_by_var: dict[str, str] = {}
    for t in tracked_graphs:
        if t.variable_name is not None:
            roots.add(t.variable_name)
            graph_id_by_var[t.variable_name] = t.graph_id

    if not roots:
        return result

    alias_map = build_alias_map(tree, roots)

    add_edge_calls: list[tuple[str, ast.Call]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_edge":
                continue
            receiver = node.func.value
            root = resolve_receiver(receiver, roots, alias_map)
            if root is None:
                continue
            graph_id = graph_id_by_var.get(root)
            if graph_id is None:
                continue
            if len(node.args) < 2:
                continue
            add_edge_calls.append((graph_id, node))

    for graph_id, call in add_edge_calls:
        source_str = _edge_arg_to_string(call.args[0])
        target_str = _edge_arg_to_string(call.args[1])
        line = getattr(call, "lineno", 0) or 0
        result[graph_id].append(
            ExtractedEdge(source=source_str, target=target_str, line=line)
        )

    for graph_id in result:
        result[graph_id].sort(key=lambda e: e.line)

    return result
