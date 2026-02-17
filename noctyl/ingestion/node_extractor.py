"""
Extract LangGraph nodes from add_node(name, callable) calls.

Uses AST and tracked StateGraph instances to attribute each add_node call
to a graph and extract node name and callable reference (string only).
"""

from __future__ import annotations

import ast

from noctyl_scout.graph.nodes import ExtractedNode
from noctyl_scout.ingestion.receiver_resolution import build_alias_map, resolve_receiver
from noctyl_scout.ingestion.stategraph_tracker import TrackedStateGraph


def _node_name_from_arg(node: ast.expr) -> str:
    """Extract node name string from first positional arg."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    try:
        return ast.unparse(node)
    except Exception:
        return repr(node)


def _callable_ref_from_arg(node: ast.expr) -> str:
    """Extract callable reference as string only (no inspection of internals)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except Exception:
            return "<attribute>"
    if isinstance(node, ast.Lambda):
        return "lambda"
    return "<unknown>"


def extract_add_node_calls(
    source: str,
    file_path: str,
    tracked_graphs: list[TrackedStateGraph],
) -> dict[str, list[ExtractedNode]]:
    """
    Extract all add_node(name, callable) calls and attribute them to tracked graphs.

    - source: file contents
    - file_path: path for this file (unused for extraction; for future use)
    - tracked_graphs: list of TrackedStateGraph for this file (from track_stategraph_instances)

    Returns dict mapping graph_id to list of ExtractedNode (name, callable_ref, line).
    Only add_node calls whose receiver resolves to a tracked graph variable are included.
    On SyntaxError returns {}.
    """
    result: dict[str, list[ExtractedNode]] = {
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

    add_node_calls: list[tuple[str, ast.Call]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_node":
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
            add_node_calls.append((graph_id, node))

    for graph_id, call in add_node_calls:
        name = _node_name_from_arg(call.args[0])
        callable_ref = _callable_ref_from_arg(call.args[1])
        line = getattr(call, "lineno", 0) or 0
        result[graph_id].append(
            ExtractedNode(name=name, callable_ref=callable_ref, line=line)
        )

    for graph_id in result:
        result[graph_id].sort(key=lambda n: n.line)

    return result
