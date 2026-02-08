"""
Same-file receiver resolution for attributing method calls to tracked StateGraph instances.

Used by node_extractor and edge_extractor to resolve x.add_node(...) / x.add_edge(...)
to the correct graph_id when x may be an alias of a tracked graph variable.
"""

from __future__ import annotations

import ast


def build_alias_map(
    tree: ast.AST,
    roots: set[str],
) -> dict[str, str]:
    """
    Build same-file alias map: name -> root variable name.
    Roots are tracked StateGraph variable names. Only handles x = y (y is Name).
    Assignments are processed in document order so chained aliases resolve correctly.
    """
    alias_map: dict[str, str] = {}
    assigns: list[tuple[int, str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                line = getattr(node, "lineno", 0) or 0
                assigns.append((line, target.id, node.value.id))

    for _line, lhs, rhs in sorted(assigns, key=lambda t: t[0]):
        root = rhs if rhs in roots else alias_map.get(rhs)
        if root is not None:
            alias_map[lhs] = root

    return alias_map


def resolve_receiver(
    receiver: ast.expr,
    roots: set[str],
    alias_map: dict[str, str],
) -> str | None:
    """Resolve receiver to a root graph variable name, or None."""
    if not isinstance(receiver, ast.Name):
        return None
    name = receiver.id
    if name in roots:
        return name
    return alias_map.get(name)
