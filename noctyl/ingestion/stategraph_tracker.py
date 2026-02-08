"""
Identify and track variables that hold StateGraph instances.

Uses AST-only analysis; supports multiple StateGraph instances per file
and assigns stable internal IDs per instance.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Literal


# Canonical LangGraph StateGraph origin (we only count this).
LANGGRAPH_STATEGRAPH_MODULE = "langgraph.graph"
LANGGRAPH_STATEGRAPH_NAME = "StateGraph"


@dataclass(frozen=True)
class TrackedStateGraph:
    """A single detected StateGraph instantiation."""

    graph_id: str
    file_path: str
    line: int
    variable_name: str | None
    binding_kind: Literal["assignment", "inline"]  # extensible later


def _normalize_path(path: str) -> str:
    """Normalize file path for stable IDs: separators, no . or .. segments."""
    p = os.path.normpath(path)
    if os.sep != "/":
        p = p.replace(os.sep, "/")
    return p.lstrip("/") or path


def _is_langgraph_stategraph(
    node: ast.expr,
    aliases: dict[str, str],
    has_langgraph_module: bool,
) -> bool:
    """
    Return True if node resolves to langgraph.graph.StateGraph.

    - Name: check aliases (from langgraph.graph import StateGraph as X)
    - Attribute: check langgraph.graph.StateGraph
    """
    if isinstance(node, ast.Name):
        origin = aliases.get(node.id)
        return origin == f"{LANGGRAPH_STATEGRAPH_MODULE}.{LANGGRAPH_STATEGRAPH_NAME}"
    if isinstance(node, ast.Attribute):
        if not has_langgraph_module:
            return False
        # langgraph.graph.StateGraph
        try:
            if isinstance(node.value, ast.Attribute):
                mod = node.value
                if isinstance(mod.value, ast.Name):
                    return (
                        mod.value.id == "langgraph"
                        and mod.attr == "graph"
                        and node.attr == LANGGRAPH_STATEGRAPH_NAME
                    )
        except Exception:
            pass
        return False
    return False


def _collect_imports(tree: ast.AST) -> tuple[dict[str, str], bool]:
    """
    Collect name -> canonical origin for StateGraph, and whether langgraph.graph is imported.

    Returns (aliases, has_langgraph_module).
    - aliases: local_name -> "langgraph.graph.StateGraph" when from langgraph.graph import StateGraph [as X]
    - has_langgraph_module: True if "import langgraph.graph" exists (so we can resolve langgraph.graph.StateGraph)
    """
    aliases: dict[str, str] = {}
    has_langgraph_module = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == LANGGRAPH_STATEGRAPH_MODULE:
                for alias in node.names or []:
                    if alias.name == LANGGRAPH_STATEGRAPH_NAME:
                        local = alias.asname or alias.name
                        aliases[local] = f"{LANGGRAPH_STATEGRAPH_MODULE}.{LANGGRAPH_STATEGRAPH_NAME}"
            continue
        if isinstance(node, ast.Import):
            for alias in node.names or []:
                if alias.name == LANGGRAPH_STATEGRAPH_MODULE:
                    has_langgraph_module = True
                    break

    return aliases, has_langgraph_module


def _variable_from_assignment(value_node: ast.expr, assign: ast.Assign) -> str | None:
    """
    If value_node is the RHS of assign (or an element of a tuple RHS), return the
    corresponding target name (if simple).
    """
    targets = assign.targets
    if not targets:
        return None
    t = targets[0]

    if assign.value is value_node:
        # Single RHS: x = StateGraph(...)
        if isinstance(t, ast.Name):
            return t.id
        if isinstance(t, ast.Tuple) and t.elts:
            first = t.elts[0]
            if isinstance(first, ast.Name):
                return first.id
        return None

    if isinstance(assign.value, ast.Tuple):
        # a, b = StateGraph(...), StateGraph(...)
        for idx, elt in enumerate(assign.value.elts):
            if elt is value_node:
                if isinstance(t, ast.Tuple) and idx < len(t.elts):
                    target_elt = t.elts[idx]
                    if isinstance(target_elt, ast.Name):
                        return target_elt.id
                break
    return None


def _find_binding(
    call_node: ast.Call,
    tree: ast.AST,
) -> tuple[str | None, Literal["assignment", "inline"]]:
    """
    Find variable name and binding_kind for this StateGraph(...) call.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            var_name = _variable_from_assignment(call_node, node)
            if var_name is not None:
                return (var_name, "assignment")
    return (None, "inline")


def _collect_stategraph_calls(
    tree: ast.AST,
    aliases: dict[str, str],
    has_langgraph_module: bool,
) -> list[tuple[ast.Call, int, int]]:
    """Collect (Call node, line, col) for every StateGraph(...) call. No ordering yet."""
    calls: list[tuple[ast.Call, int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_langgraph_stategraph(
                node.func, aliases, has_langgraph_module
            ):
                line = getattr(node, "lineno", 0) or 0
                col = getattr(node, "col_offset", 0) or 0
                calls.append((node, line, col))

    return calls


def track_stategraph_instances(
    source: str,
    file_path: str,
) -> list[TrackedStateGraph]:
    """
    Parse Python source and return all tracked StateGraph instances.

    - source: file contents (or snippet).
    - file_path: path used for graph_id (will be normalized).

    Returns list of TrackedStateGraph, one per StateGraph(...) call, ordered by
    (line, column). graph_id format: {normalized_file_path}:{ordinal}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    aliases, has_langgraph_module = _collect_imports(tree)
    calls = _collect_stategraph_calls(tree, aliases, has_langgraph_module)
    # Sort by (line, column) for stable ordinal
    calls.sort(key=lambda c: (c[1], c[2]))

    normalized_path = _normalize_path(file_path)
    result: list[TrackedStateGraph] = []

    for ordinal, (call_node, line, _col) in enumerate(calls):
        var_name, binding_kind = _find_binding(call_node, tree)
        graph_id = f"{normalized_path}:{ordinal}"
        result.append(
            TrackedStateGraph(
                graph_id=graph_id,
                file_path=normalized_path,
                line=line,
                variable_name=var_name,
                binding_kind=binding_kind,
            )
        )

    return result
