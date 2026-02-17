"""
File-level LangGraph detection.

AST-based predicates to identify whether a Python file contains LangGraph
workflow definitions (imports from langgraph.graph and StateGraph instantiation).
Non-LangGraph and invalid files are safely ignored.
"""

from __future__ import annotations

import ast

from noctyl_scout.ingestion.stategraph_tracker import (
    LANGGRAPH_STATEGRAPH_MODULE,
    track_stategraph_instances,
)


def has_langgraph_import(source: str) -> bool:
    """
    Return True if the source contains an import of langgraph.graph.

    Single AST pass over imports only. Use as a fast path to skip files
    with no LangGraph import before running full workflow detection.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == LANGGRAPH_STATEGRAPH_MODULE:
                return True
            continue
        if isinstance(node, ast.Import):
            for alias in node.names or []:
                if alias.name == LANGGRAPH_STATEGRAPH_MODULE:
                    return True
    return False


def file_contains_langgraph(source: str, file_path: str = "") -> bool:
    """
    Return True if the file contains at least one StateGraph instantiation.

    Uses existing track_stategraph_instances; invalid or non-LangGraph
    files yield False (safely ignored).
    """
    instances = track_stategraph_instances(source, file_path)
    return len(instances) > 0
