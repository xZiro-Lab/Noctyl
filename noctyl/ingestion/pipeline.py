"""
Safe pipeline runner: discover files, read, ingest, build WorkflowGraph, serialize.
Never raises for invalid or unreadable files; collects warnings.
"""

from __future__ import annotations

from pathlib import Path

from noctyl.graph import build_workflow_graph, workflow_graph_to_dict
from noctyl.ingestion.edge_extractor import (
    extract_add_conditional_edges,
    extract_add_edge_calls,
    extract_entry_points,
)
from noctyl.ingestion.langgraph_detector import file_contains_langgraph
from noctyl.ingestion.node_extractor import extract_add_node_calls
from noctyl.ingestion.repo_scanner import discover_python_files
from noctyl.ingestion.stategraph_tracker import track_stategraph_instances


def run_pipeline_on_directory(
    root_path: Path | str,
) -> tuple[list[dict], list[str]]:
    """
    Run the full pipeline on a directory: discover .py files, read each, ingest,
    build WorkflowGraph per graph, serialize to dict. Does not raise for
    unreadable or invalid files; skips with a warning.

    Returns:
        (list of workflow dicts from workflow_graph_to_dict, list of warning strings).
    """
    root = Path(root_path).resolve()
    paths = discover_python_files(root)
    results: list[dict] = []
    warnings: list[str] = []

    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            warnings.append(f"{path}: could not read")
            continue

        file_path = str(path.relative_to(root))
        if not file_contains_langgraph(source, file_path):
            continue

        tracked = track_stategraph_instances(source, file_path)
        entry_by_graph, entry_warnings = extract_entry_points(source, file_path, tracked)
        for w in entry_warnings:
            warnings.append(w)

        for t in tracked:
            nodes = extract_add_node_calls(source, file_path, tracked).get(t.graph_id, [])
            edges = extract_add_edge_calls(source, file_path, tracked).get(t.graph_id, [])
            cond = extract_add_conditional_edges(source, file_path, tracked).get(
                t.graph_id, []
            )
            entry_point = entry_by_graph.get(t.graph_id)
            wg = build_workflow_graph(
                t.graph_id, nodes, edges, cond, entry_point
            )
            results.append(workflow_graph_to_dict(wg))

    return (results, warnings)
