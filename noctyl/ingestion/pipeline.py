"""
Safe pipeline runner: discover files, read, ingest, build WorkflowGraph, serialize.
Never raises for invalid or unreadable files; collects warnings.
"""

from __future__ import annotations

from pathlib import Path

from noctyl.analysis import analyze
from noctyl.estimation import (
    ModelProfile,
    TokenModeler,
    load_model_profile,
    workflow_estimate_to_dict,
)
from noctyl.graph import build_workflow_graph, execution_model_to_dict, workflow_graph_to_dict
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
    *,
    enriched: bool = False,
    estimate: bool = False,
    profile: ModelProfile | str | Path | dict | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Run the full pipeline on a directory: discover .py files, read each, ingest,
    build WorkflowGraph per graph, serialize to dict. Does not raise for
    unreadable or invalid files; skips with a warning.

    Output modes:
    - estimate=False, enriched=False (default): schema_version 1.0 (Phase 1)
    - estimate=False, enriched=True: schema_version 2.0 (Phase 2, enriched)
    - estimate=True: schema_version 3.0 (Phase 3, estimated)
    
    When estimate=True, enriched is automatically set to True (Phase 3 requires Phase 2 data).

    Args:
        root_path: Directory or file path to process
        enriched: If True, include Phase 2 analysis (cycles, shape, metrics, etc.)
        estimate: If True, include Phase 3 token estimation (requires enriched=True)
        profile: Model profile for estimation. Can be:
            - ModelProfile instance
            - str/Path: YAML file path
            - dict: Profile dict
            - None: Use default profile
    
    Returns:
        (list of workflow dicts, list of warning strings).
    """
    root = Path(root_path).resolve()
    paths = discover_python_files(root)
    results: list[dict] = []
    warnings: list[str] = []
    
    # If estimate=True, automatically enable enriched=True (Phase 3 requires Phase 2)
    if estimate:
        enriched = True
    
    # Load model profile if estimate mode
    model_profile = None
    token_modeler = None
    if estimate:
        try:
            model_profile = load_model_profile(profile)
        except Exception as e:
            warnings.append(f"Failed to load model profile: {e}. Using default profile.")
            model_profile = load_model_profile(None)
        token_modeler = TokenModeler()

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
            
            if estimate:
                # Phase 3: Estimate mode
                # 1. Build WorkflowGraph (Phase 1) - already done
                # 2. Analyze via GraphAnalyzer → ExecutionModel (Phase 2)
                execution_model = analyze(wg, source=source, file_path=file_path)
                # 3. Estimate via TokenModeler → WorkflowEstimate (Phase 3)
                workflow_estimate = token_modeler.estimate(
                    execution_model, model_profile, source=source, file_path=file_path
                )
                # 4. Serialize via workflow_estimate_to_dict() → schema 3.0 dict
                result_dict = workflow_estimate_to_dict(workflow_estimate)
                results.append(result_dict)
                # Merge TokenModeler warnings
                for w in workflow_estimate.warnings:
                    warnings.append(w)
            elif enriched:
                # Phase 2: Enriched mode
                results.append(
                    execution_model_to_dict(
                        analyze(wg, source=source, file_path=file_path)
                    )
                )
            else:
                # Phase 1: Basic mode
                results.append(workflow_graph_to_dict(wg))

    return (results, warnings)
