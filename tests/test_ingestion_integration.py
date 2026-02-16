"""
High-level integration tests: detector, tracker, node extractor, and edge extractor in sync.

One source file is run through the full ingestion pipeline; we assert all components
agree on graph presence, graph_id, and that nodes/edges belong to the same graph.
Integration with graph schema: ingestion outputs -> WorkflowGraph -> deterministic JSON.
Integration with repo scanner: discover_python_files -> per-file ingestion -> WorkflowGraph.
"""

import json
import tempfile
from pathlib import Path

from noctyl.graph import build_workflow_graph, workflow_graph_to_dict
from noctyl.ingestion import (
    discover_python_files,
    extract_add_conditional_edges,
    extract_add_edge_calls,
    extract_add_node_calls,
    extract_entry_points,
    file_contains_langgraph,
    has_langgraph_import,
    run_pipeline_on_directory,
    track_stategraph_instances,
)


def test_full_system_sync_all_components():
    """
    Full system sync: detector, tracker, nodes, sequential edges, and conditional edges
    all run on one workflow; same graph_id everywhere; nodes/edges/conditional edges consistent.
    """
    source = """
from langgraph.graph import StateGraph, START, END

def agent(s): return s
def tool(s): return s
def router(s): return "continue" if s else "stop"

graph = StateGraph(dict)
graph.add_node("agent", agent)
graph.add_node("tool", tool)
graph.add_edge(START, "agent")
graph.add_edge("agent", "tool")
graph.add_conditional_edges("tool", router, {"continue": "agent", "stop": END})
"""
    file_path = "app/full_workflow.py"

    # 1. Detector
    assert has_langgraph_import(source) is True
    assert file_contains_langgraph(source, file_path) is True

    # 2. Tracker: one graph
    tracked = track_stategraph_instances(source, file_path)
    assert len(tracked) == 1
    graph_id = tracked[0].graph_id

    # 3. Nodes: same graph_id, expected names
    nodes_by_graph = extract_add_node_calls(source, file_path, tracked)
    assert set(nodes_by_graph.keys()) == {graph_id}
    nodes = nodes_by_graph[graph_id]
    node_names = {n.name for n in nodes}
    assert node_names == {"agent", "tool"}

    # 4. Sequential edges: same graph_id
    edges_by_graph = extract_add_edge_calls(source, file_path, tracked)
    assert set(edges_by_graph.keys()) == {graph_id}
    edges = edges_by_graph[graph_id]
    assert len(edges) >= 2
    edge_sources = {e.source for e in edges}
    edge_targets = {e.target for e in edges}
    assert "START" in edge_sources
    assert "agent" in edge_sources or "agent" in edge_targets
    assert "tool" in edge_targets or "tool" in edge_sources

    # 5. Conditional edges: same graph_id
    cond_by_graph = extract_add_conditional_edges(source, file_path, tracked)
    assert set(cond_by_graph.keys()) == {graph_id}
    cond_edges = cond_by_graph[graph_id]
    assert len(cond_edges) == 2
    cond_labels = {e.condition_label for e in cond_edges}
    assert cond_labels == {"continue", "stop"}
    for ce in cond_edges:
        assert ce.source == "tool"
        assert ce.target in ("agent", "END")

    # 6. Cross-component consistency: conditional source is a node; targets are node or END
    for ce in cond_edges:
        assert ce.source in node_names
        assert ce.target in node_names or ce.target == "END"
    for e in edges:
        assert e.source in node_names or e.source == "START"
        assert e.target in node_names or e.target == "END"

    # 7. Entry point: detected or inferred (from add_edge(START, "agent"))
    entry_by_graph, entry_warnings = extract_entry_points(source, file_path, tracked)
    assert entry_by_graph.get(graph_id) == "agent"
    assert entry_warnings == []


def test_ingestion_to_workflow_graph_and_deterministic_json():
    """
    Full integration: ingestion outputs -> WorkflowGraph -> JSON.
    Asserts build_workflow_graph assembles correctly, workflow_graph_to_dict matches
    schema shape, and serialization is deterministic (same JSON twice).
    """
    source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"next": "a", "done": END})
"""
    file_path = "app/workflow.py"
    tracked = track_stategraph_instances(source, file_path)
    assert len(tracked) == 1
    graph_id = tracked[0].graph_id

    nodes = extract_add_node_calls(source, file_path, tracked)[graph_id]
    edges = extract_add_edge_calls(source, file_path, tracked)[graph_id]
    cond_edges = extract_add_conditional_edges(source, file_path, tracked)[graph_id]
    entry_by_graph, _ = extract_entry_points(source, file_path, tracked)
    entry_point = entry_by_graph.get(graph_id)

    wg = build_workflow_graph(
        graph_id, nodes, edges, cond_edges, entry_point
    )
    assert wg.graph_id == graph_id
    assert wg.entry_point == "a"
    assert len(wg.nodes) == 2
    assert len(wg.edges) == 2
    assert len(wg.conditional_edges) == 2

    d = workflow_graph_to_dict(wg)
    assert d["schema_version"] == "1.0"
    assert d["graph_id"] == graph_id
    assert d["entry_point"] == "a"
    assert "terminal_nodes" in d
    assert "b" in d["terminal_nodes"], "b has conditional edge to END"
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 2
    assert len(d["conditional_edges"]) == 2
    node_names = {n["name"] for n in d["nodes"]}
    assert node_names == {"a", "b"}

    json1 = json.dumps(d, sort_keys=True)
    d2 = workflow_graph_to_dict(wg)
    json2 = json.dumps(d2, sort_keys=True)
    assert json1 == json2, "serialization must be deterministic"

    parsed = json.loads(json1)
    assert parsed["entry_point"] == "a"
    assert "b" in parsed["terminal_nodes"]
    assert len(parsed["nodes"]) == 2
    assert len(parsed["conditional_edges"]) == 2


def test_full_pipeline_one_workflow_in_sync():
    """
    Full ingestion pipeline on one workflow: detector, tracker, nodes, edges all agree.
    - File contains LangGraph -> detector True
    - One tracked graph -> one graph_id
    - Nodes and edges both keyed by that graph_id; edge source/target match node names
    """
    source = """
from langgraph.graph import StateGraph

def agent(state):
    return state

def tool(state):
    return state

graph = StateGraph(dict)
graph.add_node("agent", agent)
graph.add_node("tool", tool)
graph.add_edge("agent", "tool")
"""
    file_path = "app/workflow.py"

    # 1. Detector: file contains LangGraph
    assert has_langgraph_import(source) is True
    assert file_contains_langgraph(source, file_path) is True

    # 2. Tracker: exactly one StateGraph instance
    tracked = track_stategraph_instances(source, file_path)
    assert len(tracked) == 1
    graph_id = tracked[0].graph_id
    assert tracked[0].variable_name == "graph"

    # 3. Node extractor: nodes for that graph_id
    nodes_by_graph = extract_add_node_calls(source, file_path, tracked)
    assert set(nodes_by_graph.keys()) == {graph_id}
    nodes = nodes_by_graph[graph_id]
    node_names = {n.name for n in nodes}
    assert node_names == {"agent", "tool"}

    # 4. Edge extractor: edges for same graph_id
    edges_by_graph = extract_add_edge_calls(source, file_path, tracked)
    assert set(edges_by_graph.keys()) == {graph_id}
    edges = edges_by_graph[graph_id]
    assert len(edges) == 1
    assert edges[0].source == "agent" and edges[0].target == "tool"

    # 5. Consistency: edge endpoints are extracted node names
    assert edges[0].source in node_names
    assert edges[0].target in node_names


def test_full_pipeline_no_langgraph_all_empty():
    """No LangGraph in file -> detector False, tracker empty, nodes/edges empty."""
    source = "def main(): return 42"
    file_path = "other.py"

    assert has_langgraph_import(source) is False
    assert file_contains_langgraph(source, file_path) is False

    tracked = track_stategraph_instances(source, file_path)
    assert tracked == []

    nodes_by_graph = extract_add_node_calls(source, file_path, tracked)
    assert nodes_by_graph == {}

    edges_by_graph = extract_add_edge_calls(source, file_path, tracked)
    assert edges_by_graph == {}


def test_full_pipeline_multiple_graphs_same_file():
    """Two graphs in one file -> two graph_ids; nodes and edges attributed to correct graph."""
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
a.add_node("n1", f1)
a.add_edge("n1", "n1")
b.add_node("n2", f2)
b.add_edge("n2", "n2")
"""
    file_path = "multi.py"

    assert file_contains_langgraph(source, file_path) is True
    tracked = track_stategraph_instances(source, file_path)
    assert len(tracked) == 2
    graph_ids = sorted(tracked, key=lambda t: t.graph_id)
    g0, g1 = graph_ids[0].graph_id, graph_ids[1].graph_id
    assert g0 != g1

    nodes_by_graph = extract_add_node_calls(source, file_path, tracked)
    edges_by_graph = extract_add_edge_calls(source, file_path, tracked)

    assert set(nodes_by_graph.keys()) == {g0, g1}
    assert set(edges_by_graph.keys()) == {g0, g1}

    # Each graph has one node and one edge; node names and edge endpoints match per graph
    for gid in (g0, g1):
        nodes = nodes_by_graph[gid]
        edges = edges_by_graph[gid]
        assert len(nodes) == 1
        assert len(edges) == 1
        assert edges[0].source == nodes[0].name
        assert edges[0].target == nodes[0].name


def test_full_pipeline_with_conditional_edges():
    """Pipeline with add_conditional_edges: conditional edges extracted for same graph_id."""
    source = """
from langgraph.graph import StateGraph, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_conditional_edges("a", router, {"go": "b", "stop": END})
"""
    file_path = "cond.py"
    assert file_contains_langgraph(source, file_path) is True
    tracked = track_stategraph_instances(source, file_path)
    assert len(tracked) == 1
    graph_id = tracked[0].graph_id

    cond_by_graph = extract_add_conditional_edges(source, file_path, tracked)
    assert set(cond_by_graph.keys()) == {graph_id}
    cond_edges = cond_by_graph[graph_id]
    assert len(cond_edges) == 2
    by_label = {e.condition_label: e for e in cond_edges}
    assert by_label["go"].source == "a" and by_label["go"].target == "b"
    assert by_label["stop"].source == "a" and by_label["stop"].target == "END"


def test_repo_scanner_to_ingestion_to_workflow_graph():
    """
    Full integration: discover_python_files on a temp tree -> read each file ->
    run ingestion (track, nodes, edges, conditional_edges, entry_point) ->
    build WorkflowGraph and serialize to JSON.
    Asserts default ignores exclude tests/; only discovered files are analyzed;
    one workflow file yields one graph with expected structure.
    """
    workflow_source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"next": "a", "done": END})
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "workflow.py").write_text(workflow_source)
        (root / "tests").mkdir()
        (root / "tests" / "test_workflow.py").write_text(
            "from langgraph.graph import StateGraph\nx = StateGraph(dict)\n"
        )

        paths = discover_python_files(root)
        rel_paths = [p.relative_to(root) for p in paths]
        assert any(p.name == "workflow.py" and "src" in p.parts for p in rel_paths), "src/workflow.py should be discovered"
        assert all("tests" not in p.parts for p in rel_paths), "tests/ should be ignored by default"

        all_graphs = []
        for path in paths:
            source = path.read_text()
            file_path = str(path.relative_to(root))
            if not file_contains_langgraph(source, file_path):
                continue
            tracked = track_stategraph_instances(source, file_path)
            for t in tracked:
                nodes = extract_add_node_calls(source, file_path, tracked).get(t.graph_id, [])
                edges = extract_add_edge_calls(source, file_path, tracked).get(t.graph_id, [])
                cond = extract_add_conditional_edges(source, file_path, tracked).get(t.graph_id, [])
                entry_by_graph, _ = extract_entry_points(source, file_path, tracked)
                entry_point = entry_by_graph.get(t.graph_id)
                wg = build_workflow_graph(t.graph_id, nodes, edges, cond, entry_point)
                all_graphs.append(workflow_graph_to_dict(wg))

        assert len(all_graphs) >= 1
        d = all_graphs[0]
        assert d["entry_point"] == "a"
        assert "terminal_nodes" in d and "b" in d["terminal_nodes"]
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 2
        assert len(d["conditional_edges"]) == 2
        assert d["schema_version"] == "1.0"
        json_out = json.dumps(d, sort_keys=True)
        assert "a" in json_out and "b" in json_out


def test_run_pipeline_on_directory_unreadable_file_does_not_crash():
    """
    run_pipeline_on_directory does not crash when a .py file cannot be read
    (e.g. invalid UTF-8). Returns graphs from valid files and a warning for the bad file.
    """
    workflow_source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "workflow.py").write_text(workflow_source)
        (root / "src" / "bad.py").write_bytes(b"\xff\xfe invalid utf-8")

        results, warnings = run_pipeline_on_directory(root)

        assert len(results) >= 1
        assert results[0]["entry_point"] == "a"
        assert "terminal_nodes" in results[0]
        assert any("could not read" in w for w in warnings)


def test_run_pipeline_on_directory_syntax_error_only_does_not_crash():
    """
    run_pipeline_on_directory on a dir with only a syntax-error file returns
    empty results and does not crash.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "broken.py").write_text("from langgraph.graph import StateGraph\nx = (  # unclosed")

        results, warnings = run_pipeline_on_directory(root)

        assert results == []
        # May have warnings or not; must not raise


def test_run_pipeline_on_directory_enriched_contains_phase2_fields():
    """
    Enriched mode returns schema 2.0 dicts with Phase-2 fields while preserving base graph fields.
    """
    workflow_source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "workflow.py").write_text(workflow_source)

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert warnings == []
        assert len(results) >= 1
        d = results[0]
        assert d["schema_version"] == "2.0"
        assert d["enriched"] is True
        assert "shape" in d and "cycles" in d and "metrics" in d
        assert "node_annotations" in d and "risks" in d
        assert "nodes" in d and "edges" in d and "entry_point" in d and "terminal_nodes" in d
        assert "token" not in " ".join(k.lower() for k in d.keys())
        assert "cost" not in " ".join(k.lower() for k in d.keys())


def test_run_pipeline_on_directory_enriched_is_deterministic():
    """
    Same repo + enriched mode should produce identical JSON output across runs.
    """
    workflow_source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"loop": "a", "done": END})
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "workflow.py").write_text(workflow_source)

        r1, w1 = run_pipeline_on_directory(root, enriched=True)
        r2, w2 = run_pipeline_on_directory(root, enriched=True)
        assert w1 == w2
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_run_pipeline_on_directory_enriched_unreadable_file_does_not_crash():
    """
    Enriched mode also handles unreadable files safely and still returns warnings.
    """
    workflow_source = """
from langgraph.graph import StateGraph, START
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "workflow.py").write_text(workflow_source)
        (root / "src" / "bad.py").write_bytes(b"\xff\xfe invalid utf-8")

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) >= 1
        assert results[0]["schema_version"] == "2.0"
        assert results[0]["enriched"] is True
        assert any("could not read" in w for w in warnings)


def test_run_pipeline_on_directory_enriched_syntax_error_only_does_not_crash():
    """
    Enriched mode on a directory with only syntax-error files returns empty results.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "broken.py").write_text("from langgraph.graph import StateGraph\nx = (  # unclosed")

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert results == []
        assert isinstance(warnings, list)


def test_run_pipeline_on_directory_enriched_multiple_graphs_same_file():
    """
    Enriched mode with two graphs in one file produces two enriched dicts.
    """
    source = """
from langgraph.graph import StateGraph, START, END
a = StateGraph(dict)
a.add_node("x", fx)
a.add_edge(START, "x")
a.add_edge("x", END)

b = StateGraph(dict)
b.add_node("p", fp)
b.add_node("q", fq)
b.add_edge(START, "p")
b.add_edge("p", "q")
b.add_edge("q", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "multi.py").write_text(source)

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) == 2
        for d in results:
            assert d["schema_version"] == "2.0"
            assert d["enriched"] is True
            assert "shape" in d and "cycles" in d and "metrics" in d
        # One has 1 node, the other has 2
        node_counts = sorted(d["metrics"]["node_count"] for d in results)
        assert node_counts == [1, 2]


def test_run_pipeline_on_directory_enriched_annotations_with_source():
    """
    Enriched mode: node annotations use source code — local functions get origin=local_function.
    """
    source = """
from langgraph.graph import StateGraph, START, END

def agent(state):
    state['result'] = 'done'
    return state

def checker(state):
    return state['result']

graph = StateGraph(dict)
graph.add_node("agent", agent)
graph.add_node("checker", checker)
graph.add_edge(START, "agent")
graph.add_edge("agent", "checker")
graph.add_edge("checker", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "wf.py").write_text(source)

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) == 1
        d = results[0]
        ann_map = {a["node_name"]: a for a in d["node_annotations"]}
        assert ann_map["agent"]["origin"] == "local_function"
        assert ann_map["checker"]["origin"] == "local_function"
        assert ann_map["agent"]["state_interaction"] == "mutates_state"
        assert ann_map["checker"]["state_interaction"] == "read_only"


def test_run_pipeline_on_directory_enriched_cyclic_graph():
    """
    Enriched mode on cyclic workflow: shape is cyclic, at least one cycle detected.
    """
    source = """
from langgraph.graph import StateGraph, START, END

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"loop": "a", "done": END})
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "cyclic.py").write_text(source)

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) == 1
        d = results[0]
        assert d["shape"] == "cyclic"
        assert len(d["cycles"]) >= 1
        assert d["cycles"][0]["reaches_terminal"] is True
        assert d["metrics"]["max_depth_before_cycle"] is not None


def test_run_pipeline_on_directory_enriched_branching_graph():
    """
    Enriched mode on branching workflow: shape is branching, no cycles.
    """
    source = """
from langgraph.graph import StateGraph, START, END

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_node("c", fc)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("a", "c")
graph.add_edge("b", END)
graph.add_edge("c", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "branch.py").write_text(source)

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) == 1
        d = results[0]
        assert d["shape"] == "branching"
        assert d["cycles"] == []
        assert d["metrics"]["node_count"] == 3
        assert d["metrics"]["max_depth_before_cycle"] is None


def test_run_pipeline_on_directory_enriched_no_dead_ends_linear():
    """
    Enriched mode on properly terminated linear graph: no dead-end risks.
    """
    source = """
from langgraph.graph import StateGraph, START, END

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "linear.py").write_text(source)

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) == 1
        d = results[0]
        assert d["shape"] == "linear"
        r = d["risks"]
        assert r["unreachable_node_ids"] == []
        assert r["dead_end_ids"] == []
        assert r["non_terminating_cycle_ids"] == []
        assert r["multiple_entry_points"] is False


# ── Pipeline edge cases ──────────────────────────────────────────────────


def test_run_pipeline_on_directory_empty_dir():
    """Empty directory returns no results and no warnings."""
    with tempfile.TemporaryDirectory() as tmp:
        results, warnings = run_pipeline_on_directory(Path(tmp))
        assert results == []
        assert warnings == []


def test_run_pipeline_on_directory_only_non_langgraph():
    """Directory with only non-LangGraph .py files returns no results."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "util.py").write_text("def add(a, b): return a + b\n")
        (root / "main.py").write_text("import util\nprint(util.add(1, 2))\n")
        results, warnings = run_pipeline_on_directory(root)
        assert results == []
        assert warnings == []


def test_run_pipeline_on_directory_enriched_empty_dir():
    """Enriched mode on empty directory returns no results."""
    with tempfile.TemporaryDirectory() as tmp:
        results, warnings = run_pipeline_on_directory(Path(tmp), enriched=True)
        assert results == []
        assert warnings == []


def test_run_pipeline_on_directory_mixed_valid_invalid_enriched():
    """
    Enriched mode: directory with one valid workflow, one syntax-error, one unreadable.
    Only the valid file's graph is returned; warnings collected for bad files.
    """
    valid_source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "valid.py").write_text(valid_source)
        (root / "src" / "broken.py").write_text(
            "from langgraph.graph import StateGraph\nx = (  # unclosed"
        )
        (root / "src" / "bad.py").write_bytes(b"\xff\xfe invalid utf-8")

        results, warnings = run_pipeline_on_directory(root, enriched=True)
        assert len(results) >= 1
        assert results[0]["schema_version"] == "2.0"
        assert results[0]["enriched"] is True
        assert any("could not read" in w for w in warnings)


def test_run_pipeline_default_vs_enriched_same_graph():
    """
    Same directory: default and enriched mode produce same base graph fields,
    but enriched adds Phase-2 keys.
    """
    source = """
from langgraph.graph import StateGraph, START, END
graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_edge("b", END)
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "wf.py").write_text(source)

        r_base, _ = run_pipeline_on_directory(root, enriched=False)
        r_enriched, _ = run_pipeline_on_directory(root, enriched=True)

        assert len(r_base) == len(r_enriched) == 1
        base = r_base[0]
        enr = r_enriched[0]

        # Same base graph fields
        assert base["graph_id"] == enr["graph_id"]
        assert base["entry_point"] == enr["entry_point"]
        assert base["terminal_nodes"] == enr["terminal_nodes"]
        assert base["nodes"] == enr["nodes"]
        assert base["edges"] == enr["edges"]

        # Enriched has extra keys
        assert base["schema_version"] == "1.0"
        assert enr["schema_version"] == "2.0"
        assert "enriched" not in base
        assert enr["enriched"] is True
        assert "shape" in enr and "cycles" in enr


# ── JSON schema validation for enriched output ───────────────────────────


def test_enriched_output_schema_validation():
    """
    Rigorous schema validation: every enriched dict has all required keys
    with correct types, recursively.
    """
    source = """
from langgraph.graph import StateGraph, START, END

def agent(state):
    state['result'] = 'ok'
    return state

def checker(state):
    return state['result']

graph = StateGraph(dict)
graph.add_node("agent", agent)
graph.add_node("checker", checker)
graph.add_edge(START, "agent")
graph.add_edge("agent", "checker")
graph.add_conditional_edges("checker", lambda s: "done" if s else "retry", {"done": END, "retry": "agent"})
"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "wf.py").write_text(source)

        results, _ = run_pipeline_on_directory(root, enriched=True)
        assert len(results) >= 1

        for d in results:
            # Top-level keys
            assert isinstance(d["schema_version"], str) and d["schema_version"] == "2.0"
            assert d["enriched"] is True
            assert isinstance(d["graph_id"], str)
            assert isinstance(d["entry_point"], (str, type(None)))
            assert isinstance(d["terminal_nodes"], list)
            assert isinstance(d["nodes"], list)
            assert isinstance(d["edges"], list)
            assert isinstance(d["conditional_edges"], list)
            assert isinstance(d["shape"], str)
            assert d["shape"] in ("linear", "branching", "cyclic", "disconnected", "invalid")
            assert isinstance(d["cycles"], list)
            assert isinstance(d["metrics"], dict)
            assert isinstance(d["node_annotations"], list)
            assert isinstance(d["risks"], dict)

            # Nodes schema
            for n in d["nodes"]:
                assert isinstance(n["name"], str)
                assert isinstance(n["callable_ref"], str)
                assert isinstance(n["line"], int)

            # Edges schema
            for e in d["edges"]:
                assert isinstance(e["source"], str)
                assert isinstance(e["target"], str)
                assert isinstance(e["line"], int)

            # Conditional edges schema
            for e in d["conditional_edges"]:
                assert isinstance(e["source"], str)
                assert isinstance(e["target"], str)
                assert isinstance(e["condition_label"], str)
                assert isinstance(e["line"], int)

            # Cycles schema
            for c in d["cycles"]:
                assert c["cycle_type"] in (
                    "self_loop", "multi_node", "conditional", "non_terminating"
                )
                assert isinstance(c["nodes"], list)
                assert all(isinstance(n, str) for n in c["nodes"])
                assert isinstance(c["reaches_terminal"], bool)

            # Metrics schema
            m = d["metrics"]
            assert isinstance(m["node_count"], int)
            assert isinstance(m["edge_count"], int)
            assert isinstance(m["entry_node"], (str, type(None)))
            assert isinstance(m["terminal_nodes"], list)
            assert isinstance(m["unreachable_nodes"], list)
            assert isinstance(m["longest_acyclic_path"], int)
            assert isinstance(m["avg_branching_factor"], (int, float))
            assert isinstance(m["max_depth_before_cycle"], (int, type(None)))

            # Node annotations schema
            for a in d["node_annotations"]:
                assert isinstance(a["node_name"], str)
                assert a["origin"] in (
                    "local_function", "imported_function", "class_method",
                    "lambda", "unknown"
                )
                assert a["state_interaction"] in (
                    "pure", "read_only", "mutates_state", "unknown"
                )
                assert a["role"] in (
                    "llm_like", "tool_like", "control_node", "unknown"
                )

            # Risks schema
            r = d["risks"]
            assert isinstance(r["unreachable_node_ids"], list)
            assert isinstance(r["dead_end_ids"], list)
            assert isinstance(r["non_terminating_cycle_ids"], list)
            assert isinstance(r["multiple_entry_points"], bool)

            # Annotation count == node count
            assert len(d["node_annotations"]) == len(d["nodes"])

            # No token/cost keys anywhere
            all_keys = " ".join(str(k).lower() for k in d.keys())
            assert "token" not in all_keys
            assert "cost" not in all_keys
