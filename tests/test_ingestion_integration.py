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
