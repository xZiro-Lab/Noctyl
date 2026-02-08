"""
High-level integration tests: detector, tracker, node extractor, and edge extractor in sync.

One source file is run through the full ingestion pipeline; we assert all components
agree on graph presence, graph_id, and that nodes/edges belong to the same graph.
"""

from noctyl.ingestion import (
    extract_add_conditional_edges,
    extract_add_edge_calls,
    extract_add_node_calls,
    extract_entry_points,
    file_contains_langgraph,
    has_langgraph_import,
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
