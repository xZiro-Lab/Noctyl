"""Tests for add_edge extraction."""

from noctyl.graph.edges import ExtractedEdge
from noctyl.ingestion.edge_extractor import extract_add_edge_calls
from noctyl.ingestion.stategraph_tracker import track_stategraph_instances


def _extract_edges(source: str, file_path: str = "file.py"):
    tracked = track_stategraph_instances(source, file_path)
    return extract_add_edge_calls(source, file_path, tracked)


def test_one_graph_one_add_edge():
    """One graph, one add_edge(source, target) -> edge with correct source/target."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("a", fn_a)
g.add_node("b", fn_b)
g.add_edge("a", "b")
"""
    result = _extract_edges(source)
    assert len(result) == 1
    edges = result[list(result.keys())[0]]
    assert len(edges) == 1
    assert edges[0].source == "a"
    assert edges[0].target == "b"
    assert edges[0].line > 0


def test_one_graph_multiple_add_edge():
    """One graph, multiple add_edge calls -> all extracted in order."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("a", fa)
g.add_node("b", fb)
g.add_node("c", fc)
g.add_edge("a", "b")
g.add_edge("b", "c")
"""
    result = _extract_edges(source)
    assert len(result) == 1
    edges = result[list(result.keys())[0]]
    assert len(edges) == 2
    assert edges[0].source == "a" and edges[0].target == "b"
    assert edges[1].source == "b" and edges[1].target == "c"


def test_literal_source_target():
    """Literal string source and target."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_edge("from_here", "to_there")
"""
    result = _extract_edges(source)
    edges = result[list(result.keys())[0]]
    assert edges[0].source == "from_here"
    assert edges[0].target == "to_there"


def test_start_end_names():
    """START and END as source/target (Name nodes) -> stringified as START/END."""
    source = """
from langgraph.graph import StateGraph, START, END
g = StateGraph(dict)
g.add_node("n", f)
g.add_edge(START, "n")
g.add_edge("n", END)
"""
    result = _extract_edges(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 2
    assert edges[0].source == "START" and edges[0].target == "n"
    assert edges[1].source == "n" and edges[1].target == "END"


def test_aliased_receiver():
    """g = StateGraph(...); h = g; h.add_edge(...) -> edge attributed to same graph_id."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
h = g
h.add_edge("a", "b")
"""
    result = _extract_edges(source)
    assert len(result) == 1
    edges = result[list(result.keys())[0]]
    assert len(edges) == 1
    assert edges[0].source == "a" and edges[0].target == "b"


def test_no_add_edge():
    """No add_edge calls -> empty list for graph."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("a", f)
"""
    result = _extract_edges(source)
    assert len(result) == 1
    assert result[list(result.keys())[0]] == []


def test_non_stategraph_add_edge_ignored():
    """add_edge on non-StateGraph receiver -> not attributed to our graph."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
other = SomeClass()
other.add_edge("a", "b")
"""
    result = _extract_edges(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 0


def test_syntax_error_returns_empty():
    """Syntax error -> empty dict."""
    result = _extract_edges("from langgraph.graph import StateGraph\n g = StateGraph( ", "x.py")
    assert result == {}


def test_multiple_graphs_edges_assigned_correctly():
    """Multiple graphs -> edges assigned to correct graph_id."""
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
a.add_edge("x", "y")
b.add_edge("p", "q")
"""
    result = _extract_edges(source)
    assert len(result) == 2
    all_edges = []
    for gid, edges in result.items():
        all_edges.extend(edges)
    sources = {e.source for e in all_edges}
    targets = {e.target for e in all_edges}
    assert sources == {"x", "p"}
    assert targets == {"y", "q"}


def test_missing_nodes_edge_still_extracted():
    """add_edge('a', 'b') with no add_node for a/b -> edge still extracted (missing nodes)."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_edge("a", "b")
"""
    result = _extract_edges(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 1
    assert edges[0].source == "a"
    assert edges[0].target == "b"


def test_non_literal_source_target_stringified():
    """Non-literal source/target (e.g. variable) -> stringified."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
src = "node_a"
tgt = "node_b"
g.add_edge(src, tgt)
"""
    result = _extract_edges(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 1
    assert edges[0].source == "src"
    assert edges[0].target == "tgt"


def test_extracted_edge_dataclass():
    """ExtractedEdge has source, target, line."""
    edge = ExtractedEdge(source="a", target="b", line=10)
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.line == 10


def test_empty_source_no_tracked_graphs():
    """Empty source -> no tracked graphs -> empty dict."""
    result = _extract_edges("")
    assert result == {}
