"""Tests for add_conditional_edges extraction."""

from noctyl_scout.graph.edges import ExtractedConditionalEdge
from noctyl_scout.ingestion.edge_extractor import extract_add_conditional_edges
from noctyl_scout.ingestion.stategraph_tracker import track_stategraph_instances


def _extract_conditional(source: str, file_path: str = "file.py"):
    tracked = track_stategraph_instances(source, file_path)
    return extract_add_conditional_edges(source, file_path, tracked)


def test_one_conditional_edge_literal_path_map():
    """Literal path_map with one entry -> one conditional edge."""
    source = """
from langgraph.graph import StateGraph, END
g = StateGraph(dict)
g.add_node("a", fa)
g.add_conditional_edges("a", router, {"yes": "b", "no": END})
"""
    result = _extract_conditional(source)
    assert len(result) == 1
    edges = result[list(result.keys())[0]]
    assert len(edges) == 2
    by_label = {e.condition_label: e for e in edges}
    assert by_label["yes"].source == "a" and by_label["yes"].target == "b"
    assert by_label["no"].source == "a" and by_label["no"].target == "END"


def test_condition_labels_preserved():
    """Condition labels from path_map keys are preserved."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_conditional_edges("x", decide, {"continue": "next", "stop": "end_node"})
"""
    result = _extract_conditional(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 2
    labels = {e.condition_label for e in edges}
    assert labels == {"continue", "stop"}
    targets = {e.target for e in edges}
    assert targets == {"next", "end_node"}


def test_end_as_target():
    """END as path_map value -> target is 'END' (terminal)."""
    source = """
from langgraph.graph import StateGraph, END
g = StateGraph(dict)
g.add_conditional_edges("n", r, {"done": END})
"""
    result = _extract_conditional(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 1
    assert edges[0].target == "END"
    assert edges[0].condition_label == "done"


def test_aliased_receiver():
    """Conditional edges on aliased graph receiver -> attributed to same graph_id."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
h = g
h.add_conditional_edges("src", path, {"a": "x"})
"""
    result = _extract_conditional(source)
    assert len(result) == 1
    edges = result[list(result.keys())[0]]
    assert len(edges) == 1
    assert edges[0].source == "src" and edges[0].condition_label == "a" and edges[0].target == "x"


def test_no_path_map_literal_skipped():
    """add_conditional_edges with only source and path (no path_map dict) -> no edges."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_conditional_edges("a", router)
"""
    result = _extract_conditional(source)
    assert len(result) == 1
    assert result[list(result.keys())[0]] == []


def test_path_map_variable_skipped():
    """path_map as variable (non-dict literal) -> skipped, no edges for that call."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
path_map = {"k": "v"}
g.add_conditional_edges("a", router, path_map)
"""
    result = _extract_conditional(source)
    assert len(result) == 1
    assert result[list(result.keys())[0]] == []


def test_multiple_graphs_conditional_assigned_correctly():
    """Multiple graphs -> conditional edges assigned to correct graph_id."""
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
a.add_conditional_edges("n1", r1, {"x": "y"})
b.add_conditional_edges("n2", r2, {"p": "q"})
"""
    result = _extract_conditional(source)
    assert len(result) == 2
    all_edges = []
    for edges in result.values():
        all_edges.extend(edges)
    assert len(all_edges) == 2
    labels = {e.condition_label for e in all_edges}
    assert labels == {"x", "p"}


def test_syntax_error_returns_empty():
    """Syntax error -> empty dict."""
    result = _extract_conditional("from langgraph.graph import StateGraph\n g = StateGraph( ", "x.py")
    assert result == {}


def test_empty_source_no_tracked_graphs():
    """Empty source -> no tracked graphs -> empty dict."""
    result = _extract_conditional("")
    assert result == {}


def test_path_map_keyword_arg():
    """path_map passed as keyword (dict literal) -> extracted."""
    source = """
from langgraph.graph import StateGraph, END
g = StateGraph(dict)
g.add_conditional_edges("a", router, path_map={"ok": "b", "finish": END})
"""
    result = _extract_conditional(source)
    edges = result[list(result.keys())[0]]
    assert len(edges) == 2
    by_label = {e.condition_label: e for e in edges}
    assert by_label["ok"].target == "b"
    assert by_label["finish"].target == "END"


def test_extracted_conditional_edge_dataclass():
    """ExtractedConditionalEdge has source, condition_label, target, line."""
    edge = ExtractedConditionalEdge(source="a", condition_label="l", target="b", line=10)
    assert edge.source == "a"
    assert edge.condition_label == "l"
    assert edge.target == "b"
    assert edge.line == 10
