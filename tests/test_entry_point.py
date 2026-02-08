"""Tests for entry point detection (set_entry_point and fallback from add_edge(START, ...))."""

from noctyl.ingestion.edge_extractor import extract_entry_points
from noctyl.ingestion.stategraph_tracker import track_stategraph_instances


def _extract_entry(source: str, file_path: str = "file.py"):
    tracked = track_stategraph_instances(source, file_path)
    return extract_entry_points(source, file_path, tracked)


def test_set_entry_point_present():
    """set_entry_point('name') -> entry extracted."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("agent", f)
g.set_entry_point("agent")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert len(entry_by_graph) == 1
    assert entry_by_graph[list(entry_by_graph.keys())[0]] == "agent"
    assert warnings == []


def test_aliased_receiver():
    """set_entry_point on aliased graph receiver -> attributed to same graph."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
h = g
h.set_entry_point("start_node")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert len(entry_by_graph) == 1
    assert entry_by_graph[list(entry_by_graph.keys())[0]] == "start_node"
    assert warnings == []


def test_multiple_graphs():
    """Multiple graphs -> each gets its own entry or None."""
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
a.set_entry_point("n1")
b.set_entry_point("n2")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert len(entry_by_graph) == 2
    entries = list(entry_by_graph.values())
    assert set(entries) == {"n1", "n2"}
    assert warnings == []


def test_fallback_from_single_start_edge():
    """No set_entry_point but one add_edge(START, 'x') -> inferred entry."""
    source = """
from langgraph.graph import StateGraph, START
g = StateGraph(dict)
g.add_node("first", f)
g.add_edge(START, "first")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert len(entry_by_graph) == 1
    assert entry_by_graph[list(entry_by_graph.keys())[0]] == "first"
    assert warnings == []


def test_no_set_entry_point_no_start_edge_warning():
    """No set_entry_point and no START edge -> None + warning."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("a", f)
g.add_edge("a", "a")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert len(entry_by_graph) == 1
    assert entry_by_graph[list(entry_by_graph.keys())[0]] is None
    assert len(warnings) == 1
    assert "no entry point detected" in warnings[0]


def test_multiple_start_edges_ambiguous_warning():
    """Multiple add_edge(START, ...) -> None + ambiguous warning."""
    source = """
from langgraph.graph import StateGraph, START
g = StateGraph(dict)
g.add_node("a", fa)
g.add_node("b", fb)
g.add_edge(START, "a")
g.add_edge(START, "b")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert len(entry_by_graph) == 1
    assert entry_by_graph[list(entry_by_graph.keys())[0]] is None
    assert len(warnings) == 1
    assert "ambiguous" in warnings[0]


def test_syntax_error_returns_empty():
    """Syntax error -> ({}, [])."""
    entry_by_graph, warnings = _extract_entry(
        "from langgraph.graph import StateGraph\n g = StateGraph( ", "x.py"
    )
    assert entry_by_graph == {}
    assert warnings == []


def test_explicit_wins_over_fallback():
    """set_entry_point and add_edge(START, x) both present -> explicit wins."""
    source = """
from langgraph.graph import StateGraph, START
g = StateGraph(dict)
g.add_node("a", fa)
g.add_node("b", fb)
g.add_edge(START, "a")
g.set_entry_point("b")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert entry_by_graph[list(entry_by_graph.keys())[0]] == "b"
    assert warnings == []


def test_last_set_entry_point_wins():
    """Multiple set_entry_point for same graph -> last wins."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.set_entry_point("first")
g.set_entry_point("second")
"""
    entry_by_graph, warnings = _extract_entry(source)
    assert entry_by_graph[list(entry_by_graph.keys())[0]] == "second"
    assert warnings == []


def test_empty_source_no_tracked_graphs():
    """Empty source -> no tracked graphs -> ({}, [])."""
    entry_by_graph, warnings = _extract_entry("")
    assert entry_by_graph == {}
    assert warnings == []
