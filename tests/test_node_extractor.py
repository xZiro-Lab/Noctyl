"""Tests for add_node extraction."""

from noctyl_scout.graph.nodes import ExtractedNode
from noctyl_scout.ingestion.node_extractor import extract_add_node_calls
from noctyl_scout.ingestion.stategraph_tracker import track_stategraph_instances


def _extract(source: str, file_path: str = "file.py"):
    tracked = track_stategraph_instances(source, file_path)
    return extract_add_node_calls(source, file_path, tracked)


def test_one_graph_one_add_node():
    """One graph, one add_node('name', func) -> node with correct name and callable_ref."""
    source = """
from langgraph.graph import StateGraph

def my_node(state):
    return state

g = StateGraph(dict)
g.add_node("agent", my_node)
"""
    result = _extract(source)
    assert len(result) == 1
    graph_id = list(result.keys())[0]
    nodes = result[graph_id]
    assert len(nodes) == 1
    assert nodes[0].name == "agent"
    assert nodes[0].callable_ref == "my_node"
    assert nodes[0].line > 0


def test_one_graph_multiple_add_node():
    """One graph, multiple add_node calls -> all extracted in order."""
    source = """
from langgraph.graph import StateGraph

def a(s): return s
def b(s): return s

g = StateGraph(dict)
g.add_node("first", a)
g.add_node("second", b)
"""
    result = _extract(source)
    assert len(result) == 1
    nodes = result[list(result.keys())[0]]
    assert len(nodes) == 2
    assert nodes[0].name == "first" and nodes[0].callable_ref == "a"
    assert nodes[1].name == "second" and nodes[1].callable_ref == "b"


def test_aliased_receiver():
    """g = StateGraph(...); h = g; h.add_node('n', f) -> node attributed to same graph_id."""
    source = """
from langgraph.graph import StateGraph

def f(s): return s
g = StateGraph(dict)
h = g
h.add_node("n", f)
"""
    result = _extract(source)
    assert len(result) == 1
    nodes = result[list(result.keys())[0]]
    assert len(nodes) == 1
    assert nodes[0].name == "n" and nodes[0].callable_ref == "f"


def test_callable_ref_name():
    """callable_ref from Name -> node.id."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("x", my_func)
"""
    result = _extract(source)
    nodes = result[list(result.keys())[0]]
    assert nodes[0].callable_ref == "my_func"


def test_callable_ref_attribute():
    """callable_ref from Attribute -> unparse or 'a.b'."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("x", mod.handler)
"""
    result = _extract(source)
    nodes = result[list(result.keys())[0]]
    assert nodes[0].callable_ref == "mod.handler"


def test_callable_ref_lambda():
    """callable_ref from Lambda -> 'lambda'."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("x", lambda s: s)
"""
    result = _extract(source)
    nodes = result[list(result.keys())[0]]
    assert nodes[0].callable_ref == "lambda"


def test_no_add_node():
    """No add_node calls -> empty lists per graph."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
"""
    result = _extract(source)
    assert len(result) == 1
    assert result[list(result.keys())[0]] == []


def test_non_stategraph_add_node_ignored():
    """add_node on non-StateGraph receiver -> not attributed to our graph."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
other = SomeOtherClass()
other.add_node("n", f)
"""
    result = _extract(source)
    nodes = result[list(result.keys())[0]]
    assert len(nodes) == 0


def test_syntax_error_returns_empty():
    """Syntax error -> empty dict."""
    result = _extract("from langgraph.graph import StateGraph\n g = StateGraph( ", "x.py")
    assert result == {}


def test_empty_source_no_tracked_graphs():
    """Empty source -> no tracked graphs, so extract returns {} or empty per-graph."""
    result = _extract("")
    assert result == {}


def test_multiple_graphs_nodes_assigned_correctly():
    """Multiple graphs in one file -> nodes assigned to correct graph_id by receiver."""
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
a.add_node("from_a", fn1)
b.add_node("from_b", fn2)
"""
    result = _extract(source)
    assert len(result) == 2
    graph_ids = sorted(result.keys())
    by_id = {gid: [n.name for n in result[gid]] for gid in graph_ids}
    assert "from_a" in by_id[graph_ids[0]] or "from_a" in by_id[graph_ids[1]]
    assert "from_b" in by_id[graph_ids[0]] or "from_b" in by_id[graph_ids[1]]
    for gid, nodes in result.items():
        assert len(nodes) == 1
        if nodes[0].name == "from_a":
            assert nodes[0].callable_ref == "fn1"
        else:
            assert nodes[0].name == "from_b" and nodes[0].callable_ref == "fn2"


def test_node_name_from_constant():
    """Node name from string Constant."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("constant_name", f)
"""
    result = _extract(source)
    nodes = result[list(result.keys())[0]]
    assert nodes[0].name == "constant_name"


def test_extracted_node_is_dataclass():
    """ExtractedNode has name, callable_ref, line."""
    node = ExtractedNode(name="n", callable_ref="f", line=10)
    assert node.name == "n"
    assert node.callable_ref == "f"
    assert node.line == 10
