"""Tests for file-level LangGraph detection."""

from noctyl.ingestion.langgraph_detector import (
    file_contains_langgraph,
    has_langgraph_import,
)
from noctyl.ingestion.stategraph_tracker import track_stategraph_instances


def test_file_contains_langgraph_no_import():
    """File with no LangGraph import → False."""
    source = """
def foo():
    return 42
"""
    assert file_contains_langgraph(source) is False
    assert file_contains_langgraph(source, "app.py") is False


def test_file_contains_langgraph_import_only_no_instantiation():
    """File with LangGraph import but no StateGraph() call → False."""
    source = """
from langgraph.graph import StateGraph
# never instantiated
"""
    assert file_contains_langgraph(source) is False


def test_file_contains_langgraph_with_instantiation():
    """File with one StateGraph() → True."""
    source = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
"""
    assert file_contains_langgraph(source) is True
    assert file_contains_langgraph(source, "workflow.py") is True


def test_file_contains_langgraph_multiple_instances():
    """File with multiple StateGraph() → True."""
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
"""
    assert file_contains_langgraph(source) is True


def test_file_contains_langgraph_aliased_import():
    """File with aliased import and StateGraph() → True."""
    source = """
from langgraph.graph import StateGraph as SG
g = SG(dict)
"""
    assert file_contains_langgraph(source) is True


def test_file_contains_langgraph_qualified_name():
    """File with import langgraph.graph and qualified StateGraph() → True."""
    source = """
import langgraph.graph
g = langgraph.graph.StateGraph(dict)
"""
    assert file_contains_langgraph(source) is True


def test_file_contains_langgraph_syntax_error():
    """File with syntax error → False (safe ignore)."""
    source = "from langgraph.graph import StateGraph\n g = StateGraph( "
    assert file_contains_langgraph(source) is False


def test_file_contains_langgraph_empty():
    """Empty or whitespace-only source → False."""
    assert file_contains_langgraph("") is False
    assert file_contains_langgraph("   \n\n") is False


def test_has_langgraph_import_no_import():
    """No langgraph.graph import → False."""
    source = "from other.module import StateGraph\ng = StateGraph(dict)"
    assert has_langgraph_import(source) is False


def test_has_langgraph_import_from_import():
    """from langgraph.graph import ... → True."""
    source = "from langgraph.graph import StateGraph"
    assert has_langgraph_import(source) is True


def test_has_langgraph_import_module_import():
    """import langgraph.graph → True."""
    source = "import langgraph.graph"
    assert has_langgraph_import(source) is True


def test_has_langgraph_import_present_without_instantiation():
    """Import present but no StateGraph() call → has_langgraph_import True."""
    source = """
from langgraph.graph import StateGraph
# no call
"""
    assert has_langgraph_import(source) is True
    assert file_contains_langgraph(source) is False


def test_has_langgraph_import_syntax_error():
    """Syntax error → False."""
    assert has_langgraph_import("from langgraph.graph import (") is False


# --- Integration: detector and tracker stay in sync ---


def test_detector_and_tracker_integration():
    """
    file_contains_langgraph and track_stategraph_instances agree:
    detector is True iff tracker returns at least one instance;
    instance count and file_path/graph_id are consistent.
    """
    file_path = "src/workflow.py"

    # No LangGraph: both say "none"
    no_lang = "def main(): pass"
    assert file_contains_langgraph(no_lang, file_path) is False
    assert track_stategraph_instances(no_lang, file_path) == []

    # One StateGraph: detector True, tracker returns one instance with same path
    one_graph = """
from langgraph.graph import StateGraph
g = StateGraph(dict)
"""
    assert file_contains_langgraph(one_graph, file_path) is True
    instances_one = track_stategraph_instances(one_graph, file_path)
    assert len(instances_one) == 1
    assert instances_one[0].file_path in ("src/workflow.py", "src\\workflow.py")
    assert ":0" in instances_one[0].graph_id
    assert instances_one[0].variable_name == "g"

    # Two StateGraphs: detector True, tracker returns two with distinct graph_ids
    two_graphs = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
"""
    assert file_contains_langgraph(two_graphs, file_path) is True
    instances_two = track_stategraph_instances(two_graphs, file_path)
    assert len(instances_two) == 2
    ids = {i.graph_id for i in instances_two}
    assert ids == {instances_two[0].file_path + ":0", instances_two[0].file_path + ":1"}

    # Fast path: no import => detector False and tracker empty
    assert has_langgraph_import(no_lang) is False
    assert file_contains_langgraph(no_lang) is False
    assert track_stategraph_instances(no_lang, "") == []
