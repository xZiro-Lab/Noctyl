"""Unit tests for StateGraph instance tracking (plan §8.1 T1–T12)."""

from noctyl.ingestion.stategraph_tracker import (
    TrackedStateGraph,
    track_stategraph_instances,
)


def _track(source: str, file_path: str = "file.py") -> list[TrackedStateGraph]:
    return track_stategraph_instances(source, file_path)


# T1: Single instance — one StateGraph, direct import, assigned to variable
def test_single_instance_direct_import():
    source = """
from langgraph.graph import StateGraph

class MyState:
    pass

graph = StateGraph(MyState)
"""
    instances = _track(source)
    assert len(instances) == 1
    assert instances[0].variable_name == "graph"
    assert instances[0].graph_id.endswith(":0")
    assert "file.py" in instances[0].graph_id or "file" in instances[0].file_path
    assert instances[0].binding_kind == "assignment"


# T2: Multiple per file — two StateGraphs in same file, both assigned
def test_multiple_per_file():
    source = """
from langgraph.graph import StateGraph

a = StateGraph(dict)
b = StateGraph(dict)
"""
    instances = _track(source)
    assert len(instances) == 2
    ids = [i.graph_id for i in instances]
    assert ids[0].endswith(":0") and ids[1].endswith(":1")
    assert ids[0] != ids[1]
    assert instances[0].variable_name == "a"
    assert instances[1].variable_name == "b"


# T3: Aliased import
def test_aliased_import():
    source = """
from langgraph.graph import StateGraph as SG
g = SG(dict)
"""
    instances = _track(source)
    assert len(instances) == 1
    assert instances[0].variable_name == "g"


# T4: Qualified name — import langgraph.graph then langgraph.graph.StateGraph(...)
def test_qualified_name():
    source = """
import langgraph.graph
g = langgraph.graph.StateGraph(dict)
"""
    instances = _track(source)
    assert len(instances) == 1
    assert instances[0].variable_name == "g"


# T5: Inline usage — no assignment; StateGraph(...).compile()
def test_inline_usage():
    source = """
from langgraph.graph import StateGraph

app = StateGraph(dict).compile()
"""
    # One StateGraph(...) call, bound to "app" via assignment (app = StateGraph(...).compile()).
    # The RHS is a Call (compile), not the StateGraph call directly. So the StateGraph call
    # is not the direct value of an Assign; it's inside an Attribute (StateGraph(...).compile()).
    # So variable_name should be None, binding_kind "inline".
    instances = _track(source)
    assert len(instances) == 1
    assert instances[0].variable_name is None
    assert instances[0].binding_kind == "inline"


# T6: No StateGraph — file with no LangGraph usage
def test_no_stategraph():
    source = """
def foo():
    pass
"""
    instances = _track(source)
    assert len(instances) == 0


def test_no_stategraph_unrelated_imports():
    source = """
import os
from other.module import something
"""
    instances = _track(source)
    assert len(instances) == 0


# T7: Unrelated Call — StateGraph as argument or string, not as constructor call
def test_unrelated_call_stategraph_as_argument():
    source = """
from langgraph.graph import StateGraph
something_else(StateGraph)
"""
    instances = _track(source)
    assert len(instances) == 0


def test_string_stategraph_not_counted():
    source = '''
x = "StateGraph"
'''
    instances = _track(source)
    assert len(instances) == 0


# T8: Same line (two calls) — a, b = StateGraph(...), StateGraph(...)
def test_same_line_two_calls():
    source = """
from langgraph.graph import StateGraph
a, b = StateGraph(dict), StateGraph(dict)
"""
    instances = _track(source)
    assert len(instances) == 2
    assert instances[0].graph_id.endswith(":0")
    assert instances[1].graph_id.endswith(":1")
    assert instances[0].variable_name == "a"
    assert instances[1].variable_name == "b"


# T9: Tuple unpacking — g, x = StateGraph(...), 1
def test_tuple_unpacking():
    source = """
from langgraph.graph import StateGraph
g, x = StateGraph(dict), 1
"""
    instances = _track(source)
    assert len(instances) == 1
    assert instances[0].variable_name == "g"


# T10: Determinism — two runs same file produce same graph_id list
def test_determinism():
    source = """
from langgraph.graph import StateGraph
a = StateGraph(dict)
b = StateGraph(dict)
"""
    run1 = _track(source, "path/to/file.py")
    run2 = _track(source, "path/to/file.py")
    assert [t.graph_id for t in run1] == [t.graph_id for t in run2]
    assert [t.variable_name for t in run1] == [t.variable_name for t in run2]


# T11: File path normalization — consistent path in graph_id
def test_file_path_normalization():
    source = "from langgraph.graph import StateGraph\ng = StateGraph(dict)"
    r1 = track_stategraph_instances(source, "src/a.py")
    r2 = track_stategraph_instances(source, "src\\a.py")
    # Normalized paths should be consistent (no mixed slashes)
    assert r1[0].graph_id == r2[0].graph_id
    assert r1[0].file_path == r2[0].file_path


# T12: Non-LangGraph import — from other.module import StateGraph does not count
def test_non_langgraph_import_not_counted():
    source = """
from other.module import StateGraph
g = StateGraph(dict)
"""
    instances = _track(source)
    # We only recognize langgraph.graph.StateGraph, so this is 0
    assert len(instances) == 0


def test_mixed_import_only_langgraph_counted():
    source = """
from other.module import StateGraph as OtherSG
from langgraph.graph import StateGraph
g = StateGraph(dict)
"""
    instances = _track(source)
    assert len(instances) == 1
    assert instances[0].variable_name == "g"


# Extra: syntax error returns empty list
def test_syntax_error_returns_empty():
    instances = _track("from langgraph.graph import StateGraph\n g = StateGraph( ", "x.py")
    assert len(instances) == 0
