"""
Golden tests: canonical LangGraph fixtures and validated graph extraction.

Runs run_pipeline_on_directory on tests/fixtures/golden/ and asserts expected
structure for each case (linear, conditional loop, END termination, etc.).
"""

from pathlib import Path

from noctyl.ingestion import run_pipeline_on_directory

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"


def _node_names(d):
    return {n["name"] for n in d["nodes"]}


def _find_graph(results, predicate):
    for g in results:
        if predicate(g):
            return g
    return None


def test_golden_pipeline_runs():
    """run_pipeline_on_directory on golden dir returns results and does not crash."""
    results, warnings = run_pipeline_on_directory(GOLDEN_DIR)
    # 8 files, multiple_graphs.py yields 2 graphs -> 9 graphs total
    assert len(results) >= 8, "expected at least 8 graphs (7 files + 2 from multiple_graphs)"
    assert all("schema_version" in g and "nodes" in g for g in results)


def test_golden_linear_workflow():
    """Linear: START -> A -> B -> END."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    g = _find_graph(results, lambda d: _node_names(d) == {"a", "b"} and not d["conditional_edges"] and d["entry_point"] == "a" and ("b", "END") in [(e["source"], e["target"]) for e in d["edges"]])
    assert g is not None
    assert g["entry_point"] == "a"
    assert _node_names(g) == {"a", "b"}
    assert g["terminal_nodes"] == ["b"]
    assert len(g["conditional_edges"]) == 0
    edges_sources_targets = [(e["source"], e["target"]) for e in g["edges"]]
    assert ("a", "b") in edges_sources_targets
    assert ("b", "END") in edges_sources_targets


def test_golden_conditional_loop():
    """Conditional loop: A has conditional_edges loop/next/done, one target END."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    g = _find_graph(results, lambda d: len(d["conditional_edges"]) >= 1 and any(e["target"] == "END" for e in d["conditional_edges"]) and "a" in _node_names(d) and d["entry_point"] == "a")
    assert g is not None
    assert g["entry_point"] == "a"
    cond_labels = {e["condition_label"] for e in g["conditional_edges"]}
    assert "done" in cond_labels
    assert "a" in g["terminal_nodes"] or "b" in g["terminal_nodes"]


def test_golden_end_termination():
    """END termination only: single node A, START -> A -> END."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    g = _find_graph(results, lambda d: _node_names(d) == {"a"} and len(d["nodes"]) == 1 and d["entry_point"] == "a" and d["terminal_nodes"] == ["a"] and any(e["target"] == "END" for e in d["edges"]))
    assert g is not None
    assert g["entry_point"] == "a"
    assert g["terminal_nodes"] == ["a"]
    assert len(g["edges"]) >= 2  # START->a, a->END


def test_golden_multiple_graphs():
    """One file yields two graphs with distinct node sets."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    by_nodes = {}
    for g in results:
        names = frozenset(_node_names(g))
        by_nodes.setdefault(names, []).append(g)
    # graph1: x,y  graph2: p,q
    assert frozenset({"x", "y"}) in by_nodes
    assert frozenset({"p", "q"}) in by_nodes
    assert len(by_nodes[frozenset({"x", "y"})]) >= 1
    assert len(by_nodes[frozenset({"p", "q"})]) >= 1


def test_golden_set_entry_point_explicit():
    """Explicit set_entry_point('b') -> entry_point is b."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    g = _find_graph(results, lambda d: d["entry_point"] == "b" and _node_names(d) == {"a", "b"})
    assert g is not None
    assert g["entry_point"] == "b"


def test_golden_mixed_linear_conditional():
    """Linear chain + conditional from last node; terminal_nodes from conditional to END."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    g = _find_graph(results, lambda d: "b" in _node_names(d) and "c" in _node_names(d) and len(d["conditional_edges"]) >= 1 and len(d["edges"]) >= 2)
    assert g is not None
    assert "b" in g["terminal_nodes"]
    cond_sources = {e["source"] for e in g["conditional_edges"]}
    assert "b" in cond_sources


def test_golden_single_node():
    """Single node a: entry_point a, terminal_nodes [a]."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    g = _find_graph(results, lambda d: len(d["nodes"]) == 1 and _node_names(d) == {"a"} and d["terminal_nodes"] == ["a"])
    assert g is not None
    assert g["entry_point"] == "a"


def test_golden_multiple_conditional_nodes():
    """Two nodes each with conditional_edges; both can transition to END."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    # Must have conditional_edges from two different sources (a and b)
    g = _find_graph(
        results,
        lambda d: _node_names(d) == {"a", "b"}
        and len(d["conditional_edges"]) >= 2
        and {e["source"] for e in d["conditional_edges"]} == {"a", "b"},
    )
    assert g is not None
    cond_sources = {e["source"] for e in g["conditional_edges"]}
    assert cond_sources == {"a", "b"}
    assert set(g["terminal_nodes"]) == {"a", "b"}
