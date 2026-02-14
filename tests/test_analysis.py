"""Tests for Phase 2 analysis package: digraph, control_flow, metrics, node_annotation, structural_risk, GraphAnalyzer."""

import itertools
import json

from noctyl.graph import (
    ExtractedConditionalEdge,
    ExtractedEdge,
    ExtractedNode,
    WorkflowGraph,
    build_workflow_graph,
    execution_model_to_dict,
)
from noctyl.analysis import (
    GraphAnalyzer,
    analyze,
    build_digraph,
    compute_control_flow,
    compute_metrics,
    compute_node_annotations,
    compute_structural_risk,
)


def _linear_wg() -> WorkflowGraph:
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("b", "c", 5),
        ExtractedEdge("c", "END", 6),
    ]
    return build_workflow_graph("id:0", nodes, edges, [], "a")


def _cyclic_wg() -> WorkflowGraph:
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [
        ExtractedEdge("a", "b", 3),
        ExtractedEdge("b", "a", 4),
    ]
    return build_workflow_graph("id:0", nodes, edges, [], "a")


def _self_loop_wg() -> WorkflowGraph:
    nodes = [ExtractedNode("a", "f", 1)]
    edges = [ExtractedEdge("a", "a", 2)]
    return build_workflow_graph("id:0", nodes, edges, [], "a")


def _branching_wg() -> WorkflowGraph:
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("a", "c", 5),
    ]
    return build_workflow_graph("id:0", nodes, edges, [], "a")


def _conditional_loop_wg() -> WorkflowGraph:
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
    ]
    edges = [ExtractedEdge("a", "b", 3)]
    cond = [
        ExtractedConditionalEdge("b", "loop", "a", 4),
        ExtractedConditionalEdge("b", "done", "END", 5),
    ]
    return build_workflow_graph("id:0", nodes, edges, cond, "a")


def test_digraph_build():
    """build_digraph includes START, END, workflow nodes, and START -> entry_point."""
    wg = _linear_wg()
    dg = build_digraph(wg)
    assert "START" in dg.nodes()
    assert "END" in dg.nodes()
    assert "a" in dg.nodes() and "b" in dg.nodes() and "c" in dg.nodes()
    assert "a" in dg.successors("START")
    assert "b" in dg.successors("a")
    assert "END" in dg.successors("c")


def test_digraph_conditional_edges():
    """Conditional edges are marked and present in graph."""
    wg = _conditional_loop_wg()
    dg = build_digraph(wg)
    assert dg.is_conditional_edge("b", "a") is True
    assert dg.is_conditional_edge("b", "END") is True
    assert "a" in dg.successors("b") and "END" in dg.successors("b")


def test_control_flow_linear():
    """Linear workflow has shape linear and no cycles."""
    wg = _linear_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "linear"
    assert len(cycles) == 0


def test_control_flow_cyclic():
    """Two-node cycle is detected; type is multi_node or non_terminating when no path to END."""
    wg = _cyclic_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert set(cycles[0].nodes) == {"a", "b"}
    assert cycles[0].cycle_type in ("multi_node", "non_terminating")


def test_control_flow_self_loop():
    """Self-loop is detected; type is self_loop or non_terminating when no path to END."""
    wg = _self_loop_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert cycles[0].nodes == ("a",)
    assert cycles[0].cycle_type in ("self_loop", "non_terminating")


def test_control_flow_branching():
    """Graph with out-degree > 1 is branching."""
    wg = _branching_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "branching"
    assert len(cycles) == 0


def test_control_flow_conditional_cycle():
    """Cycle that includes a conditional edge is conditional."""
    wg = _conditional_loop_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) >= 1
    cycle_with_b = [c for c in cycles if "b" in c.nodes]
    assert any(c.cycle_type == "conditional" for c in cycle_with_b)
    assert any(c.reaches_terminal is True for c in cycle_with_b)


def test_control_flow_non_terminating_cycle():
    """Cycle with no path to END is classified as non_terminating."""
    wg = _cyclic_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert cycles[0].cycle_type == "non_terminating"
    assert cycles[0].reaches_terminal is False


def test_control_flow_shape_invalid_no_entry():
    """Workflow with nodes but no entry_point is invalid."""
    nodes = [ExtractedNode("a", "f", 1)]
    edges = []
    wg = build_workflow_graph("id:0", nodes, edges, [], None)
    dg = build_digraph(wg)
    shape, _ = compute_control_flow(dg, wg)
    assert shape == "invalid"


def test_metrics_basic():
    """Metrics reflect node count, edge count, entry, terminals, unreachable."""
    wg = _linear_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert metrics.node_count == 3
    assert metrics.edge_count == 3
    assert metrics.entry_node == "a"
    assert metrics.terminal_nodes == ("c",)
    assert metrics.unreachable_nodes == ()
    assert metrics.longest_acyclic_path >= 2
    assert metrics.avg_branching_factor >= 0.0


def test_metrics_unreachable():
    """Node not reachable from START is in unreachable_nodes."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("orphan", "g", 2),
    ]
    edges = [ExtractedEdge("a", "END", 3)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert "orphan" in metrics.unreachable_nodes


def test_structural_risk_dead_end():
    """Node with no outgoing edge and not terminal is dead end."""
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [ExtractedEdge("a", "b", 3)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    risks = compute_structural_risk(wg, metrics, shape, cycles, dg)
    assert "b" in risks.dead_end_ids


def test_structural_risk_multiple_entries():
    """Invalid shape implies multiple_entry_points True."""
    nodes = [ExtractedNode("a", "f", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], None)
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    risks = compute_structural_risk(wg, metrics, shape, cycles, dg)
    assert risks.multiple_entry_points is True


def test_node_annotation_without_source():
    """Without source, all annotations are unknown."""
    wg = _linear_wg()
    anns = compute_node_annotations(wg, source=None, file_path=None)
    assert len(anns) == 3
    for a in anns:
        assert a.origin == "unknown"
        assert a.state_interaction == "unknown"
        assert a.role == "unknown"


def test_node_annotation_with_source_local():
    """With source defining a function, origin can be local_function."""
    wg = _linear_wg()
    source = """
def f(state):
    return state
def g(state):
    return state
def h(state):
    return state
"""
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert len(anns) == 3
    origins = {a.node_name: a.origin for a in anns}
    assert origins.get("a") == "local_function"
    assert origins.get("b") == "local_function"
    states = {a.node_name: a.state_interaction for a in anns}
    assert states.get("a") in ("read_only", "pure")


def test_node_annotation_lambda():
    """Node with callable_ref 'lambda' gets origin lambda."""
    nodes = [ExtractedNode("n", "lambda", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    anns = compute_node_annotations(wg, source="x = 1", file_path="x.py")
    assert len(anns) == 1
    assert anns[0].origin == "lambda"


def test_node_annotation_imported_dotted_not_class_method():
    """module.func callable_ref is imported_function, not class_method."""
    nodes = [ExtractedNode("n", "helpers.run", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "import helpers\n\ndef x():\n    return 1\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert len(anns) == 1
    assert anns[0].origin == "imported_function"


def test_node_annotation_class_method_self():
    """self.method callable_ref is class_method."""
    nodes = [ExtractedNode("n", "self.route", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "class C:\n    def route(self, state):\n        return state\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert len(anns) == 1
    assert anns[0].origin == "class_method"


def test_node_annotation_state_mutation_detected():
    """Assignment to state-like target marks mutates_state."""
    nodes = [ExtractedNode("n", "f", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def f(state):\n    state['x'] = 1\n    return state\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert len(anns) == 1
    assert anns[0].state_interaction == "mutates_state"


def test_analyzer_returns_execution_model():
    """analyze() returns ExecutionModel with all fields set."""
    wg = _linear_wg()
    model = analyze(wg)
    assert model.graph is wg
    assert model.entry_point == "a"
    assert model.terminal_nodes == ("c",)
    assert model.shape == "linear"
    assert model.metrics.node_count == 3
    assert len(model.node_annotations) == 3
    assert model.risks.multiple_entry_points is False


def test_analyzer_with_source():
    """analyze(wg, source=...) uses source for node annotations."""
    wg = _linear_wg()
    source = "def f(x): pass\ndef g(x): pass\ndef h(x): pass"
    model = analyze(wg, source=source)
    assert any(a.origin == "local_function" for a in model.node_annotations)


def test_analyzer_deterministic():
    """Same input produces same ExecutionModel (shape, cycles, metrics)."""
    wg = _cyclic_wg()
    m1 = analyze(wg)
    m2 = analyze(wg)
    assert m1.shape == m2.shape
    assert len(m1.cycles) == len(m2.cycles)
    assert m1.metrics.node_count == m2.metrics.node_count
    assert m1.risks == m2.risks


def test_graph_analyzer_class():
    """GraphAnalyzer().analyze() is equivalent to analyze()."""
    wg = _linear_wg()
    m1 = analyze(wg)
    m2 = GraphAnalyzer().analyze(wg)
    assert m1.shape == m2.shape
    assert m1.metrics.node_count == m2.metrics.node_count


def test_analyzer_deterministic_under_input_permutations():
    """Equivalent graphs with shuffled node/edge input orders produce identical enriched output."""
    base_nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    base_edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("b", "c", 11),
        ExtractedEdge("c", "END", 12),
    ]
    base_cond = [
        ExtractedConditionalEdge("b", "skip", "END", 13),
    ]

    outputs: set[str] = set()
    for nodes_perm in itertools.permutations(base_nodes):
        for edges_perm in itertools.permutations(base_edges):
            wg = build_workflow_graph(
                "id:det",
                list(nodes_perm),
                list(edges_perm),
                list(base_cond),
                "a",
            )
            model = analyze(wg)
            payload = execution_model_to_dict(model)
            outputs.add(json.dumps(payload, sort_keys=True))

    # All logical permutations should collapse to one deterministic output.
    assert len(outputs) == 1


def test_analyzer_deterministic_under_conditional_permutations():
    """Shuffling conditional-edge order does not change enriched output."""
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [ExtractedEdge("a", "b", 3)]
    cond = [
        ExtractedConditionalEdge("b", "loop", "a", 4),
        ExtractedConditionalEdge("b", "done", "END", 5),
    ]

    outputs: set[str] = set()
    for cond_perm in itertools.permutations(cond):
        wg = build_workflow_graph("id:cond", nodes, edges, list(cond_perm), "a")
        model = analyze(wg)
        payload = execution_model_to_dict(model)
        outputs.add(json.dumps(payload, sort_keys=True))

    assert len(outputs) == 1
