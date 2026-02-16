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


def test_structural_risk_non_terminating_cycle_ids():
    """Graph with cycle that cannot reach END has non-empty non_terminating_cycle_ids."""
    wg = _cyclic_wg()
    model = analyze(wg)
    assert len(model.cycles) == 1
    assert model.cycles[0].reaches_terminal is False
    assert len(model.risks.non_terminating_cycle_ids) >= 1


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


# ── Digraph edge cases ────────────────────────────────────────────────────


def test_digraph_predecessors():
    """predecessors returns correct incoming sources."""
    wg = _linear_wg()
    dg = build_digraph(wg)
    preds_b = dg.predecessors("b")
    assert "a" in preds_b
    preds_c = dg.predecessors("c")
    assert "b" in preds_c
    preds_a = dg.predecessors("a")
    assert "START" in preds_a


def test_digraph_out_degree():
    """out_degree reflects successor count."""
    wg = _branching_wg()
    dg = build_digraph(wg)
    assert dg.out_degree("a") == 2  # a -> b, a -> c
    assert dg.out_degree("b") == 0
    assert dg.out_degree("c") == 0


def test_digraph_no_duplicate_start_edge():
    """build_digraph never adds duplicate START -> entry_point edges."""
    # even when wg.edges already has START -> entry_point
    nodes = [ExtractedNode("a", "f", 1)]
    edges = [ExtractedEdge("START", "a", 2), ExtractedEdge("a", "END", 3)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    starts_to_a = [s for s in dg.successors("START") if s == "a"]
    assert len(starts_to_a) == 1, "exactly one START -> entry_point edge"


def test_digraph_nodes_include_isolated():
    """Isolated nodes (no edges) still appear in digraph nodes."""
    nodes = [ExtractedNode("x", "f", 1), ExtractedNode("y", "g", 2)]
    edges = [ExtractedEdge("x", "END", 3)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "x")
    dg = build_digraph(wg)
    assert "y" in dg.nodes()


# ── Control-flow edge cases ──────────────────────────────────────────────


def test_control_flow_disconnected():
    """Graph with unreachable node but only one inferred entry is disconnected;
    when _infer_entry_nodes also picks up the orphan (no internal predecessors)
    the shape may be invalid due to multiple entries."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("orphan", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 4),
        ExtractedEdge("b", "END", 5),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    # orphan has no internal predecessors -> considered an extra entry -> invalid
    assert shape == "invalid"
    assert len(cycles) == 0


def test_control_flow_single_node_to_end():
    """Single node with edge to END is linear."""
    nodes = [ExtractedNode("a", "f", 1)]
    edges = [ExtractedEdge("a", "END", 2)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "linear"
    assert len(cycles) == 0


def test_control_flow_two_independent_sccs():
    """Two separate cycles; c,d are unreachable from START -> shape is disconnected."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
        ExtractedNode("d", "i", 4),
    ]
    edges = [
        ExtractedEdge("a", "b", 5),
        ExtractedEdge("b", "a", 6),
        ExtractedEdge("c", "d", 7),
        ExtractedEdge("d", "c", 8),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    # c and d are unreachable from START, so disconnected beats cyclic
    assert shape == "disconnected"
    assert len(cycles) == 2
    all_cycle_nodes = {n for c in cycles for n in c.nodes}
    assert all_cycle_nodes == {"a", "b", "c", "d"}


def test_control_flow_self_loop_with_terminal():
    """Self-loop with a path to END is self_loop (not non_terminating)."""
    nodes = [ExtractedNode("a", "f", 1)]
    edges = [ExtractedEdge("a", "a", 2), ExtractedEdge("a", "END", 3)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert cycles[0].cycle_type == "self_loop"
    assert cycles[0].reaches_terminal is True


def test_control_flow_conditional_self_loop():
    """Conditional self-loop edge classifies as conditional cycle."""
    nodes = [ExtractedNode("a", "f", 1)]
    edges = []
    cond = [
        ExtractedConditionalEdge("a", "retry", "a", 2),
        ExtractedConditionalEdge("a", "done", "END", 3),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, cond, "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert cycles[0].cycle_type == "conditional"
    assert cycles[0].reaches_terminal is True


# ── Metrics edge cases ───────────────────────────────────────────────────


def test_metrics_empty_graph():
    """Metrics on a graph with zero nodes returns zeroes."""
    wg = build_workflow_graph("id:0", [], [], [], None)
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert metrics.node_count == 0
    assert metrics.edge_count == 0
    assert metrics.entry_node is None
    assert metrics.unreachable_nodes == ()
    assert metrics.avg_branching_factor == 0.0
    assert metrics.max_depth_before_cycle is None


def test_metrics_with_cycle_max_depth():
    """max_depth_before_cycle is non-None when cycles exist and reachable from START."""
    wg = _conditional_loop_wg()
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert metrics.max_depth_before_cycle is not None
    assert metrics.max_depth_before_cycle >= 0


def test_metrics_longest_acyclic_path_linear():
    """Longest acyclic path for linear graph a -> b -> c is >= 2."""
    wg = _linear_wg()
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    # a -> b -> c -> END is 3 edges starting from a (entry); path a->b->c is 2 hops
    assert metrics.longest_acyclic_path >= 2


def test_metrics_avg_branching_factor_branching():
    """Branching graph has avg_branching_factor > 0."""
    wg = _branching_wg()
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    # a has out_degree 2, b and c have out_degree 0 => avg = 2/3 ≈ 0.67
    assert metrics.avg_branching_factor > 0.0
    assert metrics.avg_branching_factor == round(2 / 3, 2)


# ── Structural risk edge cases ───────────────────────────────────────────


def test_structural_risk_clean_graph():
    """Linear graph with proper termination has no risks."""
    wg = _linear_wg()
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    risks = compute_structural_risk(wg, metrics, shape, cycles, dg)
    assert risks.unreachable_node_ids == ()
    assert risks.dead_end_ids == ()
    assert risks.non_terminating_cycle_ids == ()
    assert risks.multiple_entry_points is False


def test_structural_risk_unreachable_plus_dead_end():
    """Graph with both an unreachable node and a dead-end node flags both risks."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("orphan", "h", 3),
    ]
    edges = [ExtractedEdge("a", "b", 4)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    risks = compute_structural_risk(wg, metrics, shape, cycles, dg)
    assert "orphan" in risks.unreachable_node_ids
    assert "b" in risks.dead_end_ids  # b has no outgoing edge and isn't terminal


# ── Node annotation edge cases ──────────────────────────────────────────


def test_node_annotation_syntax_error_source():
    """Syntax error in source falls back to unknown for all fields."""
    nodes = [ExtractedNode("a", "f", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "a")
    anns = compute_node_annotations(wg, source="def f(  # unclosed", file_path="x.py")
    assert len(anns) == 1
    assert anns[0].origin == "unknown"
    assert anns[0].state_interaction == "unknown"
    assert anns[0].role == "unknown"


def test_node_annotation_role_llm_like():
    """Callable ref containing 'llm' triggers llm_like role."""
    nodes = [ExtractedNode("n", "call_llm", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def call_llm(state): return state"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].role == "llm_like"


def test_node_annotation_role_tool_like():
    """Callable ref containing 'tool' triggers tool_like role."""
    nodes = [ExtractedNode("n", "use_tool", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def use_tool(state): return state"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].role == "tool_like"


def test_node_annotation_role_control_node():
    """Callable ref containing 'router' triggers control_node role."""
    nodes = [ExtractedNode("n", "router_fn", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def router_fn(state): return 'next'"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].role == "control_node"


def test_node_annotation_from_import_style():
    """'from X import Y' style import gets imported_function origin."""
    nodes = [ExtractedNode("n", "summarize", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "from tools import summarize\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].origin == "imported_function"


def test_node_annotation_cls_method():
    """cls.method callable_ref is class_method."""
    nodes = [ExtractedNode("n", "cls.dispatch", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "class C:\n    @classmethod\n    def dispatch(cls, state): pass\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].origin == "class_method"


def test_node_annotation_pure_function():
    """Function that doesn't reference state at all is pure."""
    nodes = [ExtractedNode("n", "f", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def f(x):\n    return x + 1\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].state_interaction == "pure"


def test_node_annotation_read_only_state():
    """Function that reads state but doesn't assign is read_only."""
    nodes = [ExtractedNode("n", "f", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def f(state):\n    return state['key']\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].state_interaction == "read_only"


def test_node_annotation_multiple_nodes_sorted():
    """Annotations for multiple nodes come back sorted by name regardless of input order."""
    nodes = [
        ExtractedNode("z_node", "fz", 1),
        ExtractedNode("a_node", "fa", 2),
        ExtractedNode("m_node", "fm", 3),
    ]
    wg = build_workflow_graph("id:0", nodes, [], [], "a_node")
    anns = compute_node_annotations(wg, source=None, file_path=None)
    names = [a.node_name for a in anns]
    assert names == ["a_node", "m_node", "z_node"]


# ── Analyzer edge cases ─────────────────────────────────────────────────


def test_analyzer_cyclic_with_source_annotations():
    """Analyzer on cyclic graph with source sets both shape and annotations."""
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [ExtractedEdge("a", "b", 3), ExtractedEdge("b", "a", 4)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    source = "def f(state):\n    state['x'] = 1\n    return state\ndef g(state):\n    return state['x']\n"
    model = analyze(wg, source=source, file_path="test.py")
    assert model.shape == "cyclic"
    ann_map = {a.node_name: a for a in model.node_annotations}
    assert ann_map["a"].state_interaction == "mutates_state"
    assert ann_map["b"].state_interaction == "read_only"


def test_analyzer_empty_graph():
    """Analyzer on empty graph (no nodes, no edges) produces valid model.
    Empty graph with no workflow nodes is classified as linear (no branching, no cycles)."""
    wg = build_workflow_graph("id:0", [], [], [], None)
    model = analyze(wg)
    assert model.shape == "linear"  # empty graph: no branches, no cycles
    assert model.metrics.node_count == 0
    assert model.metrics.edge_count == 0
    assert model.cycles == ()
    assert model.node_annotations == ()


def test_analyzer_file_path_passed_through():
    """file_path parameter reaches node annotations."""
    wg = _linear_wg()
    source = "def f(x): pass\ndef g(x): pass\ndef h(x): pass"
    model = analyze(wg, source=source, file_path="/some/path.py")
    # file_path is just forwarded; annotations should still resolve
    assert any(a.origin == "local_function" for a in model.node_annotations)


def test_analyzer_enriched_dict_roundtrip():
    """Full pipeline: analyze -> to_dict -> json -> parse -> verify all keys."""
    wg = _conditional_loop_wg()
    model = analyze(wg)
    d = execution_model_to_dict(model)
    s = json.dumps(d, sort_keys=True)
    parsed = json.loads(s)
    assert parsed["schema_version"] == "2.0"
    assert parsed["enriched"] is True
    assert parsed["shape"] == "cyclic"
    assert len(parsed["cycles"]) >= 1
    assert parsed["metrics"]["node_count"] == 2
    assert len(parsed["node_annotations"]) == 2
    assert isinstance(parsed["risks"]["multiple_entry_points"], bool)


def test_analyzer_branching_no_cycles():
    """Branching graph has correct shape and zero cycles."""
    wg = _branching_wg()
    model = analyze(wg)
    assert model.shape == "branching"
    assert model.cycles == ()
    assert model.metrics.node_count == 3


def test_analyzer_disconnected_risks():
    """Graph with orphan node (no internal predecessors) → invalid shape with unreachable risks."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("orphan", "h", 3),
    ]
    edges = [ExtractedEdge("a", "b", 4), ExtractedEdge("b", "END", 5)]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    model = analyze(wg)
    # orphan has no internal predecessors → treated as extra entry → invalid
    assert model.shape == "invalid"
    assert "orphan" in model.risks.unreachable_node_ids
    assert "orphan" in model.metrics.unreachable_nodes


# ── Control-flow stress tests ────────────────────────────────────────────


def test_control_flow_diamond():
    """Diamond: a -> {b, c} -> d -> END is branching, no cycles."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
        ExtractedNode("d", "i", 4),
    ]
    edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("a", "c", 11),
        ExtractedEdge("b", "d", 12),
        ExtractedEdge("c", "d", 13),
        ExtractedEdge("d", "END", 14),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "branching"
    assert len(cycles) == 0


def test_control_flow_deep_chain_tail_cycle():
    """Long chain a -> b -> c -> d, then d -> c forms a tail cycle."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
        ExtractedNode("d", "i", 4),
    ]
    edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("b", "c", 11),
        ExtractedEdge("c", "d", 12),
        ExtractedEdge("d", "c", 13),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert set(cycles[0].nodes) == {"c", "d"}


def test_control_flow_three_node_scc_with_exit():
    """3-node SCC (a -> b -> c -> a) with exit c -> END reaches terminal."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("b", "c", 11),
        ExtractedEdge("c", "a", 12),
        ExtractedEdge("c", "END", 13),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) == 1
    assert set(cycles[0].nodes) == {"a", "b", "c"}
    assert cycles[0].reaches_terminal is True
    assert cycles[0].cycle_type == "multi_node"


def test_control_flow_long_linear_chain():
    """Chain of 20 nodes is linear, no cycles."""
    n = 20
    nodes = [ExtractedNode(f"n{i:02d}", "f", i) for i in range(n)]
    edges = [ExtractedEdge(f"n{i:02d}", f"n{i+1:02d}", 100 + i) for i in range(n - 1)]
    edges.append(ExtractedEdge(f"n{n-1:02d}", "END", 200))
    wg = build_workflow_graph("id:0", nodes, edges, [], "n00")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "linear"
    assert len(cycles) == 0


def test_control_flow_cycle_plus_branch():
    """Graph with both a cycle and a branch: cyclic wins."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("a", "c", 11),
        ExtractedEdge("b", "a", 12),
        ExtractedEdge("c", "END", 13),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    shape, cycles = compute_control_flow(dg, wg)
    assert shape == "cyclic"
    assert len(cycles) >= 1


# ── Node annotation edge cases ──────────────────────────────────────────


def test_node_annotation_aliased_import():
    """'from X import Y as Z' — callable Z gets imported_function origin."""
    nodes = [ExtractedNode("n", "my_func", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "from tools import run as my_func\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].origin == "imported_function"


def test_node_annotation_unresolved_symbol():
    """Callable not in local funcs or imports stays unknown."""
    nodes = [ExtractedNode("n", "mystery_fn", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    source = "def other(x): pass\n"
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    assert anns[0].origin == "unknown"


def test_node_annotation_mixed_types_in_one_graph():
    """Graph with local, imported, lambda, and class_method nodes — all identified correctly."""
    nodes = [
        ExtractedNode("local_node", "my_local", 1),
        ExtractedNode("import_node", "ext_func", 2),
        ExtractedNode("lambda_node", "lambda", 3),
        ExtractedNode("method_node", "self.run", 4),
    ]
    wg = build_workflow_graph("id:0", nodes, [], [], "local_node")
    source = (
        "from external import ext_func\n"
        "def my_local(state): return state\n"
    )
    anns = compute_node_annotations(wg, source=source, file_path="x.py")
    ann_map = {a.node_name: a for a in anns}
    assert ann_map["local_node"].origin == "local_function"
    assert ann_map["import_node"].origin == "imported_function"
    assert ann_map["lambda_node"].origin == "lambda"
    assert ann_map["method_node"].origin == "class_method"


def test_node_annotation_empty_source_string():
    """Empty string source (not None) still gives unknown — no AST to parse."""
    nodes = [ExtractedNode("n", "f", 1)]
    wg = build_workflow_graph("id:0", nodes, [], [], "n")
    anns = compute_node_annotations(wg, source="", file_path="x.py")
    assert anns[0].origin == "unknown"


# ── Metrics edge cases ───────────────────────────────────────────────────


def test_metrics_diamond_graph():
    """Diamond: a->{b,c}->d->END — exact metric values."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
        ExtractedNode("d", "i", 4),
    ]
    edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("a", "c", 11),
        ExtractedEdge("b", "d", 12),
        ExtractedEdge("c", "d", 13),
        ExtractedEdge("d", "END", 14),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert metrics.node_count == 4
    assert metrics.edge_count == 5
    assert metrics.entry_node == "a"
    assert metrics.terminal_nodes == ("d",)
    assert metrics.unreachable_nodes == ()
    # a has 2 out, b has 1 out, c has 1 out, d has 1 out = 5 / 4 = 1.25
    assert metrics.avg_branching_factor == 1.25
    assert metrics.longest_acyclic_path >= 3  # a->b->d->END or a->c->d->END
    assert metrics.max_depth_before_cycle is None


def test_metrics_multiple_terminal_nodes():
    """Graph where two nodes independently reach END."""
    nodes = [
        ExtractedNode("a", "f", 1),
        ExtractedNode("b", "g", 2),
        ExtractedNode("c", "h", 3),
    ]
    edges = [
        ExtractedEdge("a", "b", 10),
        ExtractedEdge("a", "c", 11),
        ExtractedEdge("b", "END", 12),
        ExtractedEdge("c", "END", 13),
    ]
    wg = build_workflow_graph("id:0", nodes, edges, [], "a")
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert set(metrics.terminal_nodes) == {"b", "c"}
    assert metrics.entry_node == "a"


def test_metrics_deep_chain_exact_values():
    """Chain of 5 nodes: exact longest_acyclic_path and branching factor."""
    nodes = [ExtractedNode(f"n{i}", "f", i) for i in range(5)]
    edges = [ExtractedEdge(f"n{i}", f"n{i+1}", 10 + i) for i in range(4)]
    edges.append(ExtractedEdge("n4", "END", 20))
    wg = build_workflow_graph("id:0", nodes, edges, [], "n0")
    dg = build_digraph(wg)
    _, cycles = compute_control_flow(dg, wg)
    metrics = compute_metrics(dg, wg, cycles)
    assert metrics.node_count == 5
    assert metrics.edge_count == 5
    # Each of the 5 nodes has out_degree 1 (n0->n1, n1->n2, ..., n4->END)
    assert metrics.avg_branching_factor == 1.0
    # Longest simple path from n0: n0->n1->n2->n3->n4->END = 5 hops
    assert metrics.longest_acyclic_path >= 4
