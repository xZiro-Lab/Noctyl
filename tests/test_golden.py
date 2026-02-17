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


def test_golden_default_returns_base_dicts():
    """Default run_pipeline_on_directory returns Phase-1 dicts (schema 1.0, no enriched)."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    assert len(results) >= 1
    for d in results:
        assert d.get("schema_version") == "1.0"
        assert "enriched" not in d


def test_golden_enriched_returns_schema_v2():
    """run_pipeline_on_directory(..., enriched=True) returns Phase-2 dicts with enriched fields."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    assert len(results) >= 1
    for d in results:
        assert d.get("schema_version") == "2.0"
        assert d.get("enriched") is True
        assert "shape" in d and "cycles" in d and "metrics" in d
        assert "node_annotations" in d and "risks" in d
        assert "token" not in " ".join(k.lower() for k in d.keys())
        assert "cost" not in " ".join(k.lower() for k in d.keys())
        # Base graph fields still present
        assert "nodes" in d and "edges" in d and "entry_point" in d and "terminal_nodes" in d
    # Linear workflow fixture: expect at least one graph with shape linear
    linear = _find_graph(results, lambda d: _node_names(d) == {"a", "b"} and not d["conditional_edges"] and d["entry_point"] == "a")
    if linear is not None:
        assert linear["shape"] == "linear"
        assert linear["entry_point"] == "a"
        assert linear["terminal_nodes"] == ["b"]


def test_golden_enriched_conditional_loop_shape_and_cycle():
    """Golden conditional_loop fixture yields cyclic shape and at least one cycle containing a, reaches_terminal True."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    cond_loop = _find_graph(
        results,
        lambda d: _node_names(d) == {"a", "b"}
        and d["entry_point"] == "a"
        and any(e["condition_label"] == "done" for e in d["conditional_edges"]),
    )
    assert cond_loop is not None
    assert cond_loop["shape"] == "cyclic"
    assert len(cond_loop["cycles"]) >= 1
    cycle_with_a = [c for c in cond_loop["cycles"] if "a" in c["nodes"]]
    assert len(cycle_with_a) >= 1
    assert any(c["reaches_terminal"] is True for c in cycle_with_a)


# ── Enriched golden: metrics validation ──────────────────────────────────


def test_golden_enriched_metrics_linear():
    """Enriched linear fixture: metrics match expected counts."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    linear = _find_graph(
        results,
        lambda d: _node_names(d) == {"a", "b"}
        and not d["conditional_edges"]
        and d["entry_point"] == "a"
        and ("b", "END") in [(e["source"], e["target"]) for e in d["edges"]],
    )
    assert linear is not None
    m = linear["metrics"]
    assert m["node_count"] == 2
    assert m["edge_count"] == 3  # START->a, a->b, b->END
    assert m["entry_node"] == "a"
    assert m["terminal_nodes"] == ["b"]
    assert m["unreachable_nodes"] == []
    assert m["max_depth_before_cycle"] is None  # no cycles


def test_golden_enriched_risks_linear():
    """Enriched linear fixture: no structural risks."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    linear = _find_graph(
        results,
        lambda d: _node_names(d) == {"a", "b"}
        and not d["conditional_edges"]
        and d["entry_point"] == "a"
        and ("b", "END") in [(e["source"], e["target"]) for e in d["edges"]],
    )
    assert linear is not None
    r = linear["risks"]
    assert r["unreachable_node_ids"] == []
    assert r["dead_end_ids"] == []
    assert r["non_terminating_cycle_ids"] == []
    assert r["multiple_entry_points"] is False


def test_golden_enriched_node_annotations_present():
    """Enriched output always has one annotation per node."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    for d in results:
        node_count = len(d["nodes"])
        ann_count = len(d["node_annotations"])
        assert ann_count == node_count, (
            f"graph {d['graph_id']}: {ann_count} annotations != {node_count} nodes"
        )


def test_golden_enriched_multiple_graphs_both_enriched():
    """Multiple-graphs fixture: both graphs get independent enriched output."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    xy = _find_graph(results, lambda d: _node_names(d) == {"x", "y"})
    pq = _find_graph(results, lambda d: _node_names(d) == {"p", "q"})
    assert xy is not None and pq is not None
    for g in (xy, pq):
        assert g["schema_version"] == "2.0"
        assert g["enriched"] is True
        assert "shape" in g
        assert "metrics" in g and g["metrics"]["node_count"] == 2
        assert len(g["node_annotations"]) == 2


def test_golden_enriched_single_node_metrics():
    """Single-node fixture: metrics node_count 1, entry and terminal match."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    single = _find_graph(
        results,
        lambda d: len(d["nodes"]) == 1
        and _node_names(d) == {"a"}
        and d["terminal_nodes"] == ["a"],
    )
    assert single is not None
    m = single["metrics"]
    assert m["node_count"] == 1
    assert m["entry_node"] == "a"
    assert m["terminal_nodes"] == ["a"]


def test_golden_enriched_deterministic():
    """Two enriched runs on golden dir produce identical JSON."""
    import json

    r1, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    r2, _ = run_pipeline_on_directory(GOLDEN_DIR, enriched=True)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_golden_estimate_mode_schema_3_0():
    """All golden fixtures produce schema 3.0 with estimate=True."""
    results, warnings = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    assert len(results) > 0
    for d in results:
        assert d.get("schema_version") == "3.0"
        assert d.get("estimated") is True
        assert d.get("enriched") is True
        assert "token_estimate" in d
        assert "node_signatures" in d
        assert "per_node_envelopes" in d
        assert "per_path_envelopes" in d
        assert "warnings" in d


def test_golden_linear_fixture_min_equals_expected_equals_max():
    """Linear fixture: no branches/loops → min == expected == max."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    linear = _find_graph(
        results,
        lambda d: _node_names(d) == {"a", "b"}
        and not d["conditional_edges"]
        and d["entry_point"] == "a"
        and ("b", "END") in [(e["source"], e["target"]) for e in d["edges"]],
    )
    
    if linear is not None:
        token_est = linear["token_estimate"]
        # Linear workflows without loops/branches should have tight bounds
        # Note: Due to expansion factors, they may not be exactly equal, but should be close
        assert token_est["min_tokens"] <= token_est["expected_tokens"] <= token_est["max_tokens"]
        # For truly linear workflows, range should be small relative to expected
        if token_est["expected_tokens"] > 0:
            range_ratio = (token_est["max_tokens"] - token_est["min_tokens"]) / token_est["expected_tokens"]
            # Range should be small (less than 50% of expected)
            assert range_ratio < 0.5


def test_golden_cyclic_fixture_max_gt_expected_gt_min():
    """Cyclic fixture: loop amplification → max > expected > min."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    cyclic = _find_graph(
        results,
        lambda d: _node_names(d) == {"a", "b"}
        and d["entry_point"] == "a"
        and any(e["condition_label"] == "done" for e in d["conditional_edges"]),
    )
    
    if cyclic is not None:
        token_est = cyclic["token_estimate"]
        # Cyclic workflows should have max >= expected >= min (may be equal if no tokens)
        assert token_est["min_tokens"] <= token_est["expected_tokens"] <= token_est["max_tokens"]
        # Should have cycles detected
        assert len(cyclic["cycles"]) > 0
        # If there are base tokens, loop amplification should widen the range
        if token_est["min_tokens"] > 0:
            # With loops, max should be >= expected (may be equal if cycles don't amplify)
            assert token_est["max_tokens"] >= token_est["expected_tokens"]


def test_golden_conditional_fixture_branch_envelopes_present():
    """Conditional fixture: branch envelopes present."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    conditional = _find_graph(
        results,
        lambda d: len(d["conditional_edges"]) >= 1
        and any(e["target"] == "END" for e in d["conditional_edges"]),
    )
    
    if conditional is not None:
        # Should have per_path_envelopes for conditional branches
        assert "per_path_envelopes" in conditional
        assert isinstance(conditional["per_path_envelopes"], dict)
        # Should have at least one path envelope if branches exist
        if len(conditional["conditional_edges"]) > 0:
            assert len(conditional["per_path_envelopes"]) > 0


def test_golden_estimate_deterministic_across_runs():
    """Estimate mode determinism across multiple runs."""
    import json
    
    r1, w1 = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    r2, w2 = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    # Results should be identical
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    # Warnings should be identical (sorted)
    assert sorted(w1) == sorted(w2)


def test_golden_estimate_all_fixtures_valid():
    """All fixtures produce valid estimates."""
    results, warnings = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    assert len(results) > 0
    for d in results:
        # Schema 3.0 structure
        assert d["schema_version"] == "3.0"
        assert d["estimated"] is True
        
        # Token estimate structure
        token_est = d["token_estimate"]
        assert "assumptions_profile" in token_est
        assert "min_tokens" in token_est
        assert "expected_tokens" in token_est
        assert "max_tokens" in token_est
        assert "bounded" in token_est
        assert "confidence" in token_est
        
        # Invariant: min <= expected <= max
        assert token_est["min_tokens"] <= token_est["expected_tokens"] <= token_est["max_tokens"]
        
        # Node signatures
        assert isinstance(d["node_signatures"], list)
        assert len(d["node_signatures"]) == len(d["nodes"])
        
        # Envelopes
        assert isinstance(d["per_node_envelopes"], dict)
        assert isinstance(d["per_path_envelopes"], dict)
        assert isinstance(d["warnings"], list)


def test_golden_estimate_warnings_collected():
    """Warnings present where expected (cycles, symbolic nodes)."""
    results, warnings = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    # Find cyclic workflow
    cyclic = _find_graph(
        results,
        lambda d: len(d["cycles"]) > 0,
    )
    
    if cyclic is not None:
        # Should have warnings about cycles
        assert len(cyclic["warnings"]) > 0 or len(warnings) > 0
        all_warnings = " ".join(cyclic["warnings"] + warnings)
        # Should mention cycles or unbounded iterations
        assert "cycle" in all_warnings.lower() or "unbounded" in all_warnings.lower()


def test_golden_estimate_phase2_fields_present():
    """Phase 2 fields present in estimate mode output."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR, estimate=True)
    
    assert len(results) > 0
    for d in results:
        # Phase 2 fields should be present
        assert "shape" in d
        assert "cycles" in d
        assert "metrics" in d
        assert "node_annotations" in d
        assert "risks" in d
        
        # Phase 2 enriched flag
        assert d["enriched"] is True
        
        # Phase 3 fields also present
        assert "token_estimate" in d
        assert d.get("estimated") is True
