---
name: Phase 2 – Task 2 Analysis Package
description: Add noctyl/analysis with digraph, control_flow, metrics, node_annotation, structural_risk, GraphAnalyzer
title: "[Phase 2] Add analysis package and GraphAnalyzer"
labels: ["phase-2", "enhancement"]
assignees: []
---

## Summary

Add the `noctyl/analysis/` package: digraph builder, control-flow (SCC, cycles, termination, shape), metrics, node annotation, structural risk, and GraphAnalyzer.analyze().

**Part of:** Phase 2 Implementation epic (create/link main Phase 2 issue).

## Scope

- **digraph:** Build directed graph from WorkflowGraph (nodes = node names + START + END; edges from edges + conditional_edges).
- **control_flow:** SCC/cycle detection, cycle classification (self_loop, multi_node, conditional, non_terminating), termination reachability, graph shape (linear, branching, cyclic, disconnected, invalid).
- **metrics:** Basic (node_count, edge_count, entry, terminals, unreachable) and path (longest_acyclic_path, average_branching_factor, max_depth_before_cycle).
- **node_annotation:** Callable origin, state interaction, role heuristic (requires source + file_path in analyze()).
- **structural_risk:** Unreachable nodes, dead ends, non-terminating cycles, multiple entry points.
- **analyzer:** `GraphAnalyzer.analyze(workflow_graph, *, source=None, file_path=None) -> ExecutionModel`.

## Acceptance criteria

- [ ] All analysis is static and deterministic; no execution or token logic.
- [ ] GraphAnalyzer returns ExecutionModel; node_annotation uses source/file_path when provided.
- [ ] Public API exposed via `noctyl/analysis/__init__.py` (e.g. `analyze`, `GraphAnalyzer`).

## References

- [docs/phase/phase2.md](docs/phase/phase2.md) §§5–7
- [noctyl/graph/execution_model.py](noctyl/graph/execution_model.py) (Task 1)
