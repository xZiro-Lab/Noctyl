---
name: Phase 2 Implementation (Epic)
description: Implement Phase 2 Graph Enrichment and Execution Semantics per docs/phase/phase2.md
title: "Implement Phase 2: Graph Enrichment and Execution Semantics"
labels: ["phase-2", "enhancement"]
assignees: []
---

## Summary

Implement Phase 2 as specified in [docs/phase/phase2.md](docs/phase/phase2.md): enrich the LangGraph workflow graph (from Phase 1) with control-flow analysis, cycle detection, graph shape classification, structural metrics, node semantic annotation, an ExecutionModel abstraction, and structural risk detection. Output an enriched JSON schema. No token estimation, cost, or runtime behavior.

## Scope (in)

- Control-flow: cycle detection (SCC), cycle classification (self-loop, multi-node, conditional, non-terminating), termination reachability.
- Graph shape: linear / branching / cyclic / disconnected / invalid (multi-entry).
- Structural metrics: node/edge count, entry, terminals, unreachable nodes; longest acyclic path, average branching factor, max depth before cycle.
- Node semantic annotation: callable origin (local_function, imported_function, class_method, lambda, unknown); state interaction (pure, read_only, mutates_state); role heuristic (llm_like, tool_like, control_node, unknown).
- ExecutionModel: single dataclass holding graph, entry_point, terminal_nodes, cycles, metrics, node_annotations (and optionally risks).
- Structural risk: unreachable nodes, dead ends, non-terminating cycles, multiple entry points.
- Enriched JSON output (deterministic, schema documented).

## Scope (out)

- Token estimation, cost modeling, prompt extraction, memory estimation, AI assistant integration, multi-framework support, runtime tracing, probabilistic modeling.

## Acceptance criteria

- [ ] Cycles detected and classified reliably (unit tests with cyclic and acyclic golden fixtures).
- [ ] Graph shape classification correct for linear, branching, cyclic, and conditional_loop fixtures.
- [ ] Structural metrics computed and exposed on ExecutionModel.
- [ ] Node semantic tags (origin, state interaction, role) attached per node where determinable.
- [ ] ExecutionModel abstraction implemented and consumed by a single GraphAnalyzer entry point.
- [ ] Structural risks (unreachable, dead ends, non-terminating cycles, multi-entry) reported.
- [ ] Enriched JSON output stable and documented; no token/cost fields.
- [ ] All analysis static and deterministic; LangGraph-only.

## Implementation tasks

- [ ] **Task 1:** Add `noctyl/graph/execution_model.py`: ExecutionModel dataclass and `execution_model_to_dict()`.
- [ ] **Task 2:** Add `noctyl/analysis/` package: digraph builder from WorkflowGraph, control_flow (SCC, cycles, termination, shape), metrics, node_annotation (origin + state + role), structural_risk, and GraphAnalyzer.analyze(wg, source=..., file_path=...).
- [ ] **Task 3:** Integrate analyzer into pipeline: after building WorkflowGraph, run GraphAnalyzer and produce enriched output (preserve backward compatibility for existing dict output if needed).
- [ ] **Task 4:** Add tests: control flow, metrics, annotation, risk, and golden integration (e.g. conditional_loop, linear_workflow).
- [ ] **Task 5:** Update documentation (architecture, graph model, Phase 2 status).

## References

- Design: [docs/phase/phase2.md](docs/phase/phase2.md)
- Phase 1 graph: [noctyl/graph/graph.py](noctyl/graph/graph.py), [noctyl/ingestion/pipeline.py](noctyl/ingestion/pipeline.py)
- Golden fixtures: [tests/fixtures/golden/](tests/fixtures/golden/)
