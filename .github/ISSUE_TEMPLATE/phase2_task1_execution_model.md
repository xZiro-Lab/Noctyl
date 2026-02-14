---
name: Phase 2 – Task 1 ExecutionModel
description: Add ExecutionModel dataclass and execution_model_to_dict (Phase 2)
title: "[Phase 2] Add ExecutionModel and enriched schema"
labels: ["phase-2", "enhancement"]
assignees: []
---

## Summary

Add `noctyl/graph/execution_model.py` with the ExecutionModel dataclass and `execution_model_to_dict()` for Phase 2 enriched output.

**Part of:** Phase 2 Implementation epic (create/link main Phase 2 issue).

## Scope

- Define types: `DetectedCycle`, `StructuralMetrics`, `NodeAnnotation`, `StructuralRisk`, `ExecutionModel` (see [docs/phase/phase2.md](docs/phase/phase2.md) §§5–9).
- Implement `execution_model_to_dict(model: ExecutionModel)` returning a deterministic JSON-serializable dict (schema_version 2.0, `enriched: true`, plus cycles, metrics, node_annotations, risks).
- Export from `noctyl.graph` (update `noctyl/graph/__init__.py`).

## Acceptance criteria

- [ ] `ExecutionModel` holds graph, shape, cycles, metrics, node_annotations, risks.
- [ ] `execution_model_to_dict()` is deterministic and includes base graph (via `workflow_graph_to_dict`) plus enriched fields.
- [ ] No token/cost fields; LangGraph-only.

## References

- [docs/phase/phase2.md](docs/phase/phase2.md) §§8–9
- [noctyl/graph/graph.py](noctyl/graph/graph.py)
