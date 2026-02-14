---
name: Phase 2 – Task 4 Tests
description: Add tests for control_flow, metrics, annotation, risk, and golden integration
title: "[Phase 2] Add tests for analysis and golden integration"
labels: ["phase-2", "testing"]
assignees: []
---

## Summary

Add unit tests for control-flow, metrics, node annotation, structural risk, and integration tests using golden fixtures (e.g. conditional_loop, linear_workflow).

**Part of:** Phase 2 Implementation epic (create/link main Phase 2 issue).

## Scope

- **Control-flow:** Cycle detection and classification (cyclic vs acyclic fixtures), termination reachability, graph shape (linear, branching, cyclic).
- **Metrics:** Basic and path metrics on small known graphs.
- **Node annotation:** Origin/state/role with fixtures that have resolvable callables.
- **Structural risk:** Unreachable nodes, dead ends, non-terminating cycles; tests with and without entry_point.
- **Golden integration:** Run full pipeline + analyzer on [tests/fixtures/golden/](tests/fixtures/golden/) (e.g. conditional_loop.py, linear_workflow.py) and assert expected shape, cycle presence, and enriched keys.

## Acceptance criteria

- [ ] Unit tests for each analysis module (control_flow, metrics, node_annotation, structural_risk).
- [ ] Golden integration test: conditional_loop yields cyclic shape and expected cycle; linear_workflow yields linear shape.
- [ ] No flaky or runtime-dependent tests; all static/deterministic.

## References

- [tests/fixtures/golden/](tests/fixtures/golden/)
- [docs/phase/phase2.md](docs/phase/phase2.md) §10
