---
name: Phase 2 – Task 3 Pipeline Integration
description: Integrate GraphAnalyzer into pipeline and produce enriched output
title: "[Phase 2] Integrate analyzer into pipeline and enriched output"
labels: ["phase-2", "enhancement"]
assignees: []
---

## Summary

After building WorkflowGraph in the pipeline, call GraphAnalyzer.analyze() and serialize ExecutionModel to enriched dict. Preserve backward compatibility for existing workflow dict output where needed.

**Part of:** Phase 2 Implementation epic (create/link main Phase 2 issue).

## Scope

- In `noctyl/ingestion/pipeline.py` (or agreed entry point): for each WorkflowGraph, call `analyze(wg, source=source, file_path=file_path)` and obtain ExecutionModel.
- Serialize via `execution_model_to_dict(model)` and expose enriched result (e.g. return list of enriched dicts, or add option for enriched vs base output).
- Do not remove or break existing `workflow_graph_to_dict` behavior for callers that expect Phase-1-only output.

## Acceptance criteria

- [ ] Pipeline can produce enriched JSON (schema 2.0) when Phase 2 is used.
- [ ] Backward compatibility: existing callers can still get Phase-1-style dict if required (e.g. optional flag or separate function).
- [ ] Enriched output includes cycles, shape, metrics, node_annotations, risks; no token/cost.

## References

- [noctyl/ingestion/pipeline.py](noctyl/ingestion/pipeline.py)
- [noctyl/analysis/analyzer.py](noctyl/analysis/analyzer.py)
