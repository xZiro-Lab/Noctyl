---
name: LangGraph Detection
about: Detect LangGraph usage in a Python codebase
title: "[Phase-1] Detect LangGraph workflows in repository"
labels: parser, design, phase-1
---

## Description
Implement detection logic to identify whether a Python file contains LangGraph workflow definitions.

## Detection Signals
- Imports from `langgraph.graph`
- Instantiation of `StateGraph`

## Open Questions
- How to handle aliased imports?
- Should files without LangGraph be skipped early?

## Acceptance Criteria
- [ ] AST-based detection implemented
- [ ] Non-LangGraph files safely ignored
