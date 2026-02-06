---
name: Node Extraction
about: Extract LangGraph nodes from add_node calls
title: "[Phase-1] Extract LangGraph nodes"
labels: parser, graph, phase-1
---

## Description
Extract workflow nodes defined via `add_node(name, callable)`.

## Data to Extract
- Node name
- Callable reference (string only)

## Out of Scope
- Inspecting callable internals

## Acceptance Criteria
- [ ] All add_node calls extracted
- [ ] Node metadata stored correctly
