---

## name: StateGraph Tracking
about: Track StateGraph instantiation and variable binding
title: "[Phase-1] Track StateGraph instances"
labels: graph, design, phase-1

## Description

Identify and track variables that hold `StateGraph` instances.

## Considerations

- Multiple StateGraph instances per file
- Stable internal IDs for graphs

## Acceptance Criteria

- All StateGraph instances detected
- Each instance tracked independently

