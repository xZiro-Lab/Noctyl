# Noctyl --- Full Project Design (Phase 1 → Phase 3)

## Overview

Noctyl is a **static, pre-execution analysis system** that estimates
token usage of LangGraph-based multi-agent workflows **before
execution**.

It never: - Executes user code - Calls LLM APIs - Observes runtime
behavior

All estimates are derived strictly from **source structure + declared
assumptions**.

------------------------------------------------------------------------

## Core Design Principle

Noctyl behaves like a **compiler static analyzer**, not an observability
tool.

### Allowed Inputs

-   Source code (AST)
-   Static literals and configs
-   Structural workflow graph
-   Deterministic heuristics
-   User-declared model assumptions

### Explicitly Forbidden

-   Runtime execution
-   Token tracing
-   Simulation or replay
-   API calls to models
-   Calibration against live runs

------------------------------------------------------------------------

# Phase 1 --- Workflow Extraction

## Goal

Detect LangGraph workflows and construct a structural execution graph.

## Scope

-   Python only
-   LangGraph core API only
-   Static analysis only

## Output

Versioned JSON graph containing: - Nodes - Edges - Entry point -
Terminal nodes

## What Phase‑1 Answers

> What is the structure of this workflow?

------------------------------------------------------------------------

# Phase 2 --- Graph Enrichment

## Goal

Understand how the workflow behaves structurally.

## Adds

-   Cycle detection
-   Path analysis
-   Graph classification (linear / branching / cyclic)
-   Node semantic tagging
-   Structural risk detection

## Output

`ExecutionModel` --- an analyzable semantic representation.

## What Phase‑2 Answers

> How could this workflow execute structurally?

------------------------------------------------------------------------

# Phase 3 --- Static Token Estimation

## Goal

Estimate token growth using deterministic modeling --- still
pre‑execution.

## Adds

-   Node token signatures
-   Token propagation rules
-   Loop amplification modeling
-   Branch cost envelopes
-   Configurable model profiles

## Output

A bounded estimate:

    min_tokens
    expected_tokens
    max_tokens

Never a single-point prediction.

## What Phase‑3 Answers

> What scale of token cost is implied by this structure?

------------------------------------------------------------------------

# Architecture Evolution

    Phase 1: GraphExtractor
    Phase 2: GraphAnalyzer → ExecutionModel
    Phase 3: TokenModeler → Cost Envelope

------------------------------------------------------------------------

# Estimation Philosophy

Noctyl performs **static cost semantics**, similar to:

-   Time‑complexity analysis (`O(n)`)
-   Query planner estimation
-   Memory bound reasoning

It does **not** perform profiling.

------------------------------------------------------------------------

# Deliverables by Phase

  Phase     Deliverable
  --------- --------------------------------
  Phase‑1   Workflow graph extraction
  Phase‑2   Execution semantics model
  Phase‑3   Deterministic token estimation

------------------------------------------------------------------------

# What Noctyl Ultimately Enables

Noctyl allows engineers to answer:

> "If we run this agentic workflow, what scale of token cost are we
> committing to?"

**Without running anything.**
