# Noctyl -- Phase-2 Design Document

## Graph Enrichment & Execution Semantics

------------------------------------------------------------------------

## Status

Draft -- Phase-2

## Owner

Noctyl Core

------------------------------------------------------------------------

# 1. Phase-2 Scope

## In Scope

Phase-2 is strictly limited to structural and semantic enrichment of the
LangGraph workflow extracted in Phase-1.

The following are included:

-   Control-flow analysis
-   Cycle detection
-   Loop classification
-   Termination reachability checks
-   Graph shape classification
-   Structural metrics computation
-   Node semantic annotation
-   ExecutionModel abstraction
-   Structural risk detection
-   Enriched JSON output schema

All analysis must be: - Static (no runtime execution) - Deterministic -
Framework-specific to LangGraph only

------------------------------------------------------------------------

## Explicitly Out of Scope

The following are NOT part of Phase-2:

-   Token estimation
-   Cost modeling
-   Prompt size extraction
-   LLM output modeling
-   Memory size estimation
-   AI assistant integration
-   Multi-framework support
-   Runtime tracing or execution
-   Probabilistic modeling

Any feature touching token counting or cost estimation belongs to
Phase-3.

------------------------------------------------------------------------

# 2. Overview

Phase-1 provided structural extraction of LangGraph workflows: - Nodes -
Edges - Entry point - Terminal handling

Phase-2 enriches this graph with execution semantics and structural
analysis.

This phase focuses strictly on structural reasoning.\
No token estimation or cost modeling is included.

------------------------------------------------------------------------

# 3. Objectives

Phase-2 introduces:

-   Control-flow analysis
-   Cycle detection
-   Graph shape classification
-   Structural metrics computation
-   Node semantic tagging
-   ExecutionModel abstraction
-   Structural risk detection

------------------------------------------------------------------------

# 4. Architecture

GraphExtractor (Phase-1) ↓ GraphAnalyzer (Phase-2) ↓ ExecutionModel ↓
Enriched Graph Output

------------------------------------------------------------------------

# 5. Control Flow Analysis

## 5.1 Cycle Detection

### Purpose

Identify cyclic execution paths.

### Method

Use graph algorithms: - Strongly Connected Components (SCC) - Cycle
detection

### Classification

Cycles must be categorized as: - Self-loop - Multi-node cycle -
Conditional cycle - Non-terminating cycle

------------------------------------------------------------------------

## 5.2 Termination Reachability

For each detected cycle: - Check if a path to terminal node exists -
Flag cycles without termination

------------------------------------------------------------------------

## 5.3 Graph Shape Classification

Classify workflow as:

-   linear
-   branching
-   cyclic
-   disconnected
-   invalid (multi-entry)

------------------------------------------------------------------------

# 6. Structural Metrics

## 6.1 Basic Metrics

Compute:

-   Node count
-   Edge count
-   Entry node
-   Terminal nodes
-   Unreachable nodes

------------------------------------------------------------------------

## 6.2 Path Metrics

Compute:

-   Longest acyclic path
-   Average branching factor
-   Maximum depth before cycle

------------------------------------------------------------------------

# 7. Node Semantic Annotation

## 7.1 Callable Origin Classification

For each node callable, determine:

-   local_function
-   imported_function
-   class_method
-   lambda
-   unknown

------------------------------------------------------------------------

## 7.2 State Interaction Detection

Analyze callable AST for:

-   State read
-   State mutation
-   State update

Classify node:

-   pure
-   read_only
-   mutates_state

------------------------------------------------------------------------

## 7.3 Role Tagging (Heuristic)

Tag nodes as:

-   llm_like
-   tool_like
-   control_node
-   unknown

------------------------------------------------------------------------

# 8. ExecutionModel Abstraction

Phase-2 introduces a canonical internal representation:

class ExecutionModel: graph entry_point terminal_nodes cycles metrics
node_annotations

ExecutionModel becomes the foundation for Phase-3.

------------------------------------------------------------------------

# 9. Structural Risk Detection

## 9.1 Unreachable Nodes

Nodes not reachable from entry.

------------------------------------------------------------------------

## 9.2 Dead Ends

Nodes with no outgoing edges and not terminal.

------------------------------------------------------------------------

## 9.3 Non-Terminating Cycles

Cycles that cannot reach END.

------------------------------------------------------------------------

## 9.4 Multiple Entry Points

If detected, emit warning.

------------------------------------------------------------------------

# 10. Success Criteria

Phase-2 is complete when:

-   Cycles detected reliably
-   Graph shape classification works
-   Structural metrics computed
-   Node semantic tags added
-   ExecutionModel abstraction implemented
-   Enriched JSON output stable
-   No token estimation included

------------------------------------------------------------------------

# 11. Deliverables

-   GraphAnalyzer module
-   ExecutionModel class
-   Structural metrics engine
-   Risk detection layer
-   Enriched graph output
-   Updated test suite
-   Documentation
