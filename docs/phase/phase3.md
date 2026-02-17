# Noctyl -- Phase-3 Design Document

## Static, Pre-Execution Token Estimation (No Runtime, Ever)

------------------------------------------------------------------------

## Status

In Progress -- Phase-3

**Task 1 (Data Model):** Implemented ✓
- `noctyl/estimation/` package created
- `NodeTokenSignature`, `ModelProfile`, `CostEnvelope`, `WorkflowEstimate` dataclasses
- `workflow_estimate_to_dict()` serializer (schema 3.0)
- 31 tests in `tests/test_estimation_model.py`, all passing

## Owner

Noctyl Core

------------------------------------------------------------------------

# 0. Design Invariant (Applies to All Phases)

Noctyl is a **static, pre-execution analyzer**.

It must never: - Execute user workflows - Call LLM APIs - Observe
runtime tokens - Trace or instrument execution - Require API keys or
model access - Import and run user code - Calibrate against real runs -
Perform simulation or sampling

Noctyl is architecturally equivalent to a **compiler static analyzer**,
not a profiler.

------------------------------------------------------------------------

# 1. Objective

Phase-3 introduces **deterministic token-envelope estimation** over the
ExecutionModel (from Phase-2), deriving cost implications **purely from
source structure and declared assumptions**.

We answer: \> Given only the source code, what token growth is implied
by the workflow structure?

------------------------------------------------------------------------

# 2. Scope

## In Scope

-   Static prompt-size detection (literals/templates when resolvable)
-   Node-level token signatures (base + transformation factors)
-   Structural propagation of token quantities across paths
-   Loop amplification modeling using conservative bounds
-   Branch envelope computation (min / expected / max)
-   User-provided model profiles (assumptions only)
-   Deterministic, reproducible estimates

## Explicitly Out of Scope

-   Runtime measurement, tracing, or benchmarking
-   Any execution or simulation of the workflow
-   Learning-based prediction
-   Cross-framework support (still LangGraph-only)
-   Prompt optimization or rewriting

------------------------------------------------------------------------

# 3. Architecture Extension

    GraphExtractor (Phase-1)
            ↓
    GraphAnalyzer (Phase-2)
            ↓
    ExecutionModel
            ↓
    TokenModeler (Phase-3)
            ↓
    Cost Envelope Output

------------------------------------------------------------------------

# 4. Token Flow Model (TFM)

We reinterpret the workflow as a **Token Flow Graph**:

Each node applies a transformation:

    Input Tokens → Node Signature → Output Tokens

No semantic prediction --- only structural growth modeling.

------------------------------------------------------------------------

# 5. Node Token Signature

Each node is annotated with a static signature:

``` json
{
  "node": "planner",
  "base_prompt_tokens": 120,
  "expansion_factor": 1.35,
  "input_dependency": true
}
```

Definitions: - `base_prompt_tokens`: statically measurable
literals/templates - `expansion_factor`: heuristic structural growth -
`input_dependency`: whether output scales with input

------------------------------------------------------------------------

# 6. Prompt Size Detection (Static Only)

Detectable sources: - Literal strings - Static templates - Config
constants

If dynamic or unresolved: → mark symbolic and fall back to conservative
defaults.

------------------------------------------------------------------------

# 7. Propagation Rules

For each edge traversal:

    T_out = (T_in + base_prompt) × expansion_factor

Propagation is symbolic where necessary.

------------------------------------------------------------------------

# 8. Loop Amplification Modeling

Using Phase-2 cycle detection:

If loop bound known statically:

    LoopCost = iterations × node_cost

If unknown:

    bounded = false
    assumed_iterations = configurable_default (e.g. 3–5)

We widen the estimate rather than guessing behavior.

------------------------------------------------------------------------

# 9. Branch Envelope Computation

For conditional paths we compute envelopes:

    min_cost = cheapest path
    max_cost = most expensive path
    expected_cost = midpoint heuristic

Outputs are **ranges**, never single-point claims.

------------------------------------------------------------------------

# 10. Model Profiles (User-Declared Assumptions)

Noctyl does not discover pricing --- it consumes declared assumptions.

Example:

``` yaml
model_profiles:
  gpt-4o:
    expansion_factor: 1.4
    output_ratio: 0.6
    pricing:
      input_per_1k: 0.005
      output_per_1k: 0.015
```

Profiles are external, versionable, and optional.

------------------------------------------------------------------------

# 11. Estimation Algorithm (Deterministic)

1.  Load ExecutionModel
2.  Annotate nodes with token signatures
3.  Propagate symbolic token quantities
4.  Apply loop amplification bounds
5.  Compute branch envelopes
6.  Aggregate workflow envelope
7.  Emit deterministic estimate

Same repo + same config ⇒ identical result.

------------------------------------------------------------------------

# 12. Output Schema Extension

``` json
{
  "graph_id": "workflow_1",
  "token_estimate": {
    "assumptions_profile": "gpt-4o",
    "min_tokens": 2400,
    "expected_tokens": 5100,
    "max_tokens": 9100,
    "bounded": true,
    "confidence": "structural-static"
  }
}
```

------------------------------------------------------------------------

# 13. CLI Addition

    noctyl estimate ./repo --profile gpt-4o

Still fully offline. Still static.

------------------------------------------------------------------------

# 14. Error Handling Philosophy

When information is incomplete: - widen bounds - mark symbolic - emit
warnings

Never fabricate precision.

------------------------------------------------------------------------

# 15. Validation Strategy (Static Correctness Only)

We validate: - Determinism - Structural sensitivity (loops increase
cost) - Stability across runs - Logical monotonicity of estimates

We do **not** compare to runtime usage.

------------------------------------------------------------------------

# 16. Success Criteria

Phase-3 is complete when:

-   Token propagation model implemented
-   Loop amplification handled conservatively
-   Estimates expressed as ranges
-   Model profiles supported
-   Output reproducible and deterministic
-   No runtime dependency introduced

------------------------------------------------------------------------

# 17. Deliverables

-   TokenModeler module
-   Model profile system
-   Cost-envelope computation
-   CLI `estimate` command
-   Extended schema
-   Documentation

------------------------------------------------------------------------

# 18. What Phase-3 Enables

Noctyl can now answer:

> What scale of token cost is structurally implied before execution?

Without running a single line of user code.

------------------------------------------------------------------------

# 19. Implemented (Phase-3 Task 1)

## Task 1: Data Model and Schema 3.0 Serializer

**Status:** Implemented and tested ✓

### Code

- **`noctyl/estimation/__init__.py`** — Public exports for estimation types
- **`noctyl/estimation/data_model.py`** — Frozen dataclasses:
  - `NodeTokenSignature` — per-node token signature (base_prompt_tokens, expansion_factor, input_dependency, symbolic)
  - `ModelProfile` — user-declared model assumptions (name, expansion_factor, output_ratio, pricing)
  - `CostEnvelope` — token cost envelope (min/expected/max tokens, bounded, confidence) with invariant validation
  - `WorkflowEstimate` — full estimation result (ExecutionModel reference + signatures + envelopes + warnings)
- **`noctyl/estimation/serializer.py`** — `workflow_estimate_to_dict()`:
  - Extends Phase 2 enriched dict (via `execution_model_to_dict`)
  - Sets `schema_version: "3.0"`, `estimated: true`, `enriched: true`
  - Adds `token_estimate`, `node_signatures`, `per_node_envelopes`, `per_path_envelopes`, `warnings`
  - Deterministic sorting for all collections

### Tests

- **`tests/test_estimation_model.py`** — 31 tests covering:
  - Structure and schema version enforcement
  - Deterministic serialization (sorted collections)
  - CostEnvelope invariant validation (`min <= expected <= max`)
  - Frozen dataclass immutability
  - JSON roundtrip
  - Field serialization (all fields, symbolic nodes, unbounded envelopes)
  - Edge cases (empty collections, all symbolic nodes, large estimates)
  - Phase 2 integration (enriched fields preserved)

### Schema 3.0 Output

Extends schema 2.0 with:
- `schema_version: "3.0"`
- `estimated: true`
- `token_estimate`: workflow-level envelope (assumptions_profile, min/expected/max tokens, bounded, confidence)
- `node_signatures`: sorted list of per-node signatures
- `per_node_envelopes`: sorted dict of per-node cost envelopes
- `per_path_envelopes`: sorted dict of per-path cost envelopes (for conditional branches)
- `warnings`: sorted list of estimation warnings

All Phase 2 enriched fields (`shape`, `cycles`, `metrics`, `node_annotations`, `risks`) are preserved.

### References

- Implementation: `noctyl/estimation/`
- Tests: `tests/test_estimation_model.py`
- Flow diagrams: `docs/flow-diagrams.md` §§13–14

------------------------------------------------------------------------
