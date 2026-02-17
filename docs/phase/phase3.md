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

# 20. Implemented (Phase-3 Task 2)

## Task 2: Prompt Size Detection

**Status:** Implemented and tested ✓

### Code

- **`noctyl/estimation/prompt_detection.py`** — AST-based static prompt detection:
  - `PromptFragment` — dataclass for detected string fragments (text, token_estimate, symbolic)
  - `estimate_tokens_from_string()` — heuristic token estimation (`len(text) // 4`, min 1)
  - `detect_prompt_strings()` — extracts strings from function AST bodies:
    - Literal strings (`ast.Constant`)
    - F-strings (`ast.JoinedStr`) with static/dynamic part detection
    - String concatenation (`ast.BinOp` with `+`)
    - `.format()` calls with literal argument detection
    - Marks dynamic parts as symbolic
  - `_has_input_dependency()` — heuristic to detect state/input dependencies
  - `compute_node_token_signatures()` — orchestrates callable resolution and signature computation:
    - Resolves callables (local functions, imported functions, class methods, lambdas)
    - Detects prompt strings per node
    - Sums token estimates from fragments
    - Applies ModelProfile defaults
    - Returns sorted tuple of `NodeTokenSignature`
- **`noctyl/estimation/__init__.py`** — Updated exports for prompt detection types and functions

### Tests

- **`tests/test_prompt_detection.py`** — 47 tests covering:
  - Token estimation heuristics (basic, empty, short, long strings)
  - String detection (literals, f-strings, concatenation, `.format()` calls)
  - Callable resolution (local, imported, lambda, class methods)
  - Input dependency detection
  - Symbolic marking for dynamic content
  - Integration with WorkflowGraph and ModelProfile
  - Deterministic output
  - Edge cases (nested functions, comments, multiline strings, syntax errors)
  - Empty/missing source handling
  - Unresolvable callables

### Features

- **AST-based analysis:** Static analysis without execution
- **Multiple string patterns:** Handles literals, f-strings, concatenation, `.format()` calls
- **Symbolic detection:** Marks dynamic parts as symbolic for conservative estimation
- **Callable resolution:** Reuses patterns from `noctyl/analysis/node_annotation.py`
- **Graceful degradation:** Returns symbolic defaults for unresolvable callables or syntax errors
- **Deterministic output:** Sorted signatures for reproducible results

### References

- Implementation: `noctyl/estimation/prompt_detection.py`
- Tests: `tests/test_prompt_detection.py`
- Related: `noctyl/analysis/node_annotation.py` (callable resolution patterns)

------------------------------------------------------------------------

# 21. Implemented (Phase-3 Task 3)

## Task 3: TokenModeler

**Status:** Implemented and tested ✓

### Code

- **`noctyl/estimation/propagation.py`** — Token propagation with topological traversal:
  - `propagate_tokens()` — applies `T_out = (T_in + base_prompt) × expansion_factor`
  - Topological sort (Kahn's algorithm) for acyclic traversal
  - Entry point detection (nodes with START as predecessor)
  - Convergence handling (multiple incoming edges sum envelopes)
  - Cycle node marking (bounded=False initially)
- **`noctyl/estimation/loop_amplification.py`** — Loop amplification:
  - `apply_loop_amplification()` — multiplies cycle node envelopes by iteration count
  - Uses `DetectedCycle` data from Phase 2
  - Default `assumed_iterations = 5` for unbounded cycles
  - Non-terminating cycles handled with warnings
- **`noctyl/estimation/branch_envelope.py`** — Branch envelope computation:
  - `compute_branch_envelopes()` — computes min/expected/max across conditional paths
  - Path traversal from branch point to terminals (BFS)
  - Path key format: `"{source}:{condition_label}"`
  - Bounded flag aggregation across paths
- **`noctyl/estimation/aggregation.py`** — Workflow aggregation:
  - `aggregate_workflow_envelope()` — aggregates from entry to terminals
  - Path finding from entry_point to each terminal
  - Invariant enforcement (expected_tokens between min and max)
  - Confidence always "structural-static"
- **`noctyl/estimation/token_modeler.py`** — Main TokenModeler class:
  - `TokenModeler.estimate()` — orchestrates full pipeline:
    1. Build DirectedGraph
    2. Compute node token signatures (prompt detection)
    3. Propagate tokens
    4. Apply loop amplification
    5. Compute branch envelopes
    6. Aggregate workflow envelope
    7. Collect warnings
    8. Build WorkflowEstimate
  - Warning collection: symbolic nodes, unbounded loops, non-terminating cycles
  - Deterministic output (sorted dicts and tuples)

### Tests

- **`tests/test_propagation.py`** — 9 unit tests covering:
  - Linear workflow propagation
  - Branching workflow propagation
  - Cyclic workflow propagation
  - Convergence (multiple incoming edges)
  - Symbolic nodes
  - Entry point handling
  - Custom expansion factors
  - Empty graph edge case
  - Deterministic output
- **`tests/test_loop_amplification.py`** — 8 unit tests covering:
  - Self-loop amplification
  - Multi-node cycle amplification
  - Non-terminating cycles
  - Multiple independent cycles
  - Empty cycles case
  - Custom iterations parameter
  - Non-cycle nodes preservation
  - Deterministic output
- **`tests/test_branch_envelope.py`** — 7 unit tests covering:
  - Single branch point with 2 paths
  - Multiple branch points
  - Paths with different costs
  - Paths terminating at END
  - No conditional edges case
  - Deterministic path keys
  - Bounded flag aggregation
- **`tests/test_aggregation.py`** — 8 unit tests covering:
  - Single terminal node
  - Multiple terminal nodes
  - No entry point edge case
  - No terminals edge case
  - Multiple paths to same terminal
  - Bounded flag aggregation
  - Confidence field
  - Deterministic output
- **`tests/test_token_modeler.py`** — 12 integration tests covering:
  - Linear workflow end-to-end
  - Cyclic workflow end-to-end
  - Branching workflow end-to-end
  - Complex workflows (cycles + branches)
  - Source code provided vs not provided
  - Warnings collection
  - Deterministic output
  - Empty graph edge case
  - ModelProfile integration
  - WorkflowEstimate structure validation
  - Dict sorting
  - Phase 2 integration

**Total:** 44 new tests, all passing.

### Features

- **Topological token propagation:** Formula-based propagation through workflow graph
- **Loop amplification:** Cycle cost multiplication with assumed iterations
- **Branch envelope computation:** Min/expected/max across conditional paths
- **Workflow-level aggregation:** Entry-to-terminal cost summation
- **Deterministic output:** Sorted dicts and tuples for reproducibility
- **Warning collection:** Symbolic nodes, unbounded loops, non-terminating cycles
- **Invariant enforcement:** CostEnvelope min <= expected <= max always maintained

### References

- Implementation: `noctyl/estimation/propagation.py`, `loop_amplification.py`, `branch_envelope.py`, `aggregation.py`, `token_modeler.py`
- Tests: `tests/test_propagation.py`, `tests/test_loop_amplification.py`, `tests/test_branch_envelope.py`, `tests/test_aggregation.py`, `tests/test_token_modeler.py`
- Related: `noctyl/analysis/digraph.py` (DirectedGraph), `noctyl/graph/execution_model.py` (DetectedCycle)

------------------------------------------------------------------------

## 22. Implemented (Phase-3 Task 4): Pipeline Integration & CLI

**Status:** Implemented and tested ✓

### Code

**New modules:**
- **`noctyl/estimation/profile_loader.py`** — YAML profile loading and defaults
  - `default_model_profile()` — Returns default ModelProfile
  - `load_model_profile()` — Loads from ModelProfile, YAML file, dict, or None
  - Supports single-profile and multi-profile YAML formats
  - Handles missing fields with defaults
  - Graceful error handling for invalid files
  
- **`noctyl/cli.py`** — CLI command interface
  - `main()` — Main CLI entry point
  - `noctyl estimate` command with `--profile` and `--output` flags
  - JSON output to stdout or file
  - Warnings printed to stderr
  - Exit codes: 0 on success, 1 on errors/warnings

**Modified modules:**
- **`noctyl/ingestion/pipeline.py`** — Updated with estimate and profile parameters
  - Added `estimate: bool = False` parameter
  - Added `profile: ModelProfile | str | Path | dict | None = None` parameter
  - `estimate=True` automatically sets `enriched=True` (Phase 3 requires Phase 2)
  - Integrates TokenModeler for Phase 3 estimation
  - Merges TokenModeler warnings with pipeline warnings
  - Backward compatible: existing callers unaffected

- **`noctyl/estimation/__init__.py`** — Exports profile loader functions
  - Added `default_model_profile` and `load_model_profile` to exports

- **`pyproject.toml`** — Added dependencies and console scripts
  - Added `pyyaml>=6` dependency
  - Added `[project.scripts]` section with `noctyl = "noctyl.cli:main"`

### Tests

- **`tests/test_profile_loader.py`** — 18 unit tests covering:
  - Default profile creation
  - ModelProfile passthrough
  - Dict construction (with/without pricing, nested pricing)
  - YAML file loading (single/multi-profile, missing fields, invalid syntax)
  - Error handling (invalid file, empty file, missing fields, invalid types)
  - Deterministic output
  
- **`tests/test_pipeline_integration.py`** — 22 integration tests covering:
  - Backward compatibility (estimate=False, enriched=True)
  - Estimate mode (schema 3.0 output)
  - Profile loading (default, custom ModelProfile, YAML file, dict)
  - Schema 3.0 structure validation
  - Warnings merging
  - Error handling (empty directory, no LangGraph files, profile load errors)
  - Multiple graphs in same file
  - Deterministic output
  - Golden fixtures integration
  - End-to-end profile YAML loading
  - Backward compatibility verification
  
- **`tests/test_cli.py`** — 15 CLI tests covering:
  - Basic estimate command
  - Profile flag (`--profile`)
  - Output flag (`--output`)
  - Missing/invalid path handling
  - Invalid profile file handling
  - JSON output validation
  - Warnings to stderr
  - Help flag (`--help`)
  - Exit codes (0 on success, 1 on warnings/errors)
  - Empty results handling
  - End-to-end CLI on golden fixtures

**Total:** 55 new tests, all passing.

### CLI Usage

**Command syntax:**
```bash
noctyl estimate <path> [--profile <file>] [--output <file>]
```

**Examples:**
```bash
# Basic usage (default profile)
noctyl estimate ./my_project

# With custom profile
noctyl estimate ./my_project --profile profiles/gpt-4o.yaml

# Save to file
noctyl estimate ./my_project --output estimates.json

# With profile and output file
noctyl estimate ./my_project --profile profiles/gpt-4o.yaml --output estimates.json
```

**Profile File Format (YAML):**
```yaml
# Single profile format
name: gpt-4o
expansion_factor: 1.2
output_ratio: 0.6
pricing:
  input_per_1k: 0.005
  output_per_1k: 0.015

# Multi-profile format (first profile used)
model_profiles:
  gpt-4o:
    expansion_factor: 1.2
    output_ratio: 0.6
    pricing:
      input_per_1k: 0.005
      output_per_1k: 0.015
  claude-3:
    expansion_factor: 1.1
    output_ratio: 0.5
    pricing:
      input_per_1k: 0.003
      output_per_1k: 0.015
```

### Features

- **YAML profile loading:** Supports single-profile and multi-profile YAML files
- **Default profile fallback:** Uses default profile when none provided or on load errors
- **Schema 3.0 output from pipeline:** `run_pipeline_on_directory(path, estimate=True)` produces schema 3.0 dicts
- **CLI estimate command:** Command-line interface for token estimation
- **Backward compatibility maintained:** Existing callers with `enriched=True` unaffected
- **Error handling:** Profile loading errors don't crash pipeline (uses default with warning)
- **Deterministic output:** JSON output with sorted keys for reproducibility

### References

- Implementation: `noctyl/estimation/profile_loader.py`, `noctyl/cli.py`, `noctyl/ingestion/pipeline.py`
- Tests: `tests/test_profile_loader.py`, `tests/test_pipeline_integration.py`, `tests/test_cli.py`
- Related: `noctyl/estimation/token_modeler.py` (TokenModeler), `noctyl/estimation/serializer.py` (workflow_estimate_to_dict)
- Documentation: `docs/flow-diagrams.md` (Section 14a: CLI Estimate Command Flow)

------------------------------------------------------------------------
