# Deep-Scout (formerly Noctyl)

<p align="center">
  <img src="https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif" width="120" alt="owl" />
</p>

```
  _   ___     (o o)   _   _
 | \ | _ \   (  V  ) | | | |
 |  \| __/    |~~~|  | |_| |
 |_| |_|      +---+   \__/
 N   o   c     t     y   l
    
```

**Static Token Usage Estimator for Multi-Agent AI Workflows**

Deep-Scout is a static analysis tool that **estimates token usage and cost for multi-agent AI workflows *before execution***.  
It analyzes a code repository, constructs a workflow graph, and produces structured reports that can be consumed by humans *and* AI assistants (Claude, Codex, Copilot, Cursor).

> Deep-Scout does **not** execute agents, call LLM APIs, or burn tokens.  
> It provides **pre-run intelligence** for cost, safety, and efficiency.

---

## Why Deep-Scout?

Agentic systems often fail silently due to:

- Unbounded loops
- Prompt and memory explosion
- Hidden retry costs
- Poor agent decomposition

Deep-Scout answers:

- *How many tokens will this workflow burn before I run it?*
- *Where does token growth originate?*
- *Which agents are cost hotspots?*
- *What can be optimized pre-deployment?*

---

## Core Capabilities

- 🌐 **Workflow Graph Extraction**
  - Builds a directed semantic graph from agentic codebases
  - Captures agents, tools, loops, retries, and memory interactions

- 📊 **Enriched workflow graph (Phase 2)**
  - **GraphAnalyzer** and **ExecutionModel** for control-flow, cycles, metrics, node annotations, and structural risks
  - Optional `enriched=True` pipeline output (schema 2.0); see `docs/phase/phase2.md` and `docs/flow-diagrams.md`

- 📐 **Static Token Estimation (Phase 3)**
  - Token envelope estimation (min/expected/max ranges)
  - Node-level token signatures with prompt size detection
  - Model profiles for user-declared assumptions
  - Cost envelope computation for workflows, nodes, and paths
  - Optional `estimate=True` pipeline output (schema 3.0); see `docs/phase/phase3.md` and `docs/flow-diagrams.md`

- ⚠️ **Risk Detection**
  - Unbounded loops
  - Recursive agent calls
  - Memory writes inside loops
  - Tool output amplification

- 🤖 **AI-Assistant Integration**
  - Generates structured context for Claude, Codex, Copilot, Cursor
  - No API keys required
  - File-based, assistant-agnostic design

---

## What Deep-Scout Is NOT

- ❌ Not a runtime token monitor
- ❌ Not a tracing or observability tool
- ❌ Not an LLM wrapper
- ❌ Not tied to any single agent framework

Deep-Scout runs **before execution**, not during or after.

---

## Installation

### One-line installer (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/xZiro-Lab/Noctyl/main/install.sh | bash
```

During the transition, the installer URL remains on the current repository path and installs the new `deep-scout` package first with a compatibility fallback.

### CLI Usage

After installation, use the `deep-scout estimate` command to estimate token usage:

```bash
# Basic usage (default profile)
deep-scout estimate ./my_project

# With custom model profile
deep-scout estimate ./my_project --profile profiles/gpt-4o.yaml

# Save output to file
deep-scout estimate ./my_project --output estimates.json

# With profile and output file
deep-scout estimate ./my_project --profile profiles/gpt-4o.yaml --output estimates.json
```

`noctyl estimate ...` remains available for one compatibility release and prints a deprecation warning.

## Migration (Noctyl -> Deep-Scout)

- Primary command is now `deep-scout`.
- Legacy command `noctyl` is still supported for this release.
- Python import namespace remains `noctyl` in this phase to avoid breaking code.

### Deprecation timeline

- Current release: both `deep-scout` and `noctyl` CLI commands work.
- Next release: `noctyl` CLI alias is removed; `deep-scout` remains the only supported CLI command.

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
```

**Output:** The CLI outputs JSON with schema 3.0 format, including token estimates, node signatures, per-node and per-path envelopes, and warnings. See `docs/phase/phase3.md` for details.

---

## Project structure

```
deep-scout/
├── README.md
├── LICENSE
├── pyproject.toml
├── install.sh
├── .gitignore
│
├── noctyl/                         # Core package (kept as-is for compatibility)
│   ├── __init__.py
│   │
│   ├── ingestion/                  # Repo scanning, detection & extraction (Phase 1)
│   │   ├── __init__.py
│   │   ├── pipeline.py             # run_pipeline_on_directory (Phase 1 + Phase 2 + Phase 3)
│   │   ├── repo_scanner.py         # discover_python_files + default ignores
│   │   ├── langgraph_detector.py   # has_langgraph_import / file_contains_langgraph
│   │   ├── stategraph_tracker.py   # track StateGraph instances per file
│   │   ├── node_extractor.py       # extract add_node calls per graph
│   │   ├── edge_extractor.py       # extract add_edge / add_conditional_edges / entry points
│   │   └── receiver_resolution.py  # alias map + resolve receiver to tracked graph
│   │
│   ├── graph/                      # Data model, serialization & visualization
│   │   ├── __init__.py
│   │   ├── graph.py                # WorkflowGraph, build_workflow_graph, workflow_graph_to_dict
│   │   ├── nodes.py                # ExtractedNode dataclass
│   │   ├── edges.py                # ExtractedEdge, ExtractedConditionalEdge dataclasses
│   │   ├── execution_model.py      # ExecutionModel, DetectedCycle, StructuralMetrics, etc. (Phase 2)
│   │   └── mermaid.py              # workflow_dict_to_mermaid (Mermaid flowchart generation)
│   │
│   └── analysis/                   # Static graph analysis (Phase 2)
│       ├── __init__.py
│       ├── analyzer.py             # GraphAnalyzer.analyze → ExecutionModel
│       ├── digraph.py              # DirectedGraph from WorkflowGraph
│       ├── control_flow.py         # Tarjan SCC, cycle detection, graph shape
│       ├── metrics.py              # Structural metrics (counts, paths, branching)
│       ├── node_annotation.py      # Per-node semantic annotation from AST
│       └── structural_risk.py      # Risk detection (unreachable, dead-ends, non-terminating)
│   │
│   ├── estimation/                  # Token estimation (Phase 3)
│       ├── __init__.py
│       ├── data_model.py           # NodeTokenSignature, ModelProfile, CostEnvelope, WorkflowEstimate
│       ├── serializer.py           # workflow_estimate_to_dict (schema 3.0)
│       ├── prompt_detection.py    # AST-based prompt string detection
│       ├── propagation.py          # Token propagation with topological traversal
│       ├── loop_amplification.py  # Loop amplification using DetectedCycle data
│       ├── branch_envelope.py     # Branch envelope computation for conditional paths
│       ├── aggregation.py         # Workflow-level envelope aggregation
│       ├── token_modeler.py       # TokenModeler class orchestrating the pipeline
│       └── profile_loader.py      # YAML profile loading and defaults
│   │
│   └── cli.py                       # CLI command interface (deep-scout estimate, legacy noctyl alias)
│
├── tests/                          # 493+ tests (pytest)
│   ├── fixtures/golden/            # 8 canonical LangGraph fixture files
│   ├── test_analysis.py            # Phase 2 analysis module tests
│   ├── test_execution_model.py     # ExecutionModel serialization & immutability tests
│   ├── test_estimation_model.py    # Phase 3 estimation data model & serializer tests
│   ├── test_golden.py              # Golden fixture integration tests
│   ├── test_golden_mermaid.py      # Mermaid generation for golden fixtures
│   ├── test_ingestion_integration.py  # Full pipeline integration tests
│   ├── test_receiver_resolution.py # Alias map & receiver resolution tests
│   ├── test_graph_schema.py        # WorkflowGraph schema & serialization tests
│   ├── test_mermaid.py             # Mermaid diagram generation tests
│   ├── test_langgraph_detector.py  # LangGraph detection tests
│   ├── test_stategraph_tracker.py  # StateGraph tracking tests
│   ├── test_node_extractor.py      # Node extraction tests
│   ├── test_edge_extractor.py      # Edge extraction tests
│   ├── test_conditional_edges.py   # Conditional edge extraction tests
│   ├── test_entry_point.py         # Entry point detection tests
│   ├── test_repo_scanner.py        # File discovery tests
│   ├── test_example_multi_agent.py # Multi-agent example tests
│   ├── test_prompt_detection.py    # Phase 3 prompt detection & token signature tests
│   ├── test_propagation.py         # Phase 3 token propagation tests
│   ├── test_loop_amplification.py  # Phase 3 loop amplification tests
│   ├── test_branch_envelope.py    # Phase 3 branch envelope tests
│   ├── test_aggregation.py         # Phase 3 workflow aggregation tests
│   ├── test_token_modeler.py       # Phase 3 TokenModeler integration tests
│   ├── test_profile_loader.py       # Phase 3 profile loader tests
│   ├── test_pipeline_integration.py  # Phase 3 pipeline integration tests
│   ├── test_cli.py                  # CLI tests
│   └── test_phase3_comprehensive.py # Phase 3 comprehensive integration tests
│
├── docs/
│   ├── flow-diagrams.md            # Pipeline & architecture Mermaid diagrams
│   └── phase/
│       ├── phase1-scope.md         # Phase 1 scope & design
│       ├── phase2.md               # Phase 2 design & implementation status
│       └── phase3.md               # Phase 3 design & implementation status
│
└── .github/
    └── ISSUE_TEMPLATE/             # Phase task issue templates
```

---

## Status

**Phase 1** (LangGraph ingestion pipeline) — Implemented and tested.
**Phase 2** (Static graph analysis: control-flow, metrics, annotations, risks) — Implemented and tested.
**Phase 3** (Static token estimation) — Implemented and tested.
  - **Task 1** (Data model and schema 3.0 serializer) — Implemented and tested ✓
  - **Task 2** (Prompt size detection) — Implemented and tested ✓
  - **Task 3** (TokenModeler: propagation, loops, branches, aggregation) — Implemented and tested ✓
  - **Task 4** (Pipeline integration & CLI) — Implemented and tested ✓
  - **Task 5** (Comprehensive testing) — Implemented and tested ✓
  - **Task 6** (Documentation) — Implemented ✓

493+ tests across 27 test files, all passing. APIs and behavior may evolve as new phases are added.

---

*Deep-Scout (formerly Noctyl) — know your token usage before you run.*
