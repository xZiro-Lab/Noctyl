# Noctyl

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

Noctyl is a static analysis tool that **estimates token usage and cost for multi-agent AI workflows *before execution***.  
It analyzes a code repository, constructs a workflow graph, and produces structured reports that can be consumed by humans *and* AI assistants (Claude, Codex, Copilot, Cursor).

> Noctyl does **not** execute agents, call LLM APIs, or burn tokens.  
> It provides **pre-run intelligence** for cost, safety, and efficiency.

---

## Why Noctyl?

Agentic systems often fail silently due to:

- Unbounded loops
- Prompt and memory explosion
- Hidden retry costs
- Poor agent decomposition

Noctyl answers:

- *How many tokens will this workflow burn before I run it?*
- *Where does token growth originate?*
- *Which agents are cost hotspots?*
- *What can be optimized pre-deployment?*

---

## Core Capabilities

- 🌐 **Workflow Graph Extraction**
  - Builds a directed semantic graph from agentic codebases
  - Captures agents, tools, loops, retries, and memory interactions

- 📐 **Static Token Estimation**
  - Prompt size analysis
  - Memory replay modeling
  - Loop and retry expansion

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

## What Noctyl Is NOT

- ❌ Not a runtime token monitor
- ❌ Not a tracing or observability tool
- ❌ Not an LLM wrapper
- ❌ Not tied to any single agent framework

Noctyl runs **before execution**, not during or after.

---

## Installation

### One-line installer (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/xZiro-Lab/Noctyl/main/install.sh | bash
```

---

## Project structure

```
noctyl/
├── README.md
├── LICENSE
├── pyproject.toml
├── install.sh
├── .gitignore
│
├── noctyl/                     # Core package
│   ├── __init__.py
│   │
│   ├── cli/                    # CLI entrypoints
│   │   ├── __init__.py
│   │   ├── main.py             # `noctyl` command
│   │   └── commands.py         # analyze, graph, report
│   │
│   ├── ingestion/              # Repo scanning & parsing
│   │   ├── __init__.py
│   │   ├── repo_scanner.py     # Walk filesystem
│   │   ├── ast_parser.py       # Python AST parsing
│   │   └── framework_adapters/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── langchain.py
│   │       ├── crewai.py
│   │       └── autogen.py
│   │
│   ├── graph/                  # Workflow graph construction
│   │   ├── __init__.py
│   │   ├── graph.py            # Graph data model
│   │   ├── nodes.py            # Node types
│   │   ├── edges.py            # Edge semantics
│   │   └── builder.py          # Build graph from parsed code
│   │
│   ├── prompts/                # Prompt & memory analysis
│   │   ├── __init__.py
│   │   ├── extractor.py        # Prompt extraction
│   │   ├── memory_model.py     # Memory growth modeling
│   │   └── templates/
│   │       ├── optimize.md.j2
│   │       ├── cost.md.j2
│   │       └── safety.md.j2
│   │
│   ├── tokenization/           # Token estimation logic
│   │   ├── __init__.py
│   │   ├── estimator.py
│   │   ├── tokenizer.py
│   │   ├── pricing.py
│   │   └── models.yaml
│   │
│   ├── analysis/               # Risk & heuristic analysis
│   │   ├── __init__.py
│   │   ├── risk.py
│   │   ├── heuristics.py
│   │   └── validators.py
│   │
│   ├── report/                 # Output generation
│   │   ├── __init__.py
│   │   ├── json_report.py
│   │   ├── markdown_report.py
│   │   └── html_report.py
│   │
│   ├── ai_context/             # AI-assistant integration
│   │   ├── __init__.py
│   │   ├── composer.py         # Build AI-readable context
│   │   └── schema.py           # Contract for AI tools
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── config.py
│       └── filesystem.py
│
├── examples/                   # Golden repos for testing
│   ├── linear_agent/
│   ├── agent_with_loop/
│   └── agent_with_memory/
│
├── tests/
│   ├── test_graph_builder.py
│   ├── test_loop_detection.py
│   ├── test_token_estimation.py
│   └── test_risk_analysis.py
│
├── docs/                       # Extended documentation
│   ├── architecture.md
│   ├── graph_model.md
│   ├── ai_integration.md
│   └── research_notes.md
│
└── .github/
    ├── workflows/
    │   └── release.yml         # Binary / package releases
    └── ISSUE_TEMPLATE.md
```

---

## Status

Experimental — APIs and behavior may change.

---

*Noctyl — know your token usage before you run.*
