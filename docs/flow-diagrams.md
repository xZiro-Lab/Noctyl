# Noctyl flow diagrams

This document collects flow and architecture diagrams for the Noctyl pipeline. All diagrams use Mermaid and can be rendered on GitHub or in any Mermaid-capable viewer.

**Maintenance:** Keep this document in sync with the codebase. When adding or changing pipeline steps (e.g. edge extraction, entry point, compile, repo scanner), add or update the corresponding diagram and section here.

---

## 1. LangGraph ingestion pipeline (Phase-1)

End-to-end flow from Python source to extracted graph structure.

```mermaid
flowchart LR
  subgraph input [Input]
    Source[Python source]
  end
  subgraph ingestion [Ingestion]
    Detector[langgraph_detector]
    Tracker[stategraph_tracker]
    NodeExt[node_extractor]
    EdgeExt[edge_extractor]
    Source --> Detector
    Detector -->|file_contains_langgraph / has_langgraph_import| Tracker
    Tracker -->|TrackedStateGraph list| NodeExt
    Tracker -->|TrackedStateGraph list| EdgeExt
    NodeExt -->|graph_id to ExtractedNode list| OutNodes
    EdgeExt -->|graph_id to ExtractedEdge list| OutEdges
    EdgeExt -->|graph_id to ExtractedConditionalEdge list| OutCondEdges
    EdgeExt -->|extract_entry_points| EntryOut[entry_by_graph + warnings]
  end
  subgraph graph [Graph model]
    EN[ExtractedNode]
    EE[ExtractedEdge]
    CEE[ExtractedConditionalEdge]
  end
  NodeExt -.->|uses| EN
  EdgeExt -.->|uses| EE
  EdgeExt -.->|uses| CEE
  subgraph output [Output]
    OutNodes[dict graph_id to nodes]
    OutEdges[dict graph_id to edges]
    OutCondEdges[dict graph_id to conditional edges]
    EntryOut[entry_by_graph and warnings]
    OutTerminal[terminal_nodes]
  end
```

**Steps:**
0. **Repo scanning** — `discover_python_files(root_path)` yields the list of `.py` paths to analyze; default ignores `.venv`, `venv`, `site-packages`, `tests`, `.git`, `__pycache__` (see [§8](#8-repo-scanning-file-discovery)).
1. **langgraph_detector** — File-level check: does this file contain LangGraph? (`file_contains_langgraph`, `has_langgraph_import`).
2. **stategraph_tracker** — Find every `StateGraph(...)` and track variable name and `graph_id` per instance.
3. **node_extractor** — For each tracked graph, find `add_node(name, callable)` calls whose receiver resolves to that graph; emit `ExtractedNode(name, callable_ref, line)` per graph. Uses shared **receiver_resolution** (alias map + resolve_receiver).
4. **edge_extractor** — For each tracked graph: (a) `add_edge(source, target)` -> `ExtractedEdge(source, target, line)`; (b) `add_conditional_edges(source, path, path_map)` with dict-literal path_map -> one `ExtractedConditionalEdge(source, condition_label, target, line)` per path_map entry. END as target supported. (c) **Entry point:** `extract_entry_points` returns `(entry_by_graph, warnings)` from `set_entry_point(name)` or fallback from single `add_edge(START, target)`; warning when missing. Same receiver resolution.
5. **Graph schema:** Ingestion outputs (nodes, edges, conditional_edges, entry_point per graph_id) are assembled into a **WorkflowGraph** via `build_workflow_graph`; **workflow_graph_to_dict** produces a deterministic JSON-serializable dict; JSON Schema in `noctyl/graph/schema.json` describes the serialized shape. **END and terminal nodes:** END is represented as target `"END"` in edges and conditional_edges; `terminal_nodes` is a first-class list of node names that have an outgoing edge or conditional edge to END (distinguishable without scanning edges).

---

## 2. Graph schema and serialization

Ingestion outputs per graph_id are aggregated into WorkflowGraph and serialized to JSON.

```mermaid
flowchart LR
  subgraph ingestion [Ingestion]
    Nodes[nodes by graph_id]
    Edges[edges by graph_id]
    CondEdges[conditional_edges by graph_id]
    Entry[entry_by_graph]
  end
  subgraph graph [Graph]
    WG[WorkflowGraph]
    WG -->|workflow_graph_to_dict| Dict[JSON dict with nodes edges entry_point terminal_nodes]
    Dict -->|json.dumps sort_keys| JSON[JSON string]
  end
  Nodes --> WG
  Edges --> WG
  CondEdges --> WG
  Entry --> WG
```

**Data flow:**
- **build_workflow_graph(graph_id, nodes, edges, conditional_edges, entry_point)** builds a WorkflowGraph (schema_version, graph_id, nodes, edges, conditional_edges, entry_point, terminal_nodes). `terminal_nodes` is derived from edges and conditional_edges (sources where target is `"END"`).
- **workflow_graph_to_dict(g)** returns a dict with deterministic list ordering (nodes by name/line, edges by source/target/line, conditional_edges by source/condition_label/target/line). The serialized dict includes **terminal_nodes** (sorted list of node names that transition to END).
- **JSON Schema:** `noctyl/graph/schema.json` defines the serialized document shape (nodes, directed edges, entry_point, terminal_nodes).

---

## 3. Node extraction flow

How add_node calls are attributed to tracked StateGraph instances.

```mermaid
flowchart LR
  subgraph ingestion [Ingestion]
    ST[stategraph_tracker]
    NE[node_extractor]
    ST -->|TrackedStateGraph list| NE
    NE -->|graph_id to ExtractedNode list| OUT[dict]
  end
  subgraph graph [Graph]
    EN[ExtractedNode]
  end
  NE -.->|uses| EN
```

**Data flow:**
- **Input:** `(source, file_path, list[TrackedStateGraph])`.
- **Same-file alias resolution:** Build `name -> root` so that `h = g` and `g` = StateGraph variable implies `h.add_node(...)` is attributed to `g`’s `graph_id`.
- **Output:** `dict[graph_id, list[ExtractedNode]]` with `ExtractedNode(name, callable_ref, line)`.

---

## 4. Edge extraction flow

How add_edge calls are attributed to tracked StateGraph instances.

```mermaid
flowchart LR
  subgraph ingestion [Ingestion]
    ST[stategraph_tracker]
    EE[edge_extractor]
    ST -->|TrackedStateGraph list| EE
    EE -->|graph_id to ExtractedEdge list| OutEdges[dict]
  end
  subgraph graph [Graph]
    EEdge[ExtractedEdge]
  end
  EE -.->|uses| EEdge
```

**Data flow:**
- **Input:** `(source, file_path, list[TrackedStateGraph])`.
- **Same-file alias resolution:** Same as node extraction (shared `receiver_resolution.build_alias_map`, `resolve_receiver`).
- **Source/target:** Literal string -> value; Name (e.g. START, END) -> id; other -> unparse/repr. Missing nodes do not prevent extraction.
- **Output:** `dict[graph_id, list[ExtractedEdge]]` with `ExtractedEdge(source, target, line)`.

---

## 5. Conditional edges flow

How add_conditional_edges(path_map) is attributed to tracked StateGraph instances.

```mermaid
flowchart LR
  subgraph ingestion [Ingestion]
    ST[stategraph_tracker]
    EE[edge_extractor]
    ST -->|TrackedStateGraph list| EE
    EE -->|graph_id to ExtractedConditionalEdge list| OutCond[dict]
  end
  subgraph graph [Graph]
    CondEdge[ExtractedConditionalEdge]
  end
  EE -.->|uses| CondEdge
```

**Data flow:**
- **Input:** Same as sequential edges: `(source, file_path, list[TrackedStateGraph])`.
- **path_map:** Only dict literals are supported; variable path_map is skipped. One `ExtractedConditionalEdge` per (key, value) in path_map: `condition_label` from key, `target` from value (END -> `"END"`).
- **Output:** `dict[graph_id, list[ExtractedConditionalEdge]]` with `ExtractedConditionalEdge(source, condition_label, target, line)`.

---

## 6. Entry point

How the workflow entry node is detected per graph.

```mermaid
flowchart LR
  subgraph ingestion [Ingestion]
    ST[stategraph_tracker]
    EE[edge_extractor]
    ST -->|TrackedStateGraph list| EE
    EE -->|extract_entry_points| EntryOut[entry_by_graph and warnings]
  end
  EE -->|set_entry_point| EntryOut
  EE -->|infer from add_edge START| EntryOut
```

**Data flow:**
- **Explicit:** `set_entry_point(name)` on receiver that resolves to tracked graph -> `entry_by_graph[graph_id] = name`.
- **Fallback:** If no set_entry_point, infer from single `add_edge(START, target)` for that graph; else None.
- **Warnings:** When entry is None (missing or ambiguous), append a message to `warnings` list.
- **Output:** `(dict[graph_id, str | None], list[str])`.

---

## 7. Detection and tracking (file-level)

How we decide a file has LangGraph and how we get graph instances.

```mermaid
flowchart TB
  Source[Python source] --> Parse[ast.parse]
  Parse --> ImportCheck[Import check]
  ImportCheck -->|has_langgraph_import| Track[track_stategraph_instances]
  Track -->|list TrackedStateGraph| Instances[graph_id per instance]
  ImportCheck -->|no langgraph.graph import| Skip[Skip file]
  Parse -->|SyntaxError| Empty[return empty / False]
```

- **Fast path:** `has_langgraph_import(source)` — one AST pass over imports; use to skip files with no `langgraph.graph` import.
- **Full check:** `file_contains_langgraph(source)` uses `track_stategraph_instances`; True iff at least one `StateGraph(...)` is found.

---

## 8. Repo scanning (file discovery)

Before ingestion, the set of Python files to analyze is produced by scanning the repository root with default ignore rules. Only project code is considered; virtual environments, site-packages, and tests are excluded by default.

```mermaid
flowchart LR
  Root[root_path] --> Scanner[discover_python_files]
  Scanner -->|default ignore_dirs| Filter[exclude .venv venv site-packages tests .git __pycache__]
  Filter --> List[list of .py Paths sorted]
  List -->|per file| Ingestion[Ingestion pipeline]
```

**Default ignore list** (any path containing one of these as a segment is skipped):

| Directory       | Reason |
|----------------|--------|
| `.venv`        | Virtual environment (common layout). |
| `venv`         | Alternate virtual environment name. |
| `site-packages`| Installed packages; Phase-1 is project code only. |
| `tests`        | Test code is out of scope for workflow extraction in Phase-1. |
| `.git`         | Version control metadata. |
| `__pycache__`  | Bytecode cache. |

**API:**
- **`discover_python_files(root_path, ignore_dirs=None)`** — Returns a **sorted** list of `Path` objects for every `.py` file under `root_path` that does not contain any ignored directory name. `root_path` can be `Path` or `str`; `ignore_dirs` is an optional sequence (if `None`, the default list above is used). Deterministic: same repo and same tool version produce the same list order.
- **`DEFAULT_IGNORE_DIRS`** — The default tuple of ignored directory names; exportable for reference or custom logic.

**Data flow:** Input is `root_path` (directory) and optional `ignore_dirs`. Output is the sorted list of paths. Each path is typically read and passed through the ingestion pipeline (detector, tracker, node/edge extraction).

---

## 9. Error handling (Phase-1)

**Strategy:** Best-effort. Invalid or unsupported code: warn and continue; skip and report; emit partial when possible. No fail-fast (see [phase1-scope.md](phase1-scope.md) §8).

```mermaid
flowchart TB
  Root[root_path] --> Runner[run_pipeline_on_directory]
  Runner --> Discover[discover_python_files]
  Discover --> Paths[list of .py paths]
  Paths --> Loop[for each path]
  Loop --> Read[read_text utf-8]
  Read -->|OSError or UnicodeDecodeError| WarnSkip["append warning: path could not read; skip"]
  WarnSkip --> Loop
  Read -->|success| HasLangGraph[file_contains_langgraph]
  HasLangGraph -->|False| Loop
  HasLangGraph -->|True| Ingest[track + extract nodes edges entry]
  Ingest --> Build[build_workflow_graph + workflow_graph_to_dict]
  Build --> Append[append to results]
  Append --> Loop
  Loop --> Done[return results and warnings]
```

**Where the library does not crash:** All public ingestion APIs that take `source: str` catch `SyntaxError` and return safe empty/false values:
- `has_langgraph_import` / `file_contains_langgraph` → False
- `track_stategraph_instances` → []
- `extract_add_node_calls` → {}
- `extract_add_edge_calls` / `extract_add_conditional_edges` → {}
- `extract_entry_points` → ({}, [])

**Where warnings come from:**
- **`extract_entry_points`** returns `(entry_by_graph, warnings)` with messages:
  - `"graph_id {gid}: no entry point detected"`
  - `"graph_id {gid}: ambiguous entry (multiple add_edge(START, ...))"`
- **`run_pipeline_on_directory`** (when reading files) appends:
  - `"{path}: could not read"` on OSError or UnicodeDecodeError for that file.

**Tool does not crash:** `run_pipeline_on_directory(root_path)` runs discover → read each file → ingest → build WorkflowGraph → to_dict; it catches file-read errors and skips with a warning, so it never raises for invalid or unreadable files.

---

## 10. Generate graph of agents (Mermaid)

From an extracted workflow dict you can produce a **Mermaid flowchart** (graph of agents and edges) for visualization or export.

**How to generate the graph:**
1. Get a workflow dict: run the pipeline (e.g. `run_pipeline_on_directory(root)`), then use each result dict; or build a `WorkflowGraph` and call `workflow_graph_to_dict(g)`.
2. Call `workflow_dict_to_mermaid(d)` from `noctyl.graph`. It returns a Mermaid string (flowchart TB) with:
   - **Nodes:** START and END (as distinct nodes), plus each workflow node (agent/step) by name.
   - **Edges:** Sequential edges (`source --> target`) and conditional edges (`source -->|condition_label| target`).
3. Render the string in any Mermaid-capable viewer (e.g. GitHub, Mermaid Live Editor) or write to a `.mmd` file.

**Example (Python):**
```python
from noctyl.graph import workflow_graph_to_dict, workflow_dict_to_mermaid
from noctyl.ingestion import run_pipeline_on_directory

results, _ = run_pipeline_on_directory("path/to/repo")
for d in results:
    mermaid = workflow_dict_to_mermaid(d)
    print(mermaid)  # or open("graph.mmd", "w").write(mermaid)
```

Entry and terminal nodes appear via edges: START → entry_point, and terminal nodes → END. Conditional edge labels are shown on the arrows.

---

*Add new flow diagrams to this document as the pipeline grows (entry/exit, compile, etc.).*
