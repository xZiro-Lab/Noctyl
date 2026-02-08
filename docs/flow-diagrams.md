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
  end
  subgraph graph [Graph model]
    EN[ExtractedNode]
    EE[ExtractedEdge]
  end
  NodeExt -.->|uses| EN
  EdgeExt -.->|uses| EE
  subgraph output [Output]
    OutNodes[dict graph_id to nodes]
    OutEdges[dict graph_id to edges]
  end
```

**Steps:**
1. **langgraph_detector** — File-level check: does this file contain LangGraph? (`file_contains_langgraph`, `has_langgraph_import`).
2. **stategraph_tracker** — Find every `StateGraph(...)` and track variable name and `graph_id` per instance.
3. **node_extractor** — For each tracked graph, find `add_node(name, callable)` calls whose receiver resolves to that graph; emit `ExtractedNode(name, callable_ref, line)` per graph. Uses shared **receiver_resolution** (alias map + resolve_receiver).
4. **edge_extractor** — For each tracked graph, find `add_edge(source, target)` calls; emit `ExtractedEdge(source, target, line)` per graph. Same receiver resolution; source/target stringified (literal, START/END, or best-effort).

---

## 2. Node extraction flow

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

## 3. Edge extraction flow

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

## 4. Detection and tracking (file-level)

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

*Add new flow diagrams to this document as the pipeline grows (entry/exit, compile, conditional edges, repo scanner, etc.).*
