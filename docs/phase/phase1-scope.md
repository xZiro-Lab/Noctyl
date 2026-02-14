# Phase-1 Scope: LangGraph Workflow Graph Extraction

Scope and constraints for Phase-1 of Noctyl. Focus: **LangGraph workflow graph extraction only**.

---

## 1. Supported language(s)

**Questions to answer:**

- Which language(s) will we support in Phase-1?
- Python only, or also JavaScript/TypeScript (e.g. `@langchain/langgraph`)?
- If multiple: equal support or one primary?

**Decision:**


| Item                       | Decision                  | Notes                                                                              |
| -------------------------- | ------------------------- | ---------------------------------------------------------------------------------- |
| Primary language           | Python                    | Only language in Phase-1.                                                          |
| Other languages in Phase-1 | None                      | JS/TS, etc. out of scope for Phase-1.                                              |
| Priority if multiple       | N/A                       | Single language.                                                                   |
| Minimum Python version     | 3.10+                     | Industry standard; good AST/typing support; keeps scope minimal.                   |
| Async graph/node support   | In scope (structure only) | Parse async def nodes and async graph usage; we don't execute, so extraction only. |


**Status:** [x] Locked

---

## 2. Supported framework(s)

**Questions to answer:**

- Is Phase-1 strictly LangGraph-only, or LangGraph + LangChain?
- Which LangGraph surface do we target (e.g. `StateGraph`, `Annotation`, `add_node`, `add_edge`, `compile`)?
- Do we support subgraphs / nested graphs in Phase-1?
- Do we support LangGraph “prebuilt” or higher-level APIs, or only the core graph API?

**Decision:**


| Item                                 | Decision                                                 | Notes                                                                       |
| ------------------------------------ | -------------------------------------------------------- | --------------------------------------------------------------------------- |
| Framework scope                      | LangGraph only                                           | LangChain out of scope for Phase-1.                                         |
| Target API surface                   | Core graph API only                                      | StateGraph, add_node, add_edge, add_conditional_edges, entry/exit, compile. |
| Subgraphs / nested graphs            | Out of scope                                             | No subgraphs in Phase-1.                                                    |
| Prebuilt / high-level APIs           | Out of scope                                             | No prebuilt agents (e.g. create_react_agent) in Phase-1.                    |
| LangGraph versions                   | Current stable (e.g. 0.2.x); document supported versions | One major back for compatibility.                                           |
| Conditional edge content             | Structure only                                           | "There is a conditional edge"; routing logic out of scope.                  |
| Human-in-the-loop / interrupt_*      | In scope as graph structure                              | Model as nodes/edges where applicable.                                      |
| State schema (Annotation, TypedDict) | Out of scope                                             | Only nodes/edges in Phase-1.                                                |
| Entry node / end node                | First-class                                              | Part of API; model explicitly.                                              |


**Status:** [x] Locked

---

## 3. Analysis type (static / dynamic)

**Questions to answer:**

- Are we static-only, dynamic-only, or hybrid?
- If static: what’s the minimum we need (AST, import resolution, path discovery)?
- If we ever add dynamic: what’s the boundary (e.g. discovery only vs full execution)?
- How do we handle graph construction that depends on config or environment?

**Decision:**


| Item                                       | Decision                                                               | Notes                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| Static / dynamic / hybrid                  | Static-only                                                            | No execution of user code in Phase-1.                        |
| Minimum static machinery                   | AST parsing, import resolution (LangGraph + same-repo), path discovery | Enough to find and resolve LangGraph API calls across files. |
| Config/env-dependent graphs                | Unsupported; warn and report partial/ambiguous                         | Only unconditional graph construction is supported.          |
| Future dynamic boundary (if any)           | Discovery-only (e.g. get graph via compile()); no workflow execution   | Document only for later phases.                              |
| Unresolvable (dynamic name, config-driven) | Best-effort + warning                                                  | Emit partial graph when possible; don't fail hard.           |
| Follow imports into site-packages          | No; project code only                                                  | Don't follow into site-packages/vendor.                      |
| Definition of success                      | Emit when we have ≥1 graph                                             | Allow partial + warn; not only "when confident complete".    |


**Status:** [x] Locked

---

## 4. Output format(s)

**Questions to answer:**

- What is the primary output: schema (e.g. JSON), diagram (e.g. Mermaid), or both?
- Do we need a stable, versioned schema for the graph?
- Who consumes the output (humans, other tools, AI context)?
- Do we need multiple views (e.g. full graph vs summary) in Phase-1?

**Decision:**


| Item                      | Decision                              | Notes                                                              |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| Primary output            | JSON (schema)                         | Canonical graph representation.                                    |
| Diagram                   | Derived from JSON (e.g. Mermaid)      | Secondary; for humans.                                             |
| Schema versioning         | Yes                                   | Stable, versioned schema (e.g. schema_version and/or JSON Schema). |
| Consumers                 | Humans, other tools, AI context       | All three.                                                         |
| Multiple views in Phase-1 | Full graph required; summary optional | One canonical graph view; summary = optional derivative.           |
| Formal machine contract   | Yes (JSON Schema or schema_version)   | Not best-effort JSON.                                              |
| Determinism               | Yes                                   | Same repo + same tool version → same output.                       |
| Phase-2 compatibility     | Yes                                   | Versioned schema so Phase-2 can consume without breakage.          |


**Status:** [x] Locked

---

## 5. In-scope vs out-of-scope

**Questions to answer:**

- Is “graph extraction” limited to topology (nodes, edges, types) or also metadata (prompts, tool names, state keys)?
- Do we extract only structure, or also risk/heuristic signals (e.g. loops, retries) in Phase-1?
- Are token estimation, cost modeling, and runtime execution explicitly out of scope?
- Are all non-LangGraph frameworks (LangChain-only, CrewAI, AutoGen, etc.) out of scope?
- What do we do when LangGraph and another framework appear in the same repo?

**Decision:**


| Item                                     | In scope | Out of scope | Notes                                                |
| ---------------------------------------- | -------- | ------------ | ---------------------------------------------------- |
| Graph topology                           | ✓        |              | Nodes, edges, types, entry/exit, conditional edges.  |
| Metadata (prompts, tools, state)         |          | ✓            | Phase-1; add in later phase.                         |
| Risk/heuristic signals                   |          | ✓            | Loops, retries, etc. defer to Phase-2.               |
| Token estimation                         |          | ✓            | Phase-1.                                             |
| Cost modeling                            |          | ✓            | Phase-1.                                             |
| Runtime execution                        |          | ✓            | Phase-1.                                             |
| Non-LangGraph frameworks                 |          | ✓            | Phase-1.                                             |
| Mixed repo (LangGraph + other)           | ✓        |              | Extract LangGraph only; ignore other framework code. |
| Conditional edge content (routing logic) |          | ✓            | Out of scope (structure only).                       |
| State schema extraction                  |          | ✓            | Already out.                                         |
| Persistence/checkpointing                |          | ✓            | Ignore in Phase-1; don't model.                      |


**Status:** [x] Locked

---

## 6. Boundaries and constraints

**Questions to answer:**

- What’s the smallest unit of analysis: file, package, or “entry point + reachable code”?
- How do we define “entry point” (e.g. `compile()` call or a specific pattern)?
- Do we support monorepos / multiple graphs per repo in Phase-1?
- What’s the maximum scale we design for (graph size, repo size)?

**Decision:**


| Item                         | Decision                                                                    | Notes                                                                                                |
| ---------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Unit of analysis             | Entry point + reachable code                                                | Start from graph entry; follow imports; analyze only code that contributes to that graph.            |
| Entry point definition       | StateGraph that has `.compile()` called on it                               | Discover by finding `.compile()` calls and resolving the graph object they're called on.             |
| Monorepos / multiple graphs  | Yes                                                                         | Multiple graphs per repo supported; discover all compile() sites; emit one graph per compiled graph. |
| Scale limits (design target) | Best-effort: < 500 nodes / < 2k edges per graph; < 5k Python files per repo | No hard guarantee in Phase-1; may add limits/timeouts later.                                         |
| CLI vs library               | CLI only in Phase-1                                                         | Importable library/API later.                                                                        |
| How repo is chosen           | User-provided path                                                          | Any directory; not git-only.                                                                         |
| Execute user code?           | No                                                                          | Static only; never import/run user code.                                                             |
| Timeout / resource caps      | Optional --max-files or timeout                                             | Bounded execution; document as best-effort.                                                          |


**Status:** [x] Locked

---

## 7. Acceptance and success

**Questions to answer:**

- What does “scope documented” mean (e.g. this doc, README section, ADR)?
- Who must agree on in-scope and out-of-scope?
- What’s the single Phase-1 success criterion?

**Decision:**


| Item                    | Decision                                                                                                                                                                                                                             | Notes                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Scope documentation     | This doc (`docs/phase1-scope.md`)                                                                                                                                                                                                    | "Scope documented" = all sections filled and locked, doc committed; optional: one-line pointer in README. |
| Agreement / sign-off    | Decisions locked in this doc; formal sign-off optional (e.g. maintainers or self)                                                                                                                                                    | Replace with your context (team, solo, OSS).                                                              |
| Success criterion       | Given a Python repo with at least one LangGraph workflow (StateGraph + `.compile()`), Noctyl produces a versioned JSON graph (nodes, edges, entry, exit) that correctly reflects the workflow structure, using static analysis only. | Single Phase-1 success criterion.                                                                         |
| Prove "done"            | Golden repos                                                                                                                                                                                                                         | At least one (or two) repos that produce a known graph; not ad-hoc only.                                  |
| "Agreed"                | Documented and merged                                                                                                                                                                                                                | No formal sign-off required.                                                                              |
| Scope doc after Phase-1 | Living doc                                                                                                                                                                                                                           | Update as we learn; not a frozen snapshot.                                                                |


**Status:** [x] Locked

---

## 8. Critical gaps — resolved

All items below are locked in §1–§7. This section summarizes cross-cutting decisions.

**Cross-cutting**


| Item                                                    | Decision                           | Notes                                                                  |
| ------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| Invalid code / unsupported pattern / missing dependency | Warn and continue; skip and report | Don't fail hard; emit partial when possible.                           |
| LangGraph installed in env?                             | No                                 | Parse without executing; no requirement for LangGraph to be installed. |


---

## Acceptance criteria (Phase-1)

- Scope documented (this document updated and committed).
- In-scope and out-of-scope agreed (documented and merged; formal sign-off optional — see §7).

---

*Last updated: 2026-02-07*