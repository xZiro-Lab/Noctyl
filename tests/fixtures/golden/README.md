# Golden LangGraph fixtures (Phase-1)

Canonical examples used to validate graph extraction. Each file is a minimal, self-contained LangGraph workflow. Tests in `tests/test_golden.py` run `run_pipeline_on_directory` on this directory and assert the extracted graph structure.

**Generated Mermaid diagrams:** Run `pytest tests/test_golden_mermaid.py -v` to generate `generated/*.mmd` (one per extracted graph). Open any `.mmd` file in [Mermaid Live Editor](https://mermaid.live) to view the graph of agents.

## Required cases (issue)

- **linear_workflow.py** — Linear: START -> A -> B -> END.
- **conditional_loop.py** — Conditional loop: START -> A, A conditional to loop/next/done (one target END).
- **end_termination.py** — END only: START -> A -> END.

## Additional cases

- **multiple_graphs.py** — Two StateGraphs in one file; validates correct attribution per graph_id.
- **set_entry_point_explicit.py** — Uses `set_entry_point("b")`; validates explicit entry point.
- **mixed_linear_conditional.py** — Linear chain then conditional at end (START -> A -> B, B conditional to C or END).
- **single_node.py** — Single node A, START -> A -> END.
- **multiple_conditional_nodes.py** — Nodes A and B each have conditional_edges; both can transition to END.
