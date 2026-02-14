# Multi-Agent System Example

Complex LangGraph workflows demonstrating multi-agent coordination patterns. Use this repository to test Noctyl's graph extraction and diagram generation.

## Workflows

- **workflow.py** — Main multi-agent system: coordinator routes to researcher/writer, both feed into reviewer, with approve/reject loop back to coordinator.
- **parallel_workflow.py** — Parallel execution: agent_a and agent_b run in parallel, both conditionally route to agent_c, then merge and finish.
- **simple_agent.py** — Minimal single-agent workflow for comparison.

## Test with Noctyl

```python
from noctyl.ingestion import run_pipeline_on_directory
from noctyl.graph import workflow_dict_to_mermaid
from pathlib import Path

# Extract workflows
results, warnings = run_pipeline_on_directory(Path("examples/multi_agent_system"))

# Generate Mermaid diagrams
for d in results:
    mermaid = workflow_dict_to_mermaid(d)
    print(f"\n--- {d['graph_id']} ---")
    print(mermaid)
```

## Generate Diagrams

Run the included script to generate Mermaid diagrams:

```bash
# From project root
python examples/multi_agent_system/generate_diagrams.py
```

Diagrams are saved to `examples/multi_agent_system/generated/*.mmd`.

Or run tests: `pytest tests/test_example_multi_agent.py -v`
