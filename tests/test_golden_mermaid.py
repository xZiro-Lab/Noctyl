"""
Generate Mermaid diagrams for all golden fixtures and write to files for viewing.

Run: pytest tests/test_golden_mermaid.py -v
Then open tests/fixtures/golden/generated/*.mmd in Mermaid Live Editor or any Mermaid viewer.
"""

from pathlib import Path

from noctyl.graph import workflow_dict_to_mermaid
from noctyl.ingestion import run_pipeline_on_directory

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
GENERATED_DIR = GOLDEN_DIR / "generated"


def _safe_filename(graph_id: str, index: int) -> str:
    """Turn graph_id (e.g. 'linear_workflow.py:0') into a safe .mmd filename."""
    base = graph_id.replace(":", "_").replace(".py", "").replace(" ", "_")
    return f"{base}.mmd"


def test_generate_mermaid_for_all_golden():
    """Run pipeline on golden dir, generate Mermaid for each graph, write to generated/*.mmd."""
    results, warnings = run_pipeline_on_directory(GOLDEN_DIR)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(results):
        mermaid = workflow_dict_to_mermaid(d)
        graph_id = d.get("graph_id", f"graph_{i}")
        name = _safe_filename(graph_id, i)
        path = GENERATED_DIR / name
        path.write_text(mermaid, encoding="utf-8")
    # Assert we generated at least one per golden fixture (8 files, 9 graphs)
    mmd_files = list(GENERATED_DIR.glob("*.mmd"))
    assert len(mmd_files) >= 8, f"expected at least 8 .mmd files, got {len(mmd_files)}"
    # Each file non-empty and valid-looking
    for p in mmd_files:
        content = p.read_text(encoding="utf-8")
        assert content.startswith("flowchart TB"), f"{p.name} should start with flowchart TB"
        assert len(content) > 50, f"{p.name} should have content"


def test_verify_generated_mermaid_matches_pipeline():
    """Re-run pipeline, regenerate Mermaid for each graph; verify saved .mmd matches and has valid structure."""
    results, _ = run_pipeline_on_directory(GOLDEN_DIR)
    assert results, "pipeline should return at least one graph"
    for i, d in enumerate(results):
        graph_id = d.get("graph_id", f"graph_{i}")
        name = _safe_filename(graph_id, i)
        path = GENERATED_DIR / name
        mermaid = workflow_dict_to_mermaid(d)
        # Structure: must contain flowchart TB, Start, EndNode
        assert mermaid.startswith("flowchart TB"), "generated must be flowchart TB"
        assert "Start" in mermaid and "EndNode" in mermaid, "must define Start and EndNode"
        node_names = {n["name"] for n in d.get("nodes", [])}
        for name in node_names:
            assert name in mermaid, f"node {name} must appear in Mermaid"
        # If file was written by test_generate_mermaid_for_all_golden, content should match
        if path.exists():
            saved = path.read_text(encoding="utf-8")
            assert saved == mermaid, f"{path.name}: saved content should match current pipeline output (run test_generate_mermaid_for_all_golden to refresh)"
