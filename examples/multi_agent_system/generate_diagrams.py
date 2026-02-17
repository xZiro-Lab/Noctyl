"""Generate Mermaid diagrams for all workflows in this example repo."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from noctyl_scout.graph import workflow_dict_to_mermaid
from noctyl_scout.ingestion import run_pipeline_on_directory

if __name__ == "__main__":
    example_dir = Path(__file__).parent
    results, warnings = run_pipeline_on_directory(example_dir)
    
    print(f"Found {len(results)} workflow(s)\n")
    
    for d in results:
        graph_id = d["graph_id"]
        mermaid = workflow_dict_to_mermaid(d)
        print(f"{'='*60}")
        print(f"Graph: {graph_id}")
        print(f"{'='*60}")
        print(mermaid)
        print("\n")
        
        # Save to file
        output_file = example_dir / "generated" / f"{graph_id}.mmd"
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(mermaid)
        print(f"Saved to: {output_file}\n")
