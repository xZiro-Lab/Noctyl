"""
Noctyl CLI: command-line interface for token estimation.
"""

import argparse
import json
import sys
from pathlib import Path

from noctyl.ingestion import run_pipeline_on_directory


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code: 0 on success, 1 on errors/warnings
    """
    parser = argparse.ArgumentParser(
        description="Noctyl: Static token usage estimator for LangGraph workflows"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate token usage")
    estimate_parser.add_argument("path", help="Path to directory or file")
    estimate_parser.add_argument(
        "--profile",
        help="Model profile YAML file path (default: use default profile)",
    )
    estimate_parser.add_argument(
        "--output",
        help="Output JSON file path (default: print to stdout)",
    )
    
    args = parser.parse_args()
    
    if args.command == "estimate":
        return _run_estimate(args.path, args.profile, args.output)
    else:
        parser.print_help()
        return 1


def _run_estimate(path: str, profile: str | None, output: str | None) -> int:
    """
    Run the estimate command.
    
    Args:
        path: Path to directory or file
        profile: Optional profile file path
        output: Optional output file path (None = stdout)
    
    Returns:
        Exit code: 0 on success, 1 on errors/warnings
    """
    try:
        # Validate path exists
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            return 1
        
        # Run pipeline
        results, warnings = run_pipeline_on_directory(
            path, estimate=True, profile=profile
        )
        
        # Output JSON
        output_json = json.dumps(results, indent=2, sort_keys=True)
        
        if output:
            # Write to file
            output_path = Path(output)
            output_path.write_text(output_json, encoding="utf-8")
        else:
            # Print to stdout
            print(output_json)
        
        # Print warnings to stderr
        if warnings:
            for warning in warnings:
                print(f"Warning: {warning}", file=sys.stderr)
            # Return non-zero exit code if warnings present
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
