"""
Workflow aggregation: aggregate per-node envelopes into workflow-level envelope.
"""

from __future__ import annotations

from collections import deque

from noctyl.analysis.digraph import DirectedGraph
from noctyl.estimation.data_model import CostEnvelope


def _find_paths_to_terminal(
    digraph: DirectedGraph,
    start: str,
    terminal: str,
    per_node: dict[str, CostEnvelope],
) -> list[CostEnvelope]:
    """
    Find all paths from start to terminal and return envelopes for each path.
    
    Returns list of CostEnvelope (one per path).
    """
    envelopes: list[CostEnvelope] = []
    
    # BFS to find all paths
    queue: deque[tuple[str, CostEnvelope]] = deque()
    
    # Start with initial envelope (from start node)
    start_env = per_node.get(start)
    if start_env is None:
        start_env = CostEnvelope(
            min_tokens=0,
            expected_tokens=0,
            max_tokens=0,
            bounded=True,
            confidence="structural-static",
        )
    
    queue.append((start, start_env))
    visited: set[tuple[str, ...]] = set()  # Track (node, path_signature) to avoid infinite loops
    
    while queue:
        current, path_env = queue.popleft()
        
        # Check if we reached the terminal
        if current == terminal:
            envelopes.append(path_env)
            continue
        
        # Explore successors
        for successor in digraph.successors(current):
            # Get successor envelope
            succ_env = per_node.get(successor)
            if succ_env is None:
                succ_env = CostEnvelope(
                    min_tokens=0,
                    expected_tokens=0,
                    max_tokens=0,
                    bounded=True,
                    confidence="structural-static",
                )
            
            # Accumulate envelope along path
            new_env = CostEnvelope(
                min_tokens=path_env.min_tokens + succ_env.min_tokens,
                expected_tokens=path_env.expected_tokens + succ_env.expected_tokens,
                max_tokens=path_env.max_tokens + succ_env.max_tokens,
                bounded=path_env.bounded and succ_env.bounded,
                confidence="structural-static",
            )
            
            # Simple cycle prevention: limit path length
            # (More sophisticated: track visited nodes in path)
            path_key = (current, successor)
            if path_key not in visited:
                visited.add(path_key)
                queue.append((successor, new_env))
    
    return envelopes


def aggregate_workflow_envelope(
    per_node: dict[str, CostEnvelope],
    entry_point: str | None,
    terminal_nodes: tuple[str, ...],
    digraph: DirectedGraph,
) -> CostEnvelope:
    """
    Aggregate workflow-level envelope from entry point to terminal nodes.
    
    For each terminal node:
    - Find all paths from entry_point to terminal
    - Sum envelope along each path
    - Track min/expected/max across all paths to terminals
    
    Aggregate:
    - min_tokens = minimum across all terminal paths
    - expected_tokens = sum of expected across all terminal paths
    - max_tokens = maximum across all terminal paths
    - bounded = True only if all paths have bounded=True
    - confidence = "structural-static"
    
    Returns workflow-level CostEnvelope.
    """
    # Edge cases
    if entry_point is None or not terminal_nodes:
        return CostEnvelope(
            min_tokens=0,
            expected_tokens=0,
            max_tokens=0,
            bounded=True,
            confidence="structural-static",
        )
    
    # Collect envelopes for all paths to all terminals
    all_path_envelopes: list[CostEnvelope] = []
    
    for terminal in terminal_nodes:
        # Find all paths from entry_point to this terminal
        path_envelopes = _find_paths_to_terminal(digraph, entry_point, terminal, per_node)
        
        if path_envelopes:
            # For multiple paths to same terminal, take the maximum (worst case)
            max_env = path_envelopes[0]
            for env in path_envelopes[1:]:
                if env.max_tokens > max_env.max_tokens:
                    max_env = env
            all_path_envelopes.append(max_env)
        else:
            # No path found -> use terminal node envelope directly
            terminal_env = per_node.get(terminal)
            if terminal_env is None:
                terminal_env = CostEnvelope(
                    min_tokens=0,
                    expected_tokens=0,
                    max_tokens=0,
                    bounded=True,
                    confidence="structural-static",
                )
            all_path_envelopes.append(terminal_env)
    
    if not all_path_envelopes:
        # No paths found -> return zero envelope
        return CostEnvelope(
            min_tokens=0,
            expected_tokens=0,
            max_tokens=0,
            bounded=True,
            confidence="structural-static",
        )
    
    # Aggregate across all terminals
    min_tokens = min(env.min_tokens for env in all_path_envelopes)
    max_tokens = max(env.max_tokens for env in all_path_envelopes)
    expected_tokens_raw = sum(env.expected_tokens for env in all_path_envelopes)  # Sum heuristic
    # Ensure expected is between min and max (invariant)
    expected_tokens = max(min_tokens, min(max_tokens, expected_tokens_raw))
    all_bounded = all(env.bounded for env in all_path_envelopes)
    
    return CostEnvelope(
        min_tokens=min_tokens,
        expected_tokens=expected_tokens,
        max_tokens=max_tokens,
        bounded=all_bounded,
        confidence="structural-static",
    )
