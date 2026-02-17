"""
Branch envelope computation: compute min/expected/max across conditional paths.
"""

from __future__ import annotations

from collections import deque

from noctyl.analysis.digraph import DirectedGraph
from noctyl.estimation.data_model import CostEnvelope
from noctyl.graph.edges import ExtractedConditionalEdge


def _find_paths_to_terminals(
    digraph: DirectedGraph,
    start: str,
    terminal_nodes: set[str],
    per_node: dict[str, CostEnvelope],
) -> list[tuple[list[str], CostEnvelope]]:
    """
    Find all paths from start to terminal nodes and compute envelope for each path.
    
    Returns list of (path_nodes, envelope) tuples.
    """
    paths: list[tuple[list[str], CostEnvelope]] = []
    
    # BFS to find all paths
    queue: deque[tuple[str, list[str], CostEnvelope]] = deque()
    
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
    
    queue.append((start, [start], start_env))
    visited_paths: set[tuple[str, ...]] = set()  # Track paths to avoid cycles
    
    while queue:
        current, path, path_env = queue.popleft()
        
        # Check if we reached a terminal
        if current in terminal_nodes or current == "END":
            paths.append((path, path_env))
            continue
        
        # Explore successors
        for successor in digraph.successors(current):
            # Avoid revisiting nodes in path (prevent infinite loops)
            if successor in path:
                continue
            
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
            new_path = path + [successor]
            new_env = CostEnvelope(
                min_tokens=path_env.min_tokens + succ_env.min_tokens,
                expected_tokens=path_env.expected_tokens + succ_env.expected_tokens,
                max_tokens=path_env.max_tokens + succ_env.max_tokens,
                bounded=path_env.bounded and succ_env.bounded,
                confidence="structural-static",
            )
            
            # Check if we've seen this path before
            path_key = tuple(new_path)
            if path_key not in visited_paths:
                visited_paths.add(path_key)
                queue.append((successor, new_path, new_env))
    
    return paths


def compute_branch_envelopes(
    digraph: DirectedGraph,
    per_node: dict[str, CostEnvelope],
    conditional_edges: tuple[ExtractedConditionalEdge, ...],
) -> dict[str, CostEnvelope]:
    """
    Compute branch envelopes for conditional paths.
    
    For each branch point (source of conditional edges):
    - Collect all conditional paths
    - For each path, traverse to terminal/END and sum envelopes
    - Compute branch envelope:
      - min_cost = minimum across all paths
      - max_cost = maximum across all paths
      - expected_cost = (min + max) / 2 (midpoint heuristic)
      - bounded = True if all paths have bounded=True
    
    Returns dict mapping path_key → CostEnvelope where path_key = "{source}:{condition_label}".
    """
    if not conditional_edges:
        return {}
    
    # Group conditional edges by source node
    branches: dict[str, list[ExtractedConditionalEdge]] = {}
    for edge in conditional_edges:
        if edge.source not in branches:
            branches[edge.source] = []
        branches[edge.source].append(edge)
    
    # Get terminal nodes from digraph (nodes with no successors or END)
    terminal_nodes: set[str] = {"END"}
    for node in digraph.nodes():
        if node != "END" and not digraph.successors(node):
            terminal_nodes.add(node)
    
    per_path: dict[str, CostEnvelope] = {}
    
    # Process each branch point
    for source_node, edges in branches.items():
            # For each conditional path from this branch point
            path_envelopes: list[CostEnvelope] = []
            
            for edge in edges:
                target = edge.target
                
                # Find paths from target to terminals
                paths = _find_paths_to_terminals(digraph, target, terminal_nodes, per_node)
                
                if not paths:
                    # No path found -> use target node envelope directly
                    target_env = per_node.get(target)
                    if target_env is None:
                        target_env = CostEnvelope(
                            min_tokens=0,
                            expected_tokens=0,
                            max_tokens=0,
                            bounded=True,
                            confidence="structural-static",
                        )
                    path_envelopes.append(target_env)
                else:
                    # Use the maximum envelope across all paths (worst case for this branch)
                    max_env = paths[0][1]
                    for _, env in paths[1:]:
                        if env.max_tokens > max_env.max_tokens:
                            max_env = env
                    path_envelopes.append(max_env)
            
            # Compute branch envelope: min/expected/max across all paths
            if path_envelopes:
                min_costs = [env.min_tokens for env in path_envelopes]
                max_costs = [env.max_tokens for env in path_envelopes]
                expected_costs = [env.expected_tokens for env in path_envelopes]
                bounded_flags = [env.bounded for env in path_envelopes]
                
                min_cost = min(min_costs)
                max_cost = max(max_costs)
                # Ensure expected is between min and max
                expected_cost = max(min_cost, min(max_cost, (min_cost + max_cost) // 2))
                all_bounded = all(bounded_flags)
                
                # Store envelope for each path
                for edge in edges:
                    path_key = f"{source_node}:{edge.condition_label}"
                    per_path[path_key] = CostEnvelope(
                        min_tokens=min_cost,
                        expected_tokens=expected_cost,
                        max_tokens=max_cost,
                        bounded=all_bounded,
                        confidence="structural-static",
                    )
    
    return per_path
