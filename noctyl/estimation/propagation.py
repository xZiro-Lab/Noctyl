"""
Token propagation: topological traversal applying T_out = (T_in + base_prompt) × expansion_factor.
"""

from __future__ import annotations

from collections import deque

from noctyl.analysis.digraph import DirectedGraph
from noctyl.estimation.data_model import CostEnvelope, ModelProfile, NodeTokenSignature


def _topological_sort(digraph: DirectedGraph) -> list[str]:
    """
    Kahn's algorithm for topological sort.
    Returns list of nodes in topological order (acyclic portions).
    Nodes in cycles will appear but order within cycles is arbitrary.
    """
    # Compute in-degrees
    in_degree: dict[str, int] = {node: 0 for node in digraph.nodes()}
    for node in digraph.nodes():
        for successor in digraph.successors(node):
            in_degree[successor] = in_degree.get(successor, 0) + 1
    
    # Initialize queue with nodes having no incoming edges
    queue: deque[str] = deque()
    for node in digraph.nodes():
        if in_degree[node] == 0:
            queue.append(node)
    
    result: list[str] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # Reduce in-degree for successors
        for successor in digraph.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)
    
    # Add remaining nodes (those in cycles)
    remaining = set(digraph.nodes()) - set(result)
    result.extend(sorted(remaining))  # Sort for determinism
    
    return result


def propagate_tokens(
    digraph: DirectedGraph,
    node_signatures: tuple[NodeTokenSignature, ...],
    model_profile: ModelProfile,
) -> dict[str, CostEnvelope]:
    """
    Propagate token quantities through the workflow graph.
    
    Formula: T_out = (T_in + base_prompt) × expansion_factor
    
    For each node:
    - If no predecessors (entry point): start with base_prompt_tokens
    - If has predecessors: accumulate tokens from all incoming paths
    - Multiple incoming edges: sum the envelopes (all paths contribute)
    
    Nodes in cycles are propagated normally but marked bounded=False initially
    (loop amplification will handle bounds).
    
    Returns dict mapping node_name → CostEnvelope.
    """
    # Build name → NodeTokenSignature map
    sig_map: dict[str, NodeTokenSignature] = {
        sig.node_name: sig for sig in node_signatures
    }
    
    # Initialize envelopes: START = (0, 0, 0), all others = (0, 0, 0)
    envelopes: dict[str, CostEnvelope] = {}
    
    # Initialize START node
    if "START" in digraph.nodes():
        envelopes["START"] = CostEnvelope(
            min_tokens=0,
            expected_tokens=0,
            max_tokens=0,
            bounded=True,
            confidence="structural-static",
        )
    
    # Get topological order
    topo_order = _topological_sort(digraph)
    
    # Track which nodes are in cycles (for bounded flag)
    cycle_nodes: set[str] = set()
    # Simple heuristic: if node has self-loop or is part of a cycle, mark it
    # (More precise detection would require cycle detection, but we'll mark bounded=False
    # for now and let loop_amplification handle it properly)
    
    # Propagate tokens in topological order
    for node in topo_order:
        if node == "START":
            continue  # Already initialized
        
        if node == "END":
            # END node: accumulate from all predecessors
            predecessors = digraph.predecessors(node)
            if not predecessors:
                # No predecessors -> zero envelope
                envelopes[node] = CostEnvelope(
                    min_tokens=0,
                    expected_tokens=0,
                    max_tokens=0,
                    bounded=True,
                    confidence="structural-static",
                )
            else:
                # Sum envelopes from all predecessors
                min_tokens = 0
                expected_tokens = 0
                max_tokens = 0
                all_bounded = True
                
                for pred in predecessors:
                    if pred in envelopes:
                        pred_env = envelopes[pred]
                        min_tokens += pred_env.min_tokens
                        expected_tokens += pred_env.expected_tokens
                        max_tokens += pred_env.max_tokens
                        if not pred_env.bounded:
                            all_bounded = False
                
                envelopes[node] = CostEnvelope(
                    min_tokens=min_tokens,
                    expected_tokens=expected_tokens,
                    max_tokens=max_tokens,
                    bounded=all_bounded,
                    confidence="structural-static",
                )
            continue
        
        # Get node signature (default if not found)
        sig = sig_map.get(node)
        if sig is None:
            # No signature -> use defaults
            base_prompt_tokens = 0
            expansion_factor = model_profile.expansion_factor
        else:
            base_prompt_tokens = sig.base_prompt_tokens
            expansion_factor = sig.expansion_factor
        
        # Get all predecessors
        all_predecessors = digraph.predecessors(node)
        has_start = "START" in all_predecessors
        predecessors = [p for p in all_predecessors if p != "START"]
        
        # If node has START as predecessor, it's an entry point
        # Start with base_prompt_tokens (expansion_factor applies to successors)
        if has_start and not predecessors:
            # Pure entry point (only START, no other predecessors)
            envelopes[node] = CostEnvelope(
                min_tokens=base_prompt_tokens,
                expected_tokens=base_prompt_tokens,
                max_tokens=base_prompt_tokens,
                bounded=True,
                confidence="structural-static",
            )
        elif has_start:
            # Entry point but also has other predecessors (e.g., in a cycle)
            # Still start with base_prompt_tokens, then accumulate from other predecessors
            min_tokens = base_prompt_tokens
            expected_tokens = base_prompt_tokens
            max_tokens = base_prompt_tokens
            all_bounded = True
            
            for pred in predecessors:
                if pred in envelopes:
                    pred_env = envelopes[pred]
                    pred_min = int((pred_env.min_tokens + base_prompt_tokens) * expansion_factor)
                    pred_expected = int((pred_env.expected_tokens + base_prompt_tokens) * expansion_factor)
                    pred_max = int((pred_env.max_tokens + base_prompt_tokens) * expansion_factor)
                    
                    min_tokens += pred_min
                    expected_tokens += pred_expected
                    max_tokens += pred_max
                    if not pred_env.bounded:
                        all_bounded = False
            
            # Check if node is in a cycle
            is_in_cycle = node in digraph.successors(node) or any(
                node in digraph.successors(p) for p in predecessors
            )
            if is_in_cycle:
                all_bounded = False
            
            envelopes[node] = CostEnvelope(
                min_tokens=min_tokens,
                expected_tokens=expected_tokens,
                max_tokens=max_tokens,
                bounded=all_bounded,
                confidence="structural-static",
            )
        elif not predecessors:
            # Entry point: start with base_prompt_tokens (no expansion_factor applied yet)
            # Expansion factor applies when this node's output goes to successors
            envelopes[node] = CostEnvelope(
                min_tokens=base_prompt_tokens,
                expected_tokens=base_prompt_tokens,
                max_tokens=base_prompt_tokens,
                bounded=True,
                confidence="structural-static",
            )
        else:
            # Accumulate tokens from all incoming paths
            min_tokens = 0
            expected_tokens = 0
            max_tokens = 0
            all_bounded = True
            
            for pred in all_predecessors:
                if pred in envelopes:
                    pred_env = envelopes[pred]
                    # Apply formula: T_out = (T_in + base_prompt) × expansion_factor
                    # For each path, compute the output tokens
                    pred_min = int((pred_env.min_tokens + base_prompt_tokens) * expansion_factor)
                    pred_expected = int((pred_env.expected_tokens + base_prompt_tokens) * expansion_factor)
                    pred_max = int((pred_env.max_tokens + base_prompt_tokens) * expansion_factor)
                    
                    # Accumulate across all paths
                    min_tokens += pred_min
                    expected_tokens += pred_expected
                    max_tokens += pred_max
                    
                    if not pred_env.bounded:
                        all_bounded = False
            
            # Check if node is in a cycle (simple heuristic: has self-loop or cycles back)
            is_in_cycle = node in digraph.successors(node)  # Self-loop
            if not is_in_cycle:
                # Check if any predecessor is this node (cycle)
                for pred in predecessors:
                    if pred == node or node in digraph.successors(pred):
                        # Check if there's a path back (simple cycle detection)
                        # More precise: would need full cycle detection
                        is_in_cycle = True
                        break
            
            if is_in_cycle:
                cycle_nodes.add(node)
                all_bounded = False  # Mark as unbounded initially
            
            envelopes[node] = CostEnvelope(
                min_tokens=min_tokens,
                expected_tokens=expected_tokens,
                max_tokens=max_tokens,
                bounded=all_bounded,
                confidence="structural-static",
            )
    
    return envelopes
