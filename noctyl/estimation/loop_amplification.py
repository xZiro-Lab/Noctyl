"""
Loop amplification: multiply cycle node envelopes by iteration count.
"""

from __future__ import annotations

from noctyl.estimation.data_model import CostEnvelope, ModelProfile
from noctyl.graph.execution_model import DetectedCycle


def apply_loop_amplification(
    per_node: dict[str, CostEnvelope],
    cycles: tuple[DetectedCycle, ...],
    model_profile: ModelProfile,
    assumed_iterations: int = 5,
) -> dict[str, CostEnvelope]:
    """
    Apply loop amplification to cycle nodes.
    
    For each DetectedCycle:
    - Extract cycle nodes
    - Determine iteration count:
      - Bounded loops: use static bound (not implemented in Phase 2, so assume unbounded)
      - Unbounded loops: use assumed_iterations (default 5)
      - Non-terminating cycles: use assumed_iterations
    - Multiply each cycle node's envelope by iteration count
    - Set bounded=False if iteration count is assumed
    
    Returns updated per_node dict.
    """
    # Create a copy to avoid mutating input
    result = dict(per_node)
    
    # Track which nodes are in cycles
    cycle_node_sets: list[set[str]] = []
    for cycle in cycles:
        cycle_node_sets.append(set(cycle.nodes))
    
    # Process each cycle
    for cycle in cycles:
        cycle_nodes = set(cycle.nodes)
        
        # Determine iteration count
        # Phase 2 DetectedCycle doesn't have iteration bounds yet,
        # so all cycles are treated as unbounded
        iterations = assumed_iterations
        
        # Check if cycle is non-terminating (will generate warning separately)
        is_non_terminating = cycle.cycle_type == "non_terminating"
        
        # For each node in cycle, multiply envelope by iterations
        for node_name in cycle_nodes:
            if node_name not in result:
                # Node not in per_node (shouldn't happen, but handle gracefully)
                continue
            
            env = result[node_name]
            
            # Multiply envelope by iteration count
            new_min = env.min_tokens * iterations
            new_expected = env.expected_tokens * iterations
            new_max = env.max_tokens * iterations
            
            # Set bounded=False if iteration count is assumed (not statically known)
            # For now, all cycles are unbounded, so always False
            bounded = False
            
            result[node_name] = CostEnvelope(
                min_tokens=new_min,
                expected_tokens=new_expected,
                max_tokens=new_max,
                bounded=bounded,
                confidence=env.confidence,
            )
    
    return result
