"""
Phase 2 ExecutionModel: enriched graph representation and deterministic JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from noctyl.graph.graph import WorkflowGraph, workflow_graph_to_dict

# Literal types per docs/phase/phase2.md
CycleType = Literal[
    "self_loop", "multi_node", "conditional", "non_terminating"
]
OriginType = Literal[
    "local_function", "imported_function", "class_method", "lambda", "unknown"
]
StateInteractionType = Literal["pure", "read_only", "mutates_state"]
RoleType = Literal["llm_like", "tool_like", "control_node", "unknown"]

ENRICHED_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class DetectedCycle:
    """One detected cycle with classification and optional termination reachability."""

    cycle_type: CycleType
    nodes: tuple[str, ...]  # sorted for determinism when constructed
    reaches_terminal: bool = True


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural metrics per phase2.md §6."""

    node_count: int
    edge_count: int
    entry_node: str | None
    terminal_nodes: tuple[str, ...]
    unreachable_nodes: tuple[str, ...]
    longest_acyclic_path: int
    avg_branching_factor: float
    max_depth_before_cycle: int | None


@dataclass(frozen=True)
class NodeAnnotation:
    """Per-node semantic annotation per phase2.md §7."""

    node_name: str
    origin: OriginType
    state_interaction: StateInteractionType
    role: RoleType


@dataclass(frozen=True)
class StructuralRisk:
    """Structural risk summary per phase2.md §9."""

    unreachable_node_ids: tuple[str, ...]
    dead_end_ids: tuple[str, ...]
    non_terminating_cycle_ids: tuple[str, ...]
    multiple_entry_points: bool


@dataclass(frozen=True)
class ExecutionModel:
    """Canonical Phase 2 representation: graph plus shape, cycles, metrics, annotations, risks."""

    graph: WorkflowGraph
    entry_point: str | None
    terminal_nodes: tuple[str, ...]
    shape: str  # linear | branching | cyclic | disconnected | invalid
    cycles: tuple[DetectedCycle, ...]
    metrics: StructuralMetrics
    node_annotations: tuple[NodeAnnotation, ...]
    risks: StructuralRisk


def execution_model_to_dict(model: ExecutionModel) -> dict:
    """
    Return a JSON-serializable dict with deterministic ordering.
    Includes base graph (via workflow_graph_to_dict) plus enriched fields.
    schema_version 2.0, enriched: true. No token/cost fields.
    """
    base = workflow_graph_to_dict(model.graph)
    base["schema_version"] = ENRICHED_SCHEMA_VERSION
    base["enriched"] = True
    base["shape"] = model.shape

    # Cycles: stable order by (cycle_type, tuple(nodes))
    cycles_sorted = sorted(
        model.cycles,
        key=lambda c: (c.cycle_type, c.nodes),
    )
    base["cycles"] = [
        {
            "cycle_type": c.cycle_type,
            "nodes": list(c.nodes),
            "reaches_terminal": c.reaches_terminal,
        }
        for c in cycles_sorted
    ]

    # Metrics: single dict
    m = model.metrics
    base["metrics"] = {
        "node_count": m.node_count,
        "edge_count": m.edge_count,
        "entry_node": m.entry_node,
        "terminal_nodes": list(m.terminal_nodes),
        "unreachable_nodes": list(m.unreachable_nodes),
        "longest_acyclic_path": m.longest_acyclic_path,
        "avg_branching_factor": m.avg_branching_factor,
        "max_depth_before_cycle": m.max_depth_before_cycle,
    }

    # Node annotations: sorted by node_name
    ann_sorted = sorted(model.node_annotations, key=lambda a: a.node_name)
    base["node_annotations"] = [
        {
            "node_name": a.node_name,
            "origin": a.origin,
            "state_interaction": a.state_interaction,
            "role": a.role,
        }
        for a in ann_sorted
    ]

    # Risks
    r = model.risks
    base["risks"] = {
        "unreachable_node_ids": list(r.unreachable_node_ids),
        "dead_end_ids": list(r.dead_end_ids),
        "non_terminating_cycle_ids": list(r.non_terminating_cycle_ids),
        "multiple_entry_points": r.multiple_entry_points,
    }

    return base
