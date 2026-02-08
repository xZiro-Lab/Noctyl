"""Edge types for workflow graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedEdge:
    """A directed edge extracted from an add_edge(source, target) call."""

    source: str
    target: str
    line: int = 0


@dataclass(frozen=True)
class ExtractedConditionalEdge:
    """A conditional edge extracted from add_conditional_edges path_map (one per mapping)."""

    source: str
    condition_label: str
    target: str
    line: int = 0
