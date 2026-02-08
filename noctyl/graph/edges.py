"""Edge types for workflow graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedEdge:
    """A directed edge extracted from an add_edge(source, target) call."""

    source: str
    target: str
    line: int = 0
