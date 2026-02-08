"""Node types for workflow graph."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedNode:
    """A node extracted from an add_node(name, callable) call."""

    name: str
    callable_ref: str
    line: int = 0
