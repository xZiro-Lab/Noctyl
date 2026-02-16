"""Phase 2 analysis: control-flow, metrics, node annotation, structural risk, GraphAnalyzer."""

from noctyl.analysis.analyzer import GraphAnalyzer, analyze
from noctyl.analysis.control_flow import compute_control_flow
from noctyl.analysis.digraph import build_digraph
from noctyl.analysis.metrics import compute_metrics
from noctyl.analysis.node_annotation import compute_node_annotations
from noctyl.analysis.structural_risk import compute_structural_risk

__all__ = [
    "GraphAnalyzer",
    "analyze",
    "build_digraph",
    "compute_control_flow",
    "compute_metrics",
    "compute_node_annotations",
    "compute_structural_risk",
]
