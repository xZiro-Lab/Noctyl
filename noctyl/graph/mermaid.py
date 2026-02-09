"""
Generate Mermaid flowchart from extracted workflow dict (graph of agents).
"""

from __future__ import annotations

# Mermaid reserves "end", "subgraph", etc.; use safe node IDs for START/END
START_ID = "Start"
END_ID = "EndNode"


def _mermaid_node_id(name: str) -> str:
    """Map workflow node name to Mermaid-safe node ID (no spaces, avoid reserved)."""
    if name == "START":
        return START_ID
    if name == "END":
        return END_ID
    # Use name as-is if alphanumeric + underscore; else quote
    if name.replace("_", "").isalnum():
        return name
    return f'"{name}"'


def workflow_dict_to_mermaid(d: dict) -> str:
    """
    Produce a Mermaid flowchart string from a workflow dict (from workflow_graph_to_dict).

    Nodes (agents) are labeled by name; directed edges include sequential and
    conditional; START and END are shown as distinct nodes. Entry and terminal
    nodes are present via edges (START -> entry_point; terminal -> END).

    Args:
        d: Dict with keys nodes, edges, conditional_edges (each list);
           entry_point (str | None); optional terminal_nodes.

    Returns:
        Mermaid flowchart string (flowchart TB).
    """
    lines = ["flowchart TB"]
    node_names = {n["name"] for n in d.get("nodes", [])}
    # Define START and END nodes (rounded for visibility)
    lines.append(f"  {START_ID}([START])")
    lines.append(f"  {END_ID}([END])")
    for n in d.get("nodes", []):
        name = n["name"]
        nid = _mermaid_node_id(name)
        # Node label: show name as agent/step
        lines.append(f'  {nid}["{name}"]')
    # Sequential edges
    for e in d.get("edges", []):
        src = _mermaid_node_id(e["source"])
        tgt = _mermaid_node_id(e["target"])
        lines.append(f"  {src} --> {tgt}")
    # Conditional edges (with label)
    for e in d.get("conditional_edges", []):
        src = _mermaid_node_id(e["source"])
        tgt = _mermaid_node_id(e["target"])
        label = e.get("condition_label", "?")
        if not label.replace("_", "").isalnum():
            label = f'"{label}"'
        lines.append(f"  {src} -->|{label}| {tgt}")
    return "\n".join(lines)
