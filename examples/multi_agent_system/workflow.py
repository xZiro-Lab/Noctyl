"""
Complex multi-agent system workflow example for Noctyl testing.

This workflow demonstrates:
- Multiple agents (coordinator, researcher, writer, reviewer)
- Conditional routing (approve/reject loops)
- Parallel paths (research and writing)
- Entry point and terminal nodes
- Realistic multi-agent coordination patterns
"""

from langgraph.graph import StateGraph, START, END


def coordinator(state):
    """Main coordinator agent that routes tasks."""
    return {"task": state.get("task", ""), "status": "coordinated"}


def researcher(state):
    """Research agent that gathers information."""
    return {"research": "done", **state}


def writer(state):
    """Writer agent that creates content."""
    return {"content": "draft", **state}


def reviewer(state):
    """Review agent that evaluates quality."""
    quality = state.get("quality", "good")
    return {"review": "complete", "quality": quality}


def quality_check(state):
    """Quality gate decision function."""
    quality = state.get("quality", "good")
    return "approve" if quality == "good" else "reject"


def task_router(state):
    """Route tasks to appropriate agents."""
    task_type = state.get("task_type", "research")
    return "research" if task_type == "research" else "write"


# Main workflow graph
workflow = StateGraph(dict)

# Add all agent nodes
workflow.add_node("coordinator", coordinator)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

# Entry: START -> coordinator
workflow.add_edge(START, "coordinator")

# Coordinator routes to research or writing
workflow.add_conditional_edges(
    "coordinator",
    task_router,
    {
        "research": "researcher",
        "write": "writer",
    },
)

# Research path: researcher -> reviewer
workflow.add_edge("researcher", "reviewer")

# Writing path: writer -> reviewer
workflow.add_edge("writer", "reviewer")

# Reviewer: approve -> END, reject -> coordinator (loop)
workflow.add_conditional_edges(
    "reviewer",
    quality_check,
    {
        "approve": END,
        "reject": "coordinator",  # Loop back for revision
    },
)

# Set explicit entry point
workflow.set_entry_point("coordinator")
