"""
Parallel processing workflow: multiple agents work in parallel then merge.

Demonstrates:
- Parallel agent execution paths
- Merge/synchronization point
- Multiple conditional edges from different nodes
"""

from langgraph.graph import StateGraph, START, END


def agent_a(state):
    return {"result_a": "done", **state}


def agent_b(state):
    return {"result_b": "done", **state}


def agent_c(state):
    return {"result_c": "done", **state}


def merger(state):
    return {"merged": True, **state}


def check_a(state):
    return "continue" if state.get("result_a") else "wait"


def check_b(state):
    return "continue" if state.get("result_b") else "wait"


# Parallel workflow graph
parallel_graph = StateGraph(dict)

parallel_graph.add_node("agent_a", agent_a)
parallel_graph.add_node("agent_b", agent_b)
parallel_graph.add_node("agent_c", agent_c)
parallel_graph.add_node("merger", merger)

# Start -> both A and B in parallel
parallel_graph.add_edge(START, "agent_a")
parallel_graph.add_edge(START, "agent_b")

# A and B both conditionally go to C or wait
parallel_graph.add_conditional_edges("agent_a", check_a, {"continue": "agent_c", "wait": "agent_a"})
parallel_graph.add_conditional_edges("agent_b", check_b, {"continue": "agent_c", "wait": "agent_b"})

# C -> merger -> END
parallel_graph.add_edge("agent_c", "merger")
parallel_graph.add_edge("merger", END)
