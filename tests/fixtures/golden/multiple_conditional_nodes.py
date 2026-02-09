"""Golden: two nodes each with conditional_edges to END."""
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x
def ra(x): return "stop"
def rb(x): return "stop"

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_conditional_edges("a", ra, {"go": "b", "stop": END})
graph.add_conditional_edges("b", rb, {"go": "a", "stop": END})
