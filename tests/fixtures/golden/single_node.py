"""Golden: single node START -> A -> END."""
from langgraph.graph import StateGraph, START, END

def fa(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_edge(START, "a")
graph.add_edge("a", END)
