"""Golden: explicit set_entry_point instead of START edge."""
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge("a", "b")
graph.add_edge("b", END)
graph.set_entry_point("b")
