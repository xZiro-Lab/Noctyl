"""Golden: conditional loop START -> A, A -> loop/next/done."""
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x
def router(x): return "done"

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_edge(START, "a")
graph.add_conditional_edges("a", router, {"loop": "a", "next": "b", "done": END})
graph.add_edge("b", END)
