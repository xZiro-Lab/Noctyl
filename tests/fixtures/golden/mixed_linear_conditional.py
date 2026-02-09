"""Golden: linear chain then conditional at end START -> A -> B, B -> C or END."""
from langgraph.graph import StateGraph, START, END

def fa(x): return x
def fb(x): return x
def fc(x): return x
def router(x): return "out"

graph = StateGraph(dict)
graph.add_node("a", fa)
graph.add_node("b", fb)
graph.add_node("c", fc)
graph.add_edge(START, "a")
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"next": "c", "done": END})
graph.add_edge("c", END)
