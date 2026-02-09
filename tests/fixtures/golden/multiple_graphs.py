"""Golden: two StateGraphs in one file."""
from langgraph.graph import StateGraph, START, END

def f1(x): return x
def f2(x): return x
def g1(x): return x
def g2(x): return x

graph1 = StateGraph(dict)
graph1.add_node("x", f1)
graph1.add_node("y", f2)
graph1.add_edge(START, "x")
graph1.add_edge("x", "y")
graph1.add_edge("y", END)

graph2 = StateGraph(dict)
graph2.add_node("p", g1)
graph2.add_node("q", g2)
graph2.add_edge(START, "p")
graph2.add_edge("p", "q")
graph2.add_edge("q", END)
