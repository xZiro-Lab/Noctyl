"""Simple single-agent workflow for comparison."""
from langgraph.graph import StateGraph, START, END

def agent(state):
    return state

simple = StateGraph(dict)
simple.add_node("agent", agent)
simple.add_edge(START, "agent")
simple.add_edge("agent", END)
