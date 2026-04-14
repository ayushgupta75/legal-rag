"""
LangGraph graph definition.
Wires all nodes, conditional edges, and the Redis checkpointer together.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from agent.state import AgentState
from agent.nodes.nodes import (
    query_analysis_node,
    router_node,
    get_route,
    vector_retrieve_node,
    live_tools_node,
    react_agent_node,
    merge_context_node,
    generate_node,
)
from config import get_settings

settings = get_settings()


def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph agent graph.

    Graph structure:
        query_analysis → router
            router ──[vector]──→ vector_retrieve → merge_context → generate → END
            router ──[live_tools]→ live_tools → merge_context → generate → END
            router ──[agent]───→ react_agent → END
    """
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("query_analysis", query_analysis_node)
    g.add_node("router", router_node)
    g.add_node("vector_retrieve", vector_retrieve_node)
    g.add_node("live_tools", live_tools_node)
    g.add_node("react_agent", react_agent_node)
    g.add_node("merge_context", merge_context_node)
    g.add_node("generate", generate_node)

    # Entry point
    g.set_entry_point("query_analysis")

    # Fixed edges
    g.add_edge("query_analysis", "router")
    g.add_edge("vector_retrieve", "merge_context")
    g.add_edge("live_tools", "merge_context")
    g.add_edge("merge_context", "generate")
    g.add_edge("generate", END)
    g.add_edge("react_agent", END)  # ReAct agent handles its own generation

    # Conditional edge: router → one of three paths
    g.add_conditional_edges(
        "router",
        get_route,
        {
            "vector": "vector_retrieve",
            "live_tools": "live_tools",
            "agent": "react_agent",
        },
    )

    return g.compile(checkpointer=checkpointer)


def get_graph():
    """Return a compiled graph with Redis checkpointer."""
    checkpointer = RedisSaver.from_conn_string(settings.redis_url)
    return build_graph(checkpointer=checkpointer)
