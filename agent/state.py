"""
Shared LangGraph state.  Every node reads from and writes to this TypedDict.
"""
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The original user query
    query: str

    # Query analysis results
    query_type: Literal["simple", "complex", "multi_source"]
    expanded_terms: list[str]

    # Routing decision
    route: Literal["vector", "live_tools", "agent"]

    # Retrieved context
    vector_chunks: list[dict]       # from Qdrant
    live_results: list[dict]        # from CourtListener / Congress API / web
    merged_context: list[dict]      # after dedup + rerank

    # ReAct agent messages (for complex queries)
    messages: Annotated[list, add_messages]

    # Final output
    answer: str
    citations: list[str]

    # Metadata
    thread_id: str
    latency_ms: int
