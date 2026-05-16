"""
LangGraph node functions.
Each node takes AgentState and returns a partial state update (dict).
"""
import logging
from anthropic import Anthropic
from agent.state import AgentState
from agent.tools.live_tools import search_courtlistener, search_congress, search_ecfr
from agent.prompts.legal_prompts import QUERY_ANALYSIS_PROMPT, GENERATE_PROMPT
from ingestion.embedders.embedder import similarity_search, multi_similarity_search
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
anthropic = Anthropic(api_key=settings.anthropic_api_key)


# ── Node 1: Query Analysis ────────────────────────────────────────────────────

def query_analysis_node(state: AgentState) -> dict:
    """
    Classify the query and expand legal terminology.
    Returns: query_type, expanded_terms
    """
    prompt = QUERY_ANALYSIS_PROMPT.format(query=state["query"])
    response = anthropic.messages.create(
        model=settings.claude_model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Parse structured response from Claude
    query_type = "simple"
    expanded_terms = [state["query"]]
    for line in raw.splitlines():
        if line.startswith("TYPE:"):
            query_type = line.split(":", 1)[1].strip().lower()
        if line.startswith("TERMS:"):
            expanded_terms = [t.strip() for t in line.split(":", 1)[1].split(",")]

    logger.info(f"Query type: {query_type} | Terms: {expanded_terms}")
    return {"query_type": query_type, "expanded_terms": expanded_terms}


# ── Node 2: Router (conditional edge logic) ───────────────────────────────────

def router_node(state: AgentState) -> dict:
    """
    Decide retrieval route based on query type.
    - simple   → vector (fast path)
    - complex  → agent (ReAct loop)
    - multi_source → live_tools
    """
    route_map = {
        "simple": "vector",
        "complex": "agent",
        "multi_source": "live_tools",
    }
    route = route_map.get(state.get("query_type", "simple"), "vector")
    logger.info(f"Routing to: {route}")
    return {"route": route}


def get_route(state: AgentState) -> str:
    """Used as the conditional edge function in the graph."""
    return state.get("route", "vector")


# ── Node 3a: Vector Retrieve ──────────────────────────────────────────────────

def vector_retrieve_node(state: AgentState) -> dict:
    """Embed all expanded terms in one batch, search Qdrant in parallel."""
    terms = state.get("expanded_terms") or [state["query"]]
    chunks = multi_similarity_search(terms, top_k=settings.top_k_retrieval)
    return {"vector_chunks": chunks}


# ── Node 3b: Live Tools ───────────────────────────────────────────────────────

def live_tools_node(state: AgentState) -> dict:
    """Call external legal APIs for fresh or missing data."""
    query = state["query"]
    results = []
    results.extend(search_courtlistener(query))
    results.extend(search_congress(query))
    results.extend(search_ecfr(query))
    return {"live_results": results}


# ── Node 4: Rerank + Merge ────────────────────────────────────────────────────

def merge_context_node(state: AgentState) -> dict:
    """
    Merge vector chunks and live results.
    Simple strategy: interleave by source, deduplicate, trim to top-K.
    Production: swap in Cohere Rerank API here.
    """
    all_results = (
        state.get("vector_chunks", []) +
        state.get("live_results", [])
    )

    # Deduplicate by citation
    seen_citations = set()
    merged = []
    for r in all_results:
        cit = r.get("citation", r.get("id", ""))
        if cit not in seen_citations:
            merged.append(r)
            seen_citations.add(cit)

    # Trim to rerank_top_k
    merged = merged[: settings.rerank_top_k]
    return {"merged_context": merged}


# ── Node 5: Generate ──────────────────────────────────────────────────────────

def generate_node(state: AgentState) -> dict:
    """
    Call Claude with the assembled legal context and produce a cited answer.
    """
    context_blocks = []
    for i, chunk in enumerate(state.get("merged_context", []), 1):
        text = chunk.get('text', '')[:800]  # trim to 800 chars per chunk
        context_blocks.append(f"[{i}] {chunk.get('citation', 'Unknown')}\n{text}")
    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = GENERATE_PROMPT.format(
        context=context_str,
        query=state["query"],
    )

    response = anthropic.messages.create(
        model=settings.claude_model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text.strip()
    citations = [c.get("citation", "") for c in state.get("merged_context", [])]

    return {"answer": answer, "citations": citations}


# ── Node 6: ReAct Agent (complex queries) ────────────────────────────────────

AGENT_TOOLS = [
    {
        "name": "search_case_law",
        "description": "Search CourtListener for US federal and state court opinions.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_legislation",
        "description": "Search Congress.gov for bills, statutes, and amendments.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_regulations",
        "description": "Search eCFR for current federal regulations.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_vector_db",
        "description": "Search the local vector database of pre-indexed legal documents.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_MAP = {
    "search_case_law": lambda q: search_courtlistener(q),
    "search_legislation": lambda q: search_congress(q),
    "search_regulations": lambda q: search_ecfr(q),
    "search_vector_db": lambda q: similarity_search(q),
}


def react_agent_node(state: AgentState) -> dict:
    """
    ReAct agent loop for complex multi-source legal queries.
    Loops until Claude stops calling tools and produces a final answer.
    """
    messages = list(state.get("messages", []))
    if not messages:
        messages = [{"role": "user", "content": state["query"]}]

    all_live_results = []

    for _ in range(6):  # max 6 tool-call rounds
        response = anthropic.messages.create(
            model=settings.claude_model,
            max_tokens=2000,
            tools=AGENT_TOOLS,
            messages=messages,
            system=(
                "You are a legal research assistant. Use the available tools to "
                "gather relevant legal information before answering. Always cite sources."
            ),
        )

        # No tool calls → final answer
        if response.stop_reason == "end_turn":
            answer = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return {
                "answer": answer,
                "live_results": all_live_results,
                "messages": messages + [{"role": "assistant", "content": answer}],
            }

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = TOOL_MAP.get(block.name)
            if tool_fn:
                results = tool_fn(block.input.get("query", ""))
                all_live_results.extend(results)
                content = str(results[:3]) if results else "No results found."
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })

        serialized = [
            {"type": b.type, "text": b.text} if b.type == "text"
            else {"type": b.type, "id": b.id, "name": b.name, "input": b.input}
            for b in response.content
        ]
        if not tool_results:
            # Claude signalled tool_use but produced no callable tools — treat as final
            answer = next((b.text for b in response.content if hasattr(b, "text")), "")
            return {"answer": answer, "live_results": all_live_results, "messages": messages}

        messages.append({"role": "assistant", "content": serialized})
        messages.append({"role": "user", "content": tool_results})

    # Fallback if max rounds hit
    return {"answer": "Could not resolve query within agent step limit.", "messages": messages}
