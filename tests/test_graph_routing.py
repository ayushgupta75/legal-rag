"""
tests/test_graph_routing.py — Tests for LangGraph node logic (no real API calls).
Uses monkeypatching to avoid hitting Anthropic / OpenAI / pgvector.
"""
import pytest
import sys
sys.path.insert(0, ".")

from agent.nodes.nodes import router_node, get_route
from agent.state import AgentState


def make_state(**kwargs) -> AgentState:
    defaults = {
        "query": "What does the First Amendment say?",
        "query_type": "simple",
        "expanded_terms": ["First Amendment", "freedom of speech"],
        "route": "vector",
        "vector_chunks": [],
        "live_results": [],
        "merged_context": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "thread_id": "test-thread",
        "latency_ms": 0,
    }
    defaults.update(kwargs)
    return AgentState(**defaults)


class TestRouter:
    def test_simple_routes_to_vector(self):
        state = make_state(query_type="simple")
        result = router_node(state)
        assert result["route"] == "vector"

    def test_complex_routes_to_agent(self):
        state = make_state(query_type="complex")
        result = router_node(state)
        assert result["route"] == "agent"

    def test_multi_source_routes_to_live_tools(self):
        state = make_state(query_type="multi_source")
        result = router_node(state)
        assert result["route"] == "live_tools"

    def test_unknown_type_defaults_to_vector(self):
        state = make_state(query_type="unknown_type")
        result = router_node(state)
        assert result["route"] == "vector"

    def test_get_route_reads_state(self):
        state = make_state(route="agent")
        assert get_route(state) == "agent"

    def test_get_route_defaults_to_vector(self):
        state = make_state()
        state.pop("route", None)  # remove route key
        assert get_route(state) == "vector"


class TestMergeContext:
    def test_deduplicates_by_citation(self):
        from agent.nodes.nodes import merge_context_node
        chunk = {"id": "1", "citation": "18 U.S.C. § 1030", "text": "abc"}
        state = make_state(
            vector_chunks=[chunk, chunk],  # duplicate
            live_results=[],
        )
        result = merge_context_node(state)
        assert len(result["merged_context"]) == 1

    def test_merges_vector_and_live(self):
        from agent.nodes.nodes import merge_context_node
        state = make_state(
            vector_chunks=[{"id": "1", "citation": "Const. Art. I", "text": "aaa"}],
            live_results=[{"id": "2", "citation": "5 U.S.C. § 552", "text": "bbb"}],
        )
        result = merge_context_node(state)
        assert len(result["merged_context"]) == 2

    def test_trims_to_rerank_top_k(self, monkeypatch):
        from agent.nodes.nodes import merge_context_node
        import agent.nodes.nodes as nodes_module
        monkeypatch.setattr(nodes_module.settings, "rerank_top_k", 2)
        chunks = [
            {"id": str(i), "citation": f"Citation {i}", "text": f"text {i}"}
            for i in range(10)
        ]
        state = make_state(vector_chunks=chunks, live_results=[])
        result = merge_context_node(state)
        assert len(result["merged_context"]) == 2
