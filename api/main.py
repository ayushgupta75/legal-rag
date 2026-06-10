"""
FastAPI application — exposes the LangGraph agent as a REST API.
"""
import uuid
import time
import os
import logging
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.graph import get_graph

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="US Legal RAG API",
    description="Agentic RAG over US law — Constitution, US Code, CFR, and live case law.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = get_graph()

# ── API Key security ──────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# ── Routes ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    route: str
    thread_id: str
    latency_ms: int


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_legal(req: QueryRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    start = time.time()
    try:
        result = graph.invoke(
            {
                "query": req.query,
                "thread_id": thread_id,
                "messages": [],
                "vector_chunks": [],
                "live_results": [],
                "merged_context": [],
                "expanded_terms": [],
            },
            config=config,
        )
        latency_ms = int((time.time() - start) * 1000)
        return QueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            route=result.get("route", "unknown"),
            thread_id=thread_id,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
