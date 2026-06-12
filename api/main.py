"""
FastAPI application — exposes the LangGraph agent as a REST API.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from agent.graph import get_graph
from config import get_settings

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# ── Query log (separate from uvicorn/app logs) ────────────────────────────────

os.makedirs("logs", exist_ok=True)
query_logger = logging.getLogger("query_log")
query_logger.setLevel(logging.INFO)
_log_handler = RotatingFileHandler("logs/queries.log", maxBytes=5_000_000, backupCount=3)
_log_handler.setFormatter(logging.Formatter("%(message)s"))
query_logger.addHandler(_log_handler)
query_logger.propagate = False  # don't echo into uvicorn's stdout

# Pricing per million tokens (input, output) — keep in sync with config.py / .env
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

settings = get_settings()
graph = get_graph()

# ── API Key security ──────────────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Routes ────────────────────────────────────────────────────────────────────

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
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            },
            config=config,
        )
        latency_ms = int((time.time() - start) * 1000)

        input_tok = result.get("total_input_tokens", 0)
        output_tok = result.get("total_output_tokens", 0)
        in_price, out_price = _MODEL_PRICING.get(settings.claude_model, (3.00, 15.00))
        cost_usd = (input_tok * in_price + output_tok * out_price) / 1_000_000

        query_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": req.query,
            "route": result.get("route", "unknown"),
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": latency_ms,
        }))

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
