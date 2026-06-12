# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Infrastructure (Redis + Qdrant only — no Postgres, see note below)
docker compose up -d redis qdrant

# Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start API
uvicorn api.main:app --reload

# Run all tests (no external deps — no API calls, no live DB)
pytest tests/ -v

# Run a single test file
pytest tests/test_chunker.py -v

# Lint
ruff check .

# Ingest data
python scripts/ingest_full.py --source constitution
python scripts/ingest_full.py --source uscode --titles 18 42
python scripts/ingest_full.py --source all
```

> **Note on Postgres:** `docker-compose.yml` doesn't define a Postgres service. `ingestion/db.py` defines a `QueryLog` SQLAlchemy model but it's not wired into the live query path — only init schema runs need it. Run `python -m ingestion.db` to create tables if using a local Postgres.

## Architecture

The system is a LangGraph agentic RAG pipeline that classifies incoming legal queries and routes them through one of three execution paths before generating a Claude-cited answer.

### Request flow

```
POST /query (api/main.py)
    └── graph.invoke() (agent/graph.py)
            ├── query_analysis_node  — Claude classifies query as simple/complex/multi_source,
            │                          expands legal search terms
            ├── router_node          — maps query_type → route string
            │
            ├─[vector]──→ vector_retrieve_node  — multi-query embed + Qdrant HNSW search
            │                 └── merge_context_node  — dedup by citation, trim to rerank_top_k
            │                         └── generate_node  — Claude with assembled context → cited answer
            │
            ├─[live_tools]──→ live_tools_node  — CourtListener + Congress.gov + eCFR APIs
            │                     └── merge_context_node → generate_node (same path as above)
            │
            └─[agent]──→ react_agent_node  — autonomous ReAct loop (up to 6 rounds),
                                             handles its own tool calls and generation inline
```

All nodes share a single `AgentState` TypedDict (`agent/state.py`). The graph is compiled in `agent/graph.py` with a `MemorySaver` checkpointer (in-memory, not Redis) for conversation continuity scoped by `thread_id`. Passing the same `thread_id` across multiple `POST /query` calls enables multi-turn conversations.

### Key design decisions

- **Embeddings are local** (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) — no OpenAI API key needed.
- **LLM calls use the raw Anthropic SDK** (`anthropic.Anthropic`), not LangChain's model wrappers. `nodes.py` calls `anthropic.messages.create` directly.
- **Token tracking**: `AgentState` accumulates `total_input_tokens` and `total_output_tokens` across all Claude calls via `operator.add`. `api/main.py` uses these to compute and log per-query USD cost.
- **Query cost logging**: every request is logged as JSON to `logs/queries.log` (rotating, 5 MB limit) with route, token counts, cost, and latency. The log is separate from uvicorn stdout.
- **Chunker is structural, not semantic** — `legal_chunker.py` splits recursively on legal boundaries (ARTICLE/AMENDMENT/TITLE → Section → paragraph → sentence) down to 2000-char max with 200-char overlap, preserving parent→child hierarchy via `parent_id`.
- **ReAct agent** in `react_agent_node` uses the Anthropic tool-use API directly (not LangGraph's built-in agent) with a manual 6-round loop.
- **`merge_context_node`** deduplicates by citation string. The `rerank_top_k` config trims the final context window. The README mentions Cohere Rerank as a planned replacement.

### Module map

| Path | Responsibility |
|---|---|
| `agent/graph.py` | Graph wiring — nodes, edges, checkpointer |
| `agent/state.py` | `AgentState` TypedDict (shared by all nodes) |
| `agent/nodes/nodes.py` | All node functions + ReAct loop |
| `agent/tools/live_tools.py` | CourtListener / Congress.gov / eCFR wrappers |
| `agent/prompts/legal_prompts.py` | `QUERY_ANALYSIS_PROMPT`, `GENERATE_PROMPT` |
| `ingestion/chunkers/legal_chunker.py` | Structural chunker → `LegalChunkData` |
| `ingestion/embedders/embedder.py` | sentence-transformers embed + Qdrant upsert/search |
| `ingestion/parsers/` | US Code (PDF) and CFR (API) parsers |
| `api/main.py` | FastAPI app, `POST /query`, `GET /health`, API key auth, cost logging |
| `config.py` | Pydantic settings loaded from `.env` |
| `scripts/ingest_full.py` | CLI runner for ingestion |
| `RAG_v2/` | LlamaIndex benchmark variant — not part of the live system |

## Configuration

All settings in `config.py` are read from `.env`. Required: `ANTHROPIC_API_KEY`. Key runtime settings:

| Variable | Default | Notes |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Used by all LLM calls |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `EMBEDDING_DIM` | `384` | Must match the model |
| `TOP_K_RETRIEVAL` | `8` | Chunks fetched from Qdrant per search term |
| `RERANK_TOP_K` | `4` | Final context window passed to generate node |
| `REDIS_URL` | `redis://localhost:6379` | Used by docker-compose service wiring |
| `QDRANT_HOST` | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `legal_chunks` | Collection name in Qdrant |
| `API_KEY` | `""` | If set, `X-API-Key` header is required on `POST /query` |
| `COURTLISTENER_API_KEY` | `""` | Optional — live_tools path degrades gracefully without it |
| `CONGRESS_API_KEY` | `""` | Optional — live_tools path degrades gracefully without it |
| `LANGCHAIN_TRACING_V2` | `false` | Set to `true` + `LANGCHAIN_API_KEY` to enable LangSmith traces |

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main`:
1. Spins up Redis as a service container
2. Runs `ruff check .` then `pytest tests/ -v`
3. On merge to `main`: SSHes into EC2, pulls, reinstalls deps, restarts uvicorn

Deployment is to an EC2 instance (not ECS). `infra/aws_stack.py` defines a CDK ECS stack that is not currently active.

## Tests

Tests in `tests/` make no network calls and require no running services. `test_graph_routing.py` uses `monkeypatch` to override settings values (e.g. `rerank_top_k`). The pattern for testing nodes is to construct an `AgentState` via `make_state(**overrides)` and call the node function directly.

`AgentState` includes token-tracking fields that must be initialized in `make_state`:

```python
def make_state(**kwargs) -> AgentState:
    defaults = {
        "query": "...", "query_type": "simple", "expanded_terms": [],
        "route": "vector", "vector_chunks": [], "live_results": [],
        "merged_context": [], "messages": [], "answer": "", "citations": [],
        "thread_id": "test-thread", "latency_ms": 0,
        "total_input_tokens": 0, "total_output_tokens": 0,
    }
    defaults.update(kwargs)
    return AgentState(**defaults)
```
