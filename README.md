# Constitution Compass — US Legal RAG

> Agentic retrieval-augmented generation over US law: Constitution, US Code (all 54 titles), CFR regulations, and live case law.

[![CI](https://github.com/your-org/legal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/legal-rag/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-20%2F20-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Constitution Compass is a production-grade agentic RAG system that answers questions over US federal law. It classifies every incoming query and routes it through one of three execution paths — fast vector retrieval, live legal API calls, or an autonomous multi-step reasoning loop — before generating a cited answer with Claude.

**Benchmark results vs LlamaIndex standard pipeline:**
- **3x more relevant** retrieval results on constitutional law queries
- **10.8ms** average query latency (Qdrant HNSW, 121-chunk constitution corpus)
- **73% faster** than LlamaIndex's `SentenceSplitter` pipeline (10.8ms vs 44ms)

---

## Architecture

```
User query
    │
    ▼
query_analysis          ← classifies query type, expands legal terminology (Claude)
    │
    ▼
router
    ├─── simple      ──→ vector_retrieve ──────────────┐
    ├─── multi_source ──→ live_tools ──────────────────┤
    │                                                   ▼
    │                                           merge_context   ← dedup + trim
    │                                                   │
    │                                                   ▼
    │                                              generate     ← Claude w/ citations
    │                                                   │
    └─── complex     ──→ react_agent ─────────────────END
                         (ReAct loop, up to 6 tool-call rounds)
```

Built with [LangGraph](https://github.com/langchain-ai/langgraph). All nodes share a single `AgentState` TypedDict. Conversation continuity is provided by a thread-id-scoped Redis checkpointer.

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Claude (`claude-sonnet-4-5`) via Anthropic API |
| Vector store | Qdrant (HNSW, auto-indexed on upsert) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Metadata store | PostgreSQL 16 |
| Live legal APIs | CourtListener, Congress.gov, eCFR |
| API server | FastAPI + Uvicorn |
| State cache | Redis (LangGraph checkpointer) |
| Infra | AWS ECS Fargate + RDS + ElastiCache (CDK) |
| CI/CD | GitHub Actions → ECR → ECS |

---

## Project Structure

```
legal-rag/
├── agent/
│   ├── graph.py              # LangGraph graph — nodes, edges, checkpointer
│   ├── state.py              # AgentState TypedDict
│   ├── nodes/
│   │   └── nodes.py          # query_analysis, router, vector_retrieve, live_tools,
│   │                         # merge_context, generate, react_agent
│   ├── tools/
│   │   └── live_tools.py     # CourtListener / Congress.gov / eCFR wrappers
│   └── prompts/
│       └── legal_prompts.py
├── ingestion/
│   ├── db.py                 # SQLAlchemy models (QueryLog) + schema init
│   ├── chunkers/
│   │   └── legal_chunker.py  # Recursive structural chunker — Article → Section → Clause
│   ├── embedders/
│   │   └── embedder.py       # Parallelized OpenAI embedding + Qdrant upsert
│   └── parsers/
│       ├── uscode_parser.py
│       └── cfr_parser.py
├── RAG_v2/
│   ├── main.py               # LlamaIndex ingestion pipeline (benchmark variant)
│   └── benchmark.py          # v1 vs v2 retrieval quality + latency comparison
├── api/
│   └── main.py               # FastAPI — POST /query, GET /health
├── ui/
│   └── index.html            # Chat frontend
├── infra/
│   └── aws_stack.py          # AWS CDK stack (ECS + RDS + ElastiCache)
├── scripts/
│   └── ingest_full.py        # Ingestion runner — constitution, uscode, cfr
├── tests/
│   ├── test_chunker.py       # Legal chunker unit tests (11 tests)
│   └── test_graph_routing.py # LangGraph routing + merge tests (9 tests)
├── data/
│   ├── constitution.txt
│   └── pdf_uscAll@119-73/    # 54 US Code title PDFs (119th Congress)
├── config.py                 # Pydantic settings (reads from .env)
├── docker-compose.yml        # Postgres + Redis + Qdrant
├── Dockerfile
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- Docker + Docker Compose
- Anthropic API key
- OpenAI API key (embeddings)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/legal-rag.git
cd legal-rag
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and OPENAI_API_KEY at minimum
```

### 2. Start infrastructure

```bash
docker compose up -d postgres redis qdrant
```

This starts:
- **PostgreSQL 16** on `localhost:5432` — metadata + query logs
- **Redis 7** on `localhost:6379` — LangGraph conversation checkpointer
- **Qdrant** on `localhost:6333` — vector store (HNSW auto-managed)

### 3. Install dependencies and initialise the database

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ingestion.db          # creates Postgres tables
```

### 4. Run ingestion

```bash
# Constitution only (fast, ~3 seconds)
python scripts/ingest_full.py --source constitution

# Specific US Code titles
python scripts/ingest_full.py --source uscode --titles 18 42

# Everything
python scripts/ingest_full.py --source all
```

Qdrant indexes vectors automatically on every upsert — no separate index build step needed.

### 5. Start the API

```bash
uvicorn api.main:app --reload
# http://localhost:8000
# http://localhost:8000/docs  ← interactive API docs
```

### 6. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the Fourth Amendment protect against?"}'
```

---

## API Reference

### `POST /query`

Run a legal query through the agent graph.

**Request**

```json
{
  "query": "string",
  "thread_id": "string | null"
}
```

Pass the same `thread_id` on follow-up requests to continue a multi-turn conversation.

**Response**

```json
{
  "answer": "string",
  "citations": ["string"],
  "route": "vector | live_tools | agent | unknown",
  "thread_id": "string",
  "latency_ms": 0
}
```

### `GET /health`

```json
{"status": "ok"}
```

---

## Configuration

All settings are read from environment variables or a `.env` file. See [config.py](config.py) for the full schema.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `OPENAI_API_KEY` | — | **Required.** Used for embeddings |
| `POSTGRES_HOST` | `localhost` | |
| `POSTGRES_PORT` | `5432` | |
| `POSTGRES_DB` | `legal_rag` | |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_PASSWORD` | `postgres` | |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant gRPC/HTTP port |
| `QDRANT_COLLECTION` | `legal_chunks` | Collection name |
| `REDIS_URL` | `redis://localhost:6379` | |
| `COURTLISTENER_API_KEY` | `""` | Optional — increases rate limits |
| `CONGRESS_API_KEY` | `""` | Optional — defaults to `DEMO_KEY` |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Anthropic model ID |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding dimensions |
| `TOP_K_RETRIEVAL` | `8` | Chunks retrieved per search term |
| `RERANK_TOP_K` | `4` | Chunks passed to generate node after merging |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |

---

## Data Sources

| Source | Format | Coverage |
|---|---|---|
| US Constitution | Plain text | Full text + all 27 amendments |
| US Code | PDF (54 titles) | 119th Congress, Supplement 73 |
| CFR | eCFR API (live) | All 50 titles, current version |
| Case law | CourtListener API (live) | Federal + state court opinions |
| Legislation | Congress.gov API (live) | Bills, statutes, amendments |

---

## Ingestion Pipeline

```
PDF / plain text / API response
          │
          ▼
      Parser              ← extracts clean text per source type
          │
          ▼
      Legal chunker        ← recursive structural split
          │                   Article → Section → Paragraph → Sentence
          │                   max 2,000 chars, 200-char overlap at paragraph level
          │                   parent_id preserved for hierarchy-aware retrieval
          ▼
      Embedder             ← OpenAI text-embedding-3-small
          │                   3 parallel workers, 200-chunk batches
          │                   exponential backoff on rate limits
          ▼
      Qdrant               ← upsert with HNSW auto-indexing
                              cosine distance, 1536-dim
```

---

## Benchmark: v1 (Legal Chunker) vs v2 (LlamaIndex)

`RAG_v2/` contains a parallel ingestion pipeline built with LlamaIndex's `IngestionPipeline` + `SentenceSplitter` as a benchmark baseline. Results on the US Constitution corpus:

| Metric | v1 — Legal chunker + Qdrant | v2 — LlamaIndex SentenceSplitter |
|---|---|---|
| Chunks produced | 121 | 23 |
| Avg chunk size | 496 chars | 1,980 chars |
| Retrieval latency | **10.8ms** | 44ms |
| Relevant results (top-3) | **3 / 3** | 0 / 3 |

**Why v2 fails:** `SentenceSplitter` packs 4–5 amendments per chunk. Each chunk's embedding vector becomes a centroid of multiple unrelated legal concepts, making targeted retrieval impossible. v1's chunker preserves amendment boundaries — each amendment is its own isolated vector.

Run the benchmark:

```bash
python RAG_v2/benchmark.py --skip-ingestion
python RAG_v2/benchmark.py --query "What is the First Amendment?"
```

---

## Testing

```bash
pytest tests/ -v
```

20 tests, zero external dependencies (no API calls, no DB):

| Suite | Tests | Coverage |
|---|---|---|
| `test_chunker.py` | 11 | Chunker output, citation format, `version_hash` determinism, field validation |
| `test_graph_routing.py` | 9 | Router logic, edge cases, merge deduplication, `rerank_top_k` trimming |

---

## Deployment

The CI/CD pipeline (`.github/workflows/ci.yml`) runs on every push to `main`:

1. Spins up Postgres + Redis as GitHub Actions service containers
2. Runs the full test suite
3. Builds and pushes a Docker image to Amazon ECR
4. Forces a rolling ECS Fargate redeployment

AWS infrastructure is defined in [infra/aws_stack.py](infra/aws_stack.py) using AWS CDK — ECS Fargate, RDS Postgres, ElastiCache Redis. Qdrant runs as a sidecar ECS container with an EFS volume for persistence.

### Migrating vector data to AWS

```bash
# Option A — re-ingest against RDS + Qdrant on AWS (simplest)
POSTGRES_HOST=<rds-endpoint> QDRANT_HOST=<qdrant-endpoint> \
  python scripts/ingest_full.py --source all

# Option B — dump and restore Postgres metadata only
pg_dump -h localhost -U postgres -d legal_rag -Fc -f legal_rag.dump
pg_restore -h <rds-endpoint> -U postgres -d legal_rag legal_rag.dump
# Then snapshot and restore the Qdrant EFS volume for vector data
```

---

## Roadmap

- [ ] Cohere Rerank API in `merge_context_node`
- [ ] Streaming responses via SSE
- [ ] Hybrid search — BM25 + dense vector (Qdrant native)
- [ ] State-level law support
- [ ] Evaluation harness (RAGAS)
- [ ] HNSW parameter tuning (`m`, `ef`) at production scale

---

## License

MIT
