# Constitution Compass — US Legal RAG

> Agentic retrieval-augmented generation over US law: Constitution, US Code (all 54 titles), CFR regulations, and live case law.

[![CI](https://github.com/your-org/legal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/legal-rag/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Constitution Compass is a production-grade RAG system that routes legal queries through three distinct paths depending on complexity:

- **Vector path** — fast cosine similarity search over a pre-indexed pgvector store (Constitution + full US Code)
- **Live tools path** — real-time calls to CourtListener, Congress.gov, and eCFR APIs for recent case law and regulations
- **ReAct agent path** — autonomous multi-step reasoning loop for complex, multi-source legal questions

All paths converge on a Claude-powered generation step that produces cited, structured answers.

---

## Architecture

```
User query
    │
    ▼
query_analysis          ← classifies query type, expands legal terminology
    │
    ▼
router
    ├─── simple     ──→ vector_retrieve ──────────────┐
    ├─── multi_source→ live_tools ────────────────────┤
    │                                                  ▼
    │                                           merge_context   ← dedup + rerank
    │                                                  │
    │                                                  ▼
    │                                             generate      ← Claude w/ citations
    │                                                  │
    └─── complex    ──→ react_agent ─────────────────END
                        (self-contained ReAct loop,
                         up to 6 tool-call rounds)
```

Built with [LangGraph](https://github.com/langchain-ai/langgraph). All nodes share a single `AgentState` TypedDict; conversation continuity is provided by a thread-id-scoped checkpointer.

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Claude (`claude-sonnet-4-20250514`) via Anthropic API |
| Vector store | pgvector on PostgreSQL 16 |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
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
│   ├── graph.py            # LangGraph graph definition and wiring
│   ├── state.py            # AgentState TypedDict
│   ├── nodes/
│   │   └── nodes.py        # All node functions (analysis, router, retrieve, generate, ReAct)
│   ├── tools/
│   │   └── live_tools.py   # CourtListener / Congress.gov / eCFR HTTP wrappers
│   └── prompts/
│       └── legal_prompts.py
├── ingestion/
│   ├── db.py               # SQLAlchemy models (LegalChunk, QueryLog) + schema init
│   ├── chunkers/
│   │   └── legal_chunker.py  # Recursive structural chunker (article → section → paragraph)
│   ├── embedders/
│   │   └── embedder.py     # Parallelized OpenAI embedding + pgvector upsert
│   └── parsers/
│       ├── uscode_parser.py
│       └── cfr_parser.py
├── api/
│   └── main.py             # FastAPI app — POST /query, GET /health
├── ui/
│   └── index.html          # Minimal chat frontend
├── infra/
│   └── aws_stack.py        # AWS CDK stack
├── scripts/
│   └── ingest_full.py      # One-off ingestion runner
├── tests/
├── data/
│   ├── constitution.txt
│   └── pdf_uscAll@119-73/  # 54 US Code title PDFs (119th Congress)
├── config.py               # Pydantic settings (reads from .env)
├── docker-compose.yml
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
# Edit .env — at minimum set ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. Start infrastructure

```bash
docker compose up -d postgres redis
```

### 3. Install dependencies and initialise the database

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ingestion.db          # creates pgvector extension + tables
```

### 4. Run ingestion

```bash
python scripts/ingest_full.py   # downloads, parses, chunks, embeds, and upserts all sources
```

This embeds documents in parallel (3 workers, 200-chunk batches) and upserts into pgvector with `ON CONFLICT DO UPDATE`.

### 5. Start the API

```bash
uvicorn api.main:app --reload
# API is available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
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
  "thread_id": "string | null"   // pass the same thread_id to continue a conversation
}
```

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

Returns `{"status": "ok"}`.

---

## Configuration

All configuration is managed via environment variables (`.env` file or shell). See [config.py](config.py) for the full list.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `OPENAI_API_KEY` | — | **Required.** Used for embeddings |
| `POSTGRES_HOST` | `localhost` | Postgres host |
| `POSTGRES_PORT` | `5432` | Postgres port |
| `POSTGRES_DB` | `legal_rag` | Database name |
| `POSTGRES_USER` | `postgres` | |
| `POSTGRES_PASSWORD` | `postgres` | |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL for LangGraph checkpointer |
| `COURTLISTENER_API_KEY` | `""` | Optional; increases rate limits |
| `CONGRESS_API_KEY` | `""` | Optional; defaults to `DEMO_KEY` |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Anthropic model ID |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K_RETRIEVAL` | `8` | Chunks retrieved per search term |
| `RERANK_TOP_K` | `4` | Chunks passed to the generate node after merging |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |

---

## Data Sources

| Source | Format | Coverage |
|---|---|---|
| US Constitution | Plain text | Full text with amendments |
| US Code | PDF (54 titles) | 119th Congress, as of Supplement 73 |
| CFR | eCFR API (live) | All titles, current version |
| Case law | CourtListener API (live) | Federal + state court opinions |
| Legislation | Congress.gov API (live) | Bills, statutes, amendments |

---

## Ingestion Pipeline

```
PDF / text / API
      │
      ▼
  Parser          ← extracts raw text per source type
      │
      ▼
  Legal chunker   ← recursive structural split (article → section → paragraph → sentence)
      │             max 2000 chars per chunk, 200-char overlap, parent_id for hierarchy
      ▼
  Embedder        ← OpenAI text-embedding-3-small, 3 parallel workers, 200-chunk batches
      │             exponential backoff on rate limit errors
      ▼
  pgvector        ← bulk INSERT … ON CONFLICT DO UPDATE
```

---

## Development

### Running tests

```bash
pytest tests/ -v
```

Tests require a running Postgres + Redis instance (provided by `docker compose up -d postgres redis`). The CI workflow spins these up as GitHub Actions service containers.

### Linting

```bash
ruff check .
```

### Running the full stack locally

```bash
docker compose up          # starts postgres, redis, and the API container
```

---

## Deployment

The CI/CD pipeline (`.github/workflows/ci.yml`) runs on every push to `main`:

1. Runs the test suite against live Postgres + Redis service containers
2. Builds and pushes a Docker image to Amazon ECR
3. Forces a new ECS Fargate deployment (`aws ecs update-service --force-new-deployment`)

AWS infrastructure is defined in [infra/aws_stack.py](infra/aws_stack.py) using AWS CDK (ECS Fargate + RDS + ElastiCache).

---

## Roadmap

- [ ] Cohere Rerank API integration in `merge_context_node`
- [ ] Streaming responses via SSE
- [ ] State-level law support
- [ ] HNSW index auto-rebuild after ingestion
- [ ] Evaluation harness (RAGAS)

---

## License

MIT
