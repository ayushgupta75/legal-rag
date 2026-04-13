# US Legal RAG System

Agentic RAG over US law — Constitution, US Code, CFR, and live case law.

## Stack
- **Orchestration**: LangGraph (agentic graph + ReAct agent)
- **LLM**: Claude (Anthropic API)
- **Vector DB**: pgvector on AWS RDS Postgres
- **Embeddings**: text-embedding-3-small (OpenAI) or Cohere
- **Live tools**: CourtListener API, Congress.gov API, eCFR API
- **API**: FastAPI
- **Infra**: AWS (ECS Fargate + RDS + ElastiCache Redis)

## Project structure
```
legal-rag/
├── ingestion/          # Data pipeline: download → parse → chunk → embed → store
│   ├── parsers/        # Per-source parsers (constitution, uscode, cfr, caselaw)
│   ├── chunkers/       # Legal-aware chunking strategies
│   └── embedders/      # Embedding + pgvector upsert
├── agent/              # LangGraph graph definition
│   ├── nodes/          # Individual graph nodes (router, retrieve, generate…)
│   ├── tools/          # Live API tools (CourtListener, Congress.gov, eCFR)
│   └── prompts/        # Claude prompt templates
├── api/                # FastAPI app
├── ui/                 # Minimal chat frontend
├── infra/              # AWS CDK / Terraform
├── tests/
└── scripts/            # One-off scripts (ingest, backfill, eval)
```

## Quickstart
```bash
cp .env.example .env        # fill in your keys
docker compose up -d        # starts Postgres + Redis locally
python scripts/ingest.py    # runs the full ingestion pipeline
uvicorn api.main:app --reload
```
