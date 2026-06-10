"""
RAG_v2/benchmark.py

Compares v1 (custom pipeline) vs v2 (LlamaIndex) across three dimensions:
  1. Chunk statistics  — count, avg size, size distribution
  2. Ingestion speed   — time to ingest the constitution from scratch
  3. Retrieval quality — top-K results for the same query from each table

Run:
    python RAG_v2/benchmark.py
    python RAG_v2/benchmark.py --query "What does the Fourth Amendment protect?"
    python RAG_v2/benchmark.py --skip-ingestion   # stats + retrieval only
"""
import sys
import time
import argparse
import logging
sys.path.insert(0, ".")

from sqlalchemy import text
from openai import OpenAI
from ingestion.db import SessionLocal, LegalChunk, engine, init_db
from ingestion.chunkers.legal_chunker import chunk_constitution
from ingestion.embedders.embedder import upsert_chunks
from RAG_v2.main import ingest_constitution as v2_ingest, get_vector_store, DB_TABLE_NAME as TABLE_NAME
from config import get_settings

logging.basicConfig(level=logging.WARNING)   # suppress INFO noise during benchmark
logger = logging.getLogger(__name__)
settings = get_settings()
openai_client = OpenAI(api_key=settings.openai_api_key)

CONSTITUTION_FILE = "data/constitution.txt"
SEPARATOR = "─" * 60


# ── 1. Chunk statistics ───────────────────────────────────────────────────────

def chunk_stats():
    print(f"\n{SEPARATOR}")
    print("  CHUNK STATISTICS")
    print(SEPARATOR)

    db = SessionLocal()
    try:
        v1_chunks = db.query(LegalChunk).filter(LegalChunk.source == "constitution").all()
        v1_sizes = [c.char_count for c in v1_chunks if c.char_count]
    finally:
        db.close()

    with engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT length(text) FROM {TABLE_NAME}"
        )).fetchall()
    v2_sizes = [r[0] for r in rows]

    def stats(sizes: list[int], label: str):
        if not sizes:
            print(f"  {label}: no data")
            return
        avg = sum(sizes) / len(sizes)
        print(f"  {label}")
        print(f"    Chunks      : {len(sizes)}")
        print(f"    Avg size    : {avg:.0f} chars")
        print(f"    Min / Max   : {min(sizes)} / {max(sizes)} chars")
        under_500  = sum(1 for s in sizes if s < 500)
        btw_500_2k = sum(1 for s in sizes if 500 <= s < 2000)
        over_2k    = sum(1 for s in sizes if s >= 2000)
        print(f"    <500 chars  : {under_500}  ({under_500*100//len(sizes)}%)")
        print(f"    500–2k chars: {btw_500_2k}  ({btw_500_2k*100//len(sizes)}%)")
        print(f"    >2k chars   : {over_2k}  ({over_2k*100//len(sizes)}%)")

    stats(v1_sizes, "v1  (legal chunker → legal_chunks)")
    print()
    stats(v2_sizes, "v2  (SentenceSplitter → llamaindex_chunks)")


# ── 2. Ingestion speed ────────────────────────────────────────────────────────

def ingestion_speed():
    print(f"\n{SEPARATOR}")
    print("  INGESTION SPEED  (constitution only, no HNSW)")
    print(SEPARATOR)

    with open(CONSTITUTION_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    # v1
    print("  Running v1 ingestion…")
    t0 = time.time()
    chunks = list(chunk_constitution(raw))
    upsert_chunks(chunks)
    v1_secs = time.time() - t0
    print(f"  v1 done: {v1_secs:.1f}s  ({len(chunks)} chunks)")

    # v2
    print("  Running v2 ingestion…")
    t0 = time.time()
    v2_ingest()
    v2_secs = time.time() - t0
    print(f"  v2 done: {v2_secs:.1f}s")

    print(f"\n  Difference: {'v1' if v1_secs < v2_secs else 'v2'} was "
          f"{abs(v1_secs - v2_secs):.1f}s faster")


# ── 3. Retrieval quality ──────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=settings.embedding_model,
        input=[query],
        encoding_format="float",
    )
    return resp.data[0].embedding


def retrieval_quality(query: str, top_k: int = 3):
    print(f"\n{SEPARATOR}")
    print("  RETRIEVAL QUALITY")
    print(f"  Query: \"{query}\"")
    print(SEPARATOR)

    vector = embed_query(query)

    # v1 — pgvector cosine distance via SQLAlchemy
    db = SessionLocal()
    try:
        t0 = time.time()
        v1_results = (
            db.query(LegalChunk)
            .filter(LegalChunk.source == "constitution")
            .order_by(LegalChunk.embedding.cosine_distance(vector))
            .limit(top_k)
            .all()
        )
        v1_latency = (time.time() - t0) * 1000
    finally:
        db.close()

    # v2 — LlamaIndex VectorStore retriever
    from llama_index.core.vector_stores import VectorStoreQuery

    vector_store = get_vector_store()
    t0 = time.time()
    query_result = vector_store.query(
        VectorStoreQuery(query_embedding=vector, similarity_top_k=top_k)
    )
    v2_latency = (time.time() - t0) * 1000
    v2_nodes = query_result.nodes or []

    # Print v1 results
    print(f"\n  v1  (legal_chunks)  — {v1_latency:.0f}ms")
    for i, r in enumerate(v1_results, 1):
        snippet = r.text[:120].replace("\n", " ")
        print(f"  [{i}] {r.citation}")
        print(f"      {snippet}…")

    # Print v2 results
    print(f"\n  v2  (llamaindex_chunks)  — {v2_latency:.0f}ms")
    for i, node in enumerate(v2_nodes, 1):
        snippet = node.text[:120].replace("\n", " ")
        citation = node.metadata.get("citation_prefix", "U.S. Const.")
        print(f"  [{i}] {citation}")
        print(f"      {snippet}…")

    print(f"\n  Retrieval latency: v1={v1_latency:.0f}ms  v2={v2_latency:.0f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What does the Fourth Amendment protect?")
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion speed test — only run stats and retrieval",
    )
    args = parser.parse_args()

    init_db()

    chunk_stats()

    if not args.skip_ingestion:
        ingestion_speed()

    retrieval_quality(args.query)

    print(f"\n{SEPARATOR}\n")


if __name__ == "__main__":
    main()
