"""
Embedder: converts LegalChunkData → vector embeddings → pgvector upsert.
Uses parallel OpenAI calls for maximum throughput.
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from ingestion.db import LegalChunk, SessionLocal
from ingestion.chunkers.legal_chunker import LegalChunkData
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

BATCH_SIZE   = 200  # chunks per OpenAI call
MAX_WORKERS  = 3    # parallel calls in flight
SUBMIT_DELAY = 0.5  # seconds between batch submissions


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API and return list of float vectors."""
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


def _embed_batch(batch_idx: int, batch: list[LegalChunkData]) -> tuple[int, list, list]:
    """Embed one batch with exponential backoff on rate limit errors."""
    for attempt in range(6):
        try:
            vectors = embed_texts([c.text for c in batch])
            return batch_idx, batch, vectors
        except RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit on batch {batch_idx + 1}, retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError(f"Batch {batch_idx + 1} failed after 6 retries")


def upsert_chunks(chunks: list[LegalChunkData], db: Session | None = None) -> int:
    """
    Embed chunks in parallel and upsert into pgvector.
    Returns count of rows inserted.
    """
    close_session = db is None
    if db is None:
        db = SessionLocal()

    batches = [
        (i // BATCH_SIZE, chunks[i : i + BATCH_SIZE])
        for i in range(0, len(chunks), BATCH_SIZE)
    ]
    total_batches = len(batches)
    results: dict[int, tuple[list, list]] = {}

    logger.info(f"Embedding {len(chunks)} chunks in {total_batches} batches "
                f"({MAX_WORKERS} parallel workers)…")

    # Embed in parallel with staggered submissions
    t_embed_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, batch in batches:
            futures[executor.submit(_embed_batch, idx, batch)] = idx
            time.sleep(SUBMIT_DELAY)
        for future in as_completed(futures):
            batch_idx, batch, vectors = future.result()
            results[batch_idx] = (batch, vectors)
            done = len(results)
            logger.info(f"  Embedded batch {batch_idx + 1}/{total_batches} "
                        f"({done}/{total_batches} complete, "
                        f"{done * 100 // total_batches}%)")
    embed_secs = time.time() - t_embed_start
    logger.info(f"All {total_batches} batches embedded in {embed_secs:.1f}s "
                f"({len(chunks) / embed_secs:.0f} chunks/s)")

    # Upsert in order — bulk INSERT ... ON CONFLICT DO UPDATE (one statement per batch)
    logger.info(f"Upserting {len(chunks)} chunks into pgvector…")
    t_upsert_start = time.time()
    inserted = 0
    try:
        for idx in sorted(results):
            batch, vectors = results[idx]
            rows = [
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "section": chunk.section,
                    "jurisdiction": chunk.jurisdiction,
                    "citation": chunk.citation,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "effective_date": chunk.effective_date,
                    "metadata": chunk.metadata,
                    "embedding": vector,
                }
                for chunk, vector in zip(batch, vectors)
            ]
            stmt = pg_insert(LegalChunk.__table__).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={c: stmt.excluded[c] for c in rows[0] if c != "id"},
            )
            db.execute(stmt)
            inserted += len(rows)
        db.commit()
        upsert_secs = time.time() - t_upsert_start
        logger.info(f"Done — upserted {inserted} chunks in {upsert_secs:.1f}s "
                    f"(total: {embed_secs + upsert_secs:.1f}s)")
        return inserted

    except Exception as e:
        db.rollback()
        logger.error(f"Upsert failed: {e}")
        raise
    finally:
        if close_session:
            db.close()


def similarity_search(
    query_text: str,
    top_k: int = 8,
    source_filter: str | None = None,
) -> list[dict]:
    """Embed a query and return top-K similar chunks from pgvector."""
    query_vector = embed_texts([query_text])[0]
    db = SessionLocal()
    try:
        q = db.query(LegalChunk)
        if source_filter:
            q = q.filter(LegalChunk.source == source_filter)
        results = (
            q.order_by(LegalChunk.embedding.cosine_distance(query_vector))
            .limit(top_k)
            .all()
        )
        return [
            {
                "id": r.id,
                "source": r.source,
                "citation": r.citation,
                "section": r.section,
                "title": r.title,
                "text": r.text,
                "score": 1.0,
            }
            for r in results
        ]
    finally:
        db.close()
