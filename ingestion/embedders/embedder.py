"""
Embedder: converts LegalChunkData → vector embeddings → Qdrant upsert.
Uses parallel OpenAI calls for maximum throughput.
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Query,
)
from ingestion.chunkers.legal_chunker import LegalChunkData
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

openai_client = OpenAI(api_key=settings.openai_api_key)
qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

BATCH_SIZE   = 200
MAX_WORKERS  = 3
SUBMIT_DELAY = 0.5


def _ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if settings.qdrant_collection not in existing:
        qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection '{settings.qdrant_collection}'")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API and return list of float vectors."""
    response = openai_client.embeddings.create(
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


def upsert_chunks(chunks: list[LegalChunkData]) -> int:
    """
    Embed chunks in parallel and upsert into Qdrant.
    Returns count of points inserted.
    """
    _ensure_collection()

    batches = [
        (i // BATCH_SIZE, chunks[i : i + BATCH_SIZE])
        for i in range(0, len(chunks), BATCH_SIZE)
    ]
    total_batches = len(batches)
    results: dict[int, tuple[list, list]] = {}

    logger.info(f"Embedding {len(chunks)} chunks in {total_batches} batches "
                f"({MAX_WORKERS} parallel workers)…")

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

    # Upsert into Qdrant in order
    logger.info(f"Upserting {len(chunks)} chunks into Qdrant…")
    t_upsert_start = time.time()
    inserted = 0

    for idx in sorted(results):
        batch, vectors = results[idx]
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "source":         chunk.source,
                    "title":          chunk.title,
                    "section":        chunk.section,
                    "jurisdiction":   chunk.jurisdiction,
                    "citation":       chunk.citation,
                    "text":           chunk.text,
                    "char_count":     chunk.char_count,
                    "effective_date": chunk.effective_date,
                    "parent_id":      chunk.parent_id,
                    "metadata":       chunk.metadata,
                },
            )
            for chunk, vector in zip(batch, vectors)
        ]
        qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
        inserted += len(points)

    upsert_secs = time.time() - t_upsert_start
    logger.info(f"Done — upserted {inserted} chunks in {upsert_secs:.1f}s "
                f"(total: {embed_secs + upsert_secs:.1f}s)")
    return inserted


def similarity_search(
    query_text: str,
    top_k: int = 8,
    source_filter: str | None = None,
) -> list[dict]:
    """Embed a query and return top-K similar chunks from Qdrant."""
    query_vector = embed_texts([query_text])[0]

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    response = qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    return [
        {
            "id":       r.id,
            "source":   r.payload.get("source"),
            "citation": r.payload.get("citation"),
            "section":  r.payload.get("section"),
            "title":    r.payload.get("title"),
            "text":     r.payload.get("text"),
            "score":    r.score,
        }
        for r in response.points
    ]
