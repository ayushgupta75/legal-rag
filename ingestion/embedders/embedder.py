"""
Embedder: converts LegalChunkData → vector embeddings → Qdrant upsert.
Uses sentence-transformers (all-MiniLM-L6-v2) locally — no API calls.
"""
from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from ingestion.chunkers.legal_chunker import LegalChunkData
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_model = SentenceTransformer(settings.embedding_model)
qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

BATCH_SIZE   = 64
MAX_WORKERS  = 1


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
    """Embed texts locally using sentence-transformers."""
    return _model.encode(texts, normalize_embeddings=True).tolist()


def _embed_batch(batch_idx: int, batch: list[LegalChunkData]) -> tuple[int, list, list]:
    """Embed one batch."""
    vectors = embed_texts([c.text for c in batch])
    return batch_idx, batch, vectors


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
    return multi_similarity_search([query_text], top_k=top_k, source_filter=source_filter)


def multi_similarity_search(
    query_texts: list[str],
    top_k: int = 8,
    source_filter: str | None = None,
) -> list[dict]:
    """Embed all queries in one batch, search Qdrant in parallel, deduplicate."""
    vectors = embed_texts(query_texts)

    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    def _search(vector):
        return qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        ).points

    with ThreadPoolExecutor(max_workers=len(vectors)) as executor:
        all_results = list(executor.map(_search, vectors))

    seen_ids = set()
    merged = []
    for points in all_results:
        for r in points:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                merged.append({
                    "id":       r.id,
                    "source":   r.payload.get("source"),
                    "citation": r.payload.get("citation"),
                    "section":  r.payload.get("section"),
                    "title":    r.payload.get("title"),
                    "text":     r.payload.get("text"),
                    "score":    r.score,
                })
    return merged
