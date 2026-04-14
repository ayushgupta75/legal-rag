"""
Embedder: converts LegalChunkData → vector embeddings → pgvector upsert.
Uses batched OpenAI calls to stay within rate limits.
"""
import time
import logging
from openai import OpenAI
from sqlalchemy.orm import Session
from ingestion.db import LegalChunk, SessionLocal
from ingestion.chunkers.legal_chunker import LegalChunkData
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

BATCH_SIZE = 100        # OpenAI allows up to 2048 inputs per call
RATE_LIMIT_SLEEP = 0.5  # seconds between batches


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API and return list of float vectors."""
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


def upsert_chunks(chunks: list[LegalChunkData], db: Session | None = None) -> int:
    """
    Embed and upsert a list of LegalChunkData into pgvector.
    Returns the count of rows inserted/updated.
    """
    close_session = db is None
    if db is None:
        db = SessionLocal()

    inserted = 0
    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.text for c in batch]

            logger.info(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)…")
            vectors = embed_texts(texts)

            for chunk, vector in zip(batch, vectors):
                # Skip if identical content already exists
                existing = (
                    db.query(LegalChunk)
                    .filter_by(version_hash=chunk.version_hash)
                    .first()
                )
                if existing:
                    continue

                row = LegalChunk(
                    id=chunk.id,
                    source=chunk.source,
                    title=chunk.title,
                    section=chunk.section,
                    jurisdiction=chunk.jurisdiction,
                    citation=chunk.citation,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    effective_date=chunk.effective_date,
                    version_hash=chunk.version_hash,
                    extra_metadata=chunk.metadata,
                    embedding=vector,
                )
                db.merge(row)
                inserted += 1

            db.commit()
            time.sleep(RATE_LIMIT_SLEEP)

        logger.info(f"Upserted {inserted} chunks into pgvector.")
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
    """
    Embed a query and return top-K similar chunks from pgvector.
    Returns list of dicts with text, citation, score.
    """
    query_vector = embed_texts([query_text])[0]
    db = SessionLocal()
    try:
        q = db.query(LegalChunk)
        if source_filter:
            q = q.filter(LegalChunk.source == source_filter)

        # cosine distance (<=> operator via pgvector)
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
                "score": 1.0,  # actual score requires raw SQL; placeholder
            }
            for r in results
        ]
    finally:
        db.close()
