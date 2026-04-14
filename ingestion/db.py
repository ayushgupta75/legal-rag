"""
Database models and schema setup for pgvector.
Run: python -m ingestion.db  to initialise the schema.
"""
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime,
    Integer, JSON, Index, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from datetime import datetime, UTC
from config import get_settings

settings = get_settings()
engine = create_engine(settings.postgres_dsn, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class LegalChunk(Base):
    """One indexed chunk from a legal document."""
    __tablename__ = "legal_chunks"

    id = Column(String, primary_key=True)          # uuid
    source = Column(String, nullable=False)         # constitution | uscode | cfr | caselaw
    title = Column(String)                          # e.g. "Title 18 - Crimes"
    section = Column(String)                        # e.g. "§ 1030"
    jurisdiction = Column(String, default="federal")
    citation = Column(String)                       # bluebook citation string
    text = Column(Text, nullable=False)             # raw chunk text
    char_count = Column(Integer)
    effective_date = Column(String)                 # ISO date string or None
    version_hash = Column(String)                   # sha256 of text for dedup
    extra_metadata = Column("metadata", JSON, default=dict)
    embedding = Column(Vector(1536))                # text-embedding-3-small dim
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC),
                        onupdate=lambda: datetime.now(UTC))


class QueryLog(Base):
    """Audit log of every query and which chunks were retrieved."""
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True)
    query = Column(Text)
    route = Column(String)                          # vector | live_tools | agent
    retrieved_chunk_ids = Column(JSON, default=list)
    answer = Column(Text)
    latency_ms = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


def init_db():
    """Create pgvector extension + all tables."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)

    # HNSW index for fast ANN search
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS legal_chunks_embedding_idx
            ON legal_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        conn.commit()
    print("Database initialised with pgvector + HNSW index.")


if __name__ == "__main__":
    init_db()
