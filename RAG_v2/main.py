"""
RAG_v2/main.py

LlamaIndex ingestion pipeline — mirrors v1 but uses LlamaIndex for
chunking, embedding, and storing instead of the custom embedder.

Writes to `llamaindex_chunks` table in the same pgvector DB so v1 data
is never touched.

Run:
    python RAG_v2/main.py --source constitution
    python RAG_v2/main.py --source constitution --skip-index
    python RAG_v2/main.py --rebuild-index
"""
import sys
import argparse
import logging
sys.path.insert(0, ".")

from sqlalchemy import text
from llama_index.core import Document, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from ingestion.db import engine, init_db
from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

CONSTITUTION_FILE = "data/constitution.txt"
TABLE_NAME = "llamaindex_chunks"   # LlamaIndex stores it as data_llamaindex_chunks
DB_TABLE_NAME = f"data_{TABLE_NAME}"

# Chunk size in tokens — ~512 tokens ≈ 2000 chars, matching v1 MAX_CHARS
CHUNK_SIZE = 512
CHUNK_OVERLAP = 40


def get_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        host=settings.postgres_host,
        port=str(settings.postgres_port),
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        table_name=TABLE_NAME,
        embed_dim=settings.embedding_dim,
    )


def get_embed_model() -> OpenAIEmbedding:
    return OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        embed_batch_size=200,
    )


def drop_hnsw_index():
    with engine.connect() as conn:
        conn.execute(text(f"DROP INDEX IF EXISTS {DB_TABLE_NAME}_embedding_idx"))
        conn.commit()
    logger.info("HNSW index dropped.")


def rebuild_hnsw_index():
    logger.info("Rebuilding HNSW index…")
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS {DB_TABLE_NAME}_embedding_idx
            ON {DB_TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        conn.commit()
    logger.info("HNSW index ready.")


def ingest_constitution():
    logger.info("── Ingesting US Constitution (LlamaIndex) ──")

    with open(CONSTITUTION_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    document = Document(
        text=text,
        metadata={
            "source": "constitution",
            "title": "United States Constitution",
            "jurisdiction": "federal",
            "citation_prefix": "U.S. Const.",
            "effective_date": "1788-06-21",
        },
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            get_embed_model(),
        ],
        vector_store=get_vector_store(),
    )

    nodes = pipeline.run(documents=[document], show_progress=True)
    logger.info(f"  {len(nodes)} nodes ingested into `{TABLE_NAME}`")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["constitution"],
        default="constitution",
        help="Source to ingest",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip rebuilding the HNSW index after ingestion",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Only rebuild the HNSW index — skip all ingestion",
    )
    args = parser.parse_args()

    init_db()

    if args.rebuild_index:
        rebuild_hnsw_index()
        return

    drop_hnsw_index()

    if args.source == "constitution":
        ingest_constitution()

    if args.skip_index:
        logger.info("Skipping HNSW index rebuild (--skip-index).")
    else:
        rebuild_hnsw_index()

    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()
