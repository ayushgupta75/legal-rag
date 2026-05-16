from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Live legal APIs
    courtlistener_api_key: str = ""
    congress_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "legal-rag"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "legal_chunks"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # Model config
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k_retrieval: int = 8
    rerank_top_k: int = 4

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
