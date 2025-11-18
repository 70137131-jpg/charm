"""Configuration management"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    """Configuration for RAG pipeline"""

    # API Keys
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    # Model Configuration
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    )
    llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "2000"))
    )

    # Vector Database
    vector_db_type: str = Field(
        default_factory=lambda: os.getenv("VECTOR_DB_TYPE", "faiss")
    )
    vector_db_path: str = Field(
        default_factory=lambda: os.getenv("VECTOR_DB_PATH", "./data/vector_store")
    )

    # Retrieval Configuration
    top_k: int = Field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    retrieval_mode: str = Field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid")
    )

    # Reranking
    use_reranker: bool = Field(
        default_factory=lambda: os.getenv("USE_RERANKER", "true").lower() == "true"
    )
    reranker_model: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    rerank_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_K", "10"))
    )

    class Config:
        arbitrary_types_allowed = True
