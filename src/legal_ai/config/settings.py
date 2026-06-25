"""Centralized typed settings loaded from environment / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration. Single source of truth for runtime parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="ministral-3:3b")
    ollama_request_timeout: float = Field(default=120.0, ge=1.0)
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    ollama_max_tokens: int = Field(default=1024, ge=64, le=8192)

    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    embedding_dim: int = Field(default=1024, ge=64)

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="legal_contracts")

    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    retrieval_dense_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    retrieval_bm25_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    chunk_size_tokens: int = Field(default=900, ge=100, le=4096)
    chunk_overlap_tokens: int = Field(default=120, ge=0, le=1024)
    max_document_mb: int = Field(default=25, ge=1, le=500)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_log_level: str = Field(default="INFO")
    data_dir: Path = Field(default=Path("./data"))
    upload_dir: Path = Field(default=Path("./data/uploads"))

    otel_service_name: str = Field(default="legal-ai-api")
    otel_exporter_otlp_endpoint: str | None = Field(default=None)
    otel_traces_enabled: bool = Field(default=True)
    metrics_enabled: bool = Field(default=True)
    log_format: str = Field(default="text")

    redis_url: str | None = Field(default=None)

    ollama_cb_enabled: bool = Field(default=True)
    ollama_cb_failure_threshold: int = Field(default=5, ge=1, le=100)
    ollama_cb_recovery_timeout: float = Field(default=30.0, ge=1.0, le=600.0)

    rate_limit_enabled: bool = Field(default=True)
    rate_limit_qa: str = Field(default="10/minute")
    rate_limit_risk: str = Field(default="10/minute")
    rate_limit_ingest: str = Field(default="3/minute")
    rate_limit_compare: str = Field(default="3/minute")

    ingest_async_enabled: bool = Field(default=False)
    ingest_sync_fallback: bool = Field(default=True)
    ingest_max_retries: int = Field(default=2, ge=0, le=10)

    ui_api_base_url: str = Field(default="http://localhost:8000")

    @property
    def rate_limit_storage_uri(self) -> str:
        """Storage backend for the rate limiter: Redis when configured, else in-memory."""

        return self.redis_url or "memory://"

    @field_validator("api_log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        normalized = value.upper().strip()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"api_log_level must be one of {sorted(allowed)}")
        return normalized

    @field_validator("log_format")
    @classmethod
    def _normalize_log_format(cls, value: str) -> str:
        normalized = value.lower().strip()
        allowed = {"text", "json"}
        if normalized not in allowed:
            raise ValueError(f"log_format must be one of {sorted(allowed)}")
        return normalized

    @field_validator("data_dir", "upload_dir")
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    def ensure_directories(self) -> None:
        """Create runtime directories that the app needs to write to."""

        for path in (self.data_dir, self.upload_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
