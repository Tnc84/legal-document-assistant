"""FastAPI dependency providers for shared singletons."""

from __future__ import annotations

from functools import lru_cache

from legal_ai.config.settings import Settings, get_settings
from legal_ai.inference.comparator import DocumentComparator
from legal_ai.inference.llm_client import LLMClient, OllamaClient
from legal_ai.inference.qa_chain import QAChain
from legal_ai.inference.risk_detector import RiskDetector
from legal_ai.ingestion.embedder import Embedder
from legal_ai.ingestion.pipeline import IngestionPipeline
from legal_ai.resilience.resilient_llm import ResilientLLMClient
from legal_ai.retrieval.hybrid_retriever import HybridRetriever
from legal_ai.retrieval.vector_store import QdrantVectorStore


@lru_cache(maxsize=1)
def app_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def llm_client() -> LLMClient:
    settings = app_settings()
    client = OllamaClient(settings)
    if settings.ollama_cb_enabled:
        return ResilientLLMClient(client, settings)
    return client


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder(app_settings())


@lru_cache(maxsize=1)
def vector_store() -> QdrantVectorStore:
    store = QdrantVectorStore(app_settings())
    store.ensure_collection()
    return store


@lru_cache(maxsize=1)
def hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(vector_store(), embedder(), app_settings())


@lru_cache(maxsize=1)
def ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        embedder=embedder(),
        vector_store=vector_store(),
        settings=app_settings(),
    )


@lru_cache(maxsize=1)
def qa_chain() -> QAChain:
    return QAChain(hybrid_retriever(), llm_client(), app_settings())


@lru_cache(maxsize=1)
def risk_detector() -> RiskDetector:
    return RiskDetector(vector_store(), llm_client(), app_settings())


@lru_cache(maxsize=1)
def document_comparator() -> DocumentComparator:
    return DocumentComparator(
        embedder=embedder(),
        llm=llm_client(),
        settings=app_settings(),
    )
