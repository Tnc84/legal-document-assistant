"""RAG evaluation package: golden datasets, metrics, runner and reporting.

Submodules are imported lazily by callers to avoid pulling heavy dependencies
(embedder, LLM client) at package import time.
"""
