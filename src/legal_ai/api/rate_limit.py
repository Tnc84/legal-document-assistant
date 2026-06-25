"""Rate limiting setup built on slowapi (per client IP).

A single module-level :class:`Limiter` is created from settings so endpoints can
attach per-route limits via ``@limiter.limit(...)``. The storage backend is
Redis when ``REDIS_URL`` is configured, otherwise in-memory.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from legal_ai.config.settings import get_settings


def _build_limiter() -> Limiter:
    settings = get_settings()
    return Limiter(
        key_func=get_remote_address,
        storage_uri=settings.rate_limit_storage_uri,
        enabled=settings.rate_limit_enabled,
        headers_enabled=True,
    )


limiter = _build_limiter()


def qa_limit() -> str:
    return get_settings().rate_limit_qa


def risk_limit() -> str:
    return get_settings().rate_limit_risk


def ingest_limit() -> str:
    return get_settings().rate_limit_ingest


def compare_limit() -> str:
    return get_settings().rate_limit_compare
