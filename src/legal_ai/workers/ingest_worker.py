"""ARQ worker that ingests uploaded PDFs asynchronously.

Run with::

    arq legal_ai.workers.ingest_worker.WorkerSettings

The worker reuses a single :class:`IngestionPipeline` (and its loaded embedder)
across jobs and retries transient failures with exponential backoff.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings
from arq.worker import Retry

from legal_ai.config.logging import configure_logging, get_logger
from legal_ai.config.settings import Settings, get_settings
from legal_ai.ingestion.pipeline import IngestionPipeline
from legal_ai.workers.queue import build_redis_settings

_logger = get_logger("workers.ingest")


async def ingest_document_task(
    ctx: dict[str, Any], saved_path: str, original_filename: str
) -> dict[str, Any]:
    """Ingest one persisted PDF into Qdrant; retry transient errors with backoff."""

    settings: Settings = ctx["settings"]
    pipeline: IngestionPipeline = ctx["pipeline"]
    job_try: int = ctx.get("job_try", 1)
    path = Path(saved_path)

    try:
        result = await asyncio.to_thread(pipeline.ingest_pdf, path)
    except Exception as exc:
        if job_try <= settings.ingest_max_retries:
            backoff = 2**job_try
            _logger.warning(
                f"Ingest job for {original_filename} failed (try {job_try}); "
                f"retrying in {backoff}s: {exc}"
            )
            raise Retry(defer=backoff) from exc
        path.unlink(missing_ok=True)
        _logger.error(f"Ingest job for {original_filename} exhausted retries: {exc}")
        raise

    path.unlink(missing_ok=True)
    _logger.info(f"Async ingest complete for {original_filename} -> {result.document_id}")
    return result.to_dict()


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.api_log_level, settings.log_format)
    ctx["settings"] = settings
    ctx["pipeline"] = IngestionPipeline(settings=settings)
    _logger.info("Ingest worker started")


async def shutdown(ctx: dict[str, Any]) -> None:
    _logger.info("Ingest worker shutting down")


def _worker_redis_settings() -> RedisSettings:
    settings = get_settings()
    if settings.redis_url:
        return build_redis_settings(settings.redis_url)
    return RedisSettings()


class WorkerSettings:
    """ARQ worker configuration entrypoint."""

    functions = [ingest_document_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _worker_redis_settings()
    max_tries = get_settings().ingest_max_retries + 1
