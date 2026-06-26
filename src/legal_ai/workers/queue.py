"""ARQ queue helpers: pool creation, enqueue and job status lookup.

Isolated from the FastAPI layer so the API only depends on small async helpers
and never imports the worker runtime directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job, JobStatus

from legal_ai.config.logging import get_logger
from legal_ai.config.settings import Settings

_logger = get_logger("workers.queue")

INGEST_TASK_NAME = "ingest_document_task"


@dataclass(frozen=True)
class JobState:
    """Snapshot of an ARQ job's lifecycle and (optional) result."""

    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


def build_redis_settings(redis_url: str) -> RedisSettings:
    """Parse a Redis DSN into ARQ connection settings."""

    return RedisSettings.from_dsn(redis_url)


async def create_arq_pool(settings: Settings) -> ArqRedis | None:
    """Create an ARQ Redis pool, or ``None`` when Redis is not configured."""

    if not settings.redis_url:
        return None
    pool = await create_pool(build_redis_settings(settings.redis_url))
    _logger.info("ARQ Redis pool created for ingest queue")
    return pool


async def enqueue_ingest(pool: ArqRedis, saved_path: str, original_filename: str) -> str:
    """Enqueue an ingestion job and return its job id."""

    job = await pool.enqueue_job(INGEST_TASK_NAME, saved_path, original_filename)
    if job is None:
        raise RuntimeError("Failed to enqueue ingest job (duplicate job id)")
    return job.job_id


async def fetch_job_state(pool: ArqRedis, job_id: str) -> JobState:
    """Return the current state and result of a previously enqueued job."""

    job = Job(job_id, redis=pool)
    status = await job.status()
    if status == JobStatus.not_found:
        return JobState(job_id=job_id, status="not_found")

    result_info = await job.result_info()
    if result_info is None:
        return JobState(job_id=job_id, status=status.value)
    if result_info.success:
        return JobState(job_id=job_id, status="complete", result=result_info.result)
    return JobState(job_id=job_id, status="failed", error=str(result_info.result))
