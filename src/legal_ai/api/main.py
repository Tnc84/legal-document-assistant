"""FastAPI entrypoint exposing ingestion, Q&A, risk and comparator endpoints."""

from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from arq.connections import ArqRedis
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from legal_ai.api.dependencies import (
    app_settings,
    document_comparator,
    ingestion_pipeline,
    qa_chain,
    risk_detector,
    vector_store,
)
from legal_ai.api.rate_limit import (
    compare_limit,
    ingest_limit,
    limiter,
    qa_limit,
    risk_limit,
)
from legal_ai.api.schemas import (
    CitationModel,
    ClauseDiffModel,
    ComparisonResponseModel,
    HealthResponse,
    IngestAcceptedResponse,
    IngestJobStatusResponse,
    IngestResponse,
    QARequest,
    QAResponseModel,
    RiskFindingModel,
    RiskRequest,
    RiskResponseModel,
)
from legal_ai.config.logging import configure_logging, get_logger
from legal_ai.config.settings import Settings
from legal_ai.inference.comparator import ClauseDiff, DocumentComparator
from legal_ai.inference.qa_chain import QAChain, QAResponse
from legal_ai.inference.risk_detector import RiskDetector, RiskReport
from legal_ai.ingestion.pipeline import IngestionPipeline
from legal_ai.observability.middleware import RequestContextMiddleware
from legal_ai.observability.telemetry import configure_telemetry, shutdown_telemetry
from legal_ai.resilience.circuit_breaker import CircuitBreakerError
from legal_ai.retrieval.vector_store import QdrantVectorStore
from legal_ai.workers.queue import create_arq_pool, enqueue_ingest, fetch_job_state

_logger = get_logger("api")


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    _logger.warning(f"Rate limit exceeded on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": "60"},
    )


async def _circuit_breaker_handler(
    request: Request, exc: CircuitBreakerError
) -> Response:
    _logger.warning(f"Circuit breaker open on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after)},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app_settings()
    configure_logging(settings.api_log_level, settings.log_format)
    configure_telemetry(settings, app)
    settings.ensure_directories()
    vector_store().ensure_collection()
    app.state.arq_pool = None
    if settings.ingest_async_enabled:
        app.state.arq_pool = await create_arq_pool(settings)
        if app.state.arq_pool is None:
            _logger.warning(
                "INGEST_ASYNC_ENABLED is set but REDIS_URL is missing; "
                "falling back to synchronous ingest"
            )
    _logger.info("Legal AI API started")
    yield
    if app.state.arq_pool is not None:
        await app.state.arq_pool.close()
    shutdown_telemetry()
    _logger.info("Legal AI API shutting down")


app = FastAPI(
    title="Legal AI Assistant",
    description="RAG + fine-tuning legal contract assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_exception_handler(CircuitBreakerError, _circuit_breaker_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RequestContextMiddleware)


SettingsDep = Annotated[Settings, Depends(app_settings)]
PipelineDep = Annotated[IngestionPipeline, Depends(ingestion_pipeline)]
QADep = Annotated[QAChain, Depends(qa_chain)]
RiskDep = Annotated[RiskDetector, Depends(risk_detector)]
CompareDep = Annotated[DocumentComparator, Depends(document_comparator)]
StoreDep = Annotated[QdrantVectorStore, Depends(vector_store)]


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep, store: StoreDep) -> HealthResponse:
    qdrant_ok = _check_qdrant(store)
    ollama_ok = _check_ollama(settings)
    return HealthResponse(
        status="ok" if (qdrant_ok and ollama_ok) else "degraded",
        qdrant=qdrant_ok,
        ollama=ollama_ok,
        embedding_model=settings.embedding_model,
        llm_model=settings.ollama_model,
    )


@app.post("/ingest", response_model=None)
@limiter.limit(ingest_limit)
async def ingest_document(
    request: Request,
    settings: SettingsDep,
    pipeline: PipelineDep,
    response: Response,
    file: Annotated[UploadFile, File(...)],
) -> IngestResponse | IngestAcceptedResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    saved_path = _persist_upload(file, settings)
    if saved_path.stat().st_size > settings.max_document_mb * 1024 * 1024:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="PDF exceeds configured max size")

    pool = _get_arq_pool(request)
    if settings.ingest_async_enabled and pool is not None:
        job_id = await enqueue_ingest(pool, str(saved_path), file.filename)
        response.status_code = status.HTTP_202_ACCEPTED
        return IngestAcceptedResponse(job_id=job_id, status="queued")

    if settings.ingest_async_enabled and not settings.ingest_sync_fallback:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503, detail="Ingest queue unavailable and sync fallback disabled"
        )

    result = pipeline.ingest_pdf(saved_path)
    return IngestResponse(**result.to_dict())


@app.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def ingest_job_status(
    request: Request, job_id: str
) -> IngestJobStatusResponse:
    pool = _get_arq_pool(request)
    if pool is None:
        raise HTTPException(status_code=404, detail="Ingest queue is not enabled")

    state = await fetch_job_state(pool, job_id)
    if state.status == "not_found":
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = IngestResponse(**state.result) if state.result else None
    return IngestJobStatusResponse(
        job_id=state.job_id,
        status=state.status,
        result=result,
        error=state.error,
    )


@app.post("/qa", response_model=QAResponseModel)
@limiter.limit(qa_limit)
async def run_qa(request: Request, payload: QARequest, qa: QADep) -> QAResponseModel:
    response = qa.answer(
        question=payload.question,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
    )
    return _qa_to_schema(response)


@app.post("/risk", response_model=RiskResponseModel)
@limiter.limit(risk_limit)
async def detect_risk(
    request: Request, payload: RiskRequest, detector: RiskDep
) -> RiskResponseModel:
    report = detector.analyze_document(payload.document_id, max_chunks=payload.max_chunks)
    return _risk_to_schema(report)


@app.post("/compare", response_model=ComparisonResponseModel)
@limiter.limit(compare_limit)
async def compare_documents(
    request: Request,
    settings: SettingsDep,
    comparator: CompareDep,
    left: Annotated[UploadFile, File(...)],
    right: Annotated[UploadFile, File(...)],
) -> ComparisonResponseModel:
    for upload in (left, right):
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Both files must be PDFs")
    left_path = _persist_upload(left, settings)
    right_path = _persist_upload(right, settings)
    try:
        report = comparator.compare_pdfs(left_path, right_path)
        return _comparison_to_schema(report)
    finally:
        left_path.unlink(missing_ok=True)
        right_path.unlink(missing_ok=True)


def _get_arq_pool(request: Request) -> ArqRedis | None:
    return getattr(request.app.state, "arq_pool", None)


def _persist_upload(upload: UploadFile, settings: Settings) -> Path:
    settings.ensure_directories()
    suffix = Path(upload.filename or "document.pdf").suffix or ".pdf"
    target = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return target


def _check_qdrant(store: QdrantVectorStore) -> bool:
    try:
        store.ensure_collection()
        return True
    except Exception as exc:
        _logger.warning(f"Qdrant health check failed: {exc}")
        return False


def _check_ollama(settings: Settings) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{settings.ollama_host}/api/tags")
            response.raise_for_status()
        return True
    except Exception as exc:
        _logger.warning(f"Ollama health check failed: {exc}")
        return False


def _qa_to_schema(response: QAResponse) -> QAResponseModel:
    return QAResponseModel(
        question=response.question,
        answer=response.answer,
        citations=[
            CitationModel(
                document_id=c.document_id,
                document_title=c.document_title,
                page_start=c.page_start,
                page_end=c.page_end,
                section_path=c.section_path,
                snippet=c.snippet,
                score=c.score,
            )
            for c in response.citations
        ],
    )


def _risk_to_schema(report: RiskReport) -> RiskResponseModel:
    return RiskResponseModel(
        document_id=report.document_id,
        findings=[
            RiskFindingModel(
                category=f.category,
                severity=f.severity,
                source_text=f.source_text,
                page=f.page,
                section=f.section,
                rationale=f.rationale,
                recommendation=f.recommendation,
            )
            for f in report.findings
        ],
    )


def _comparison_to_schema(report) -> ComparisonResponseModel:
    return ComparisonResponseModel(
        left_title=report.left_title,
        right_title=report.right_title,
        total_left=report.total_left,
        total_right=report.total_right,
        total_risk_delta=report.total_risk_delta,
        diffs=[_diff_to_schema(d) for d in report.diffs],
    )


def _diff_to_schema(diff: ClauseDiff) -> ClauseDiffModel:
    return ClauseDiffModel(
        change_type=diff.change_type,
        similarity=diff.similarity,
        risk_delta=diff.risk_delta,
        summary=diff.summary,
        rationale=diff.rationale,
        left_section=diff.left_section,
        right_section=diff.right_section,
        left_page=diff.left_page,
        right_page=diff.right_page,
        left_text=diff.left_text,
        right_text=diff.right_text,
    )
