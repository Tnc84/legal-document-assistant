# Legal AI Assistant

RAG + fine-tuning assistant pentru analiza contractelor juridice. Trei module:

1. **Q&A pe contracte** cu citare sursă (pagină + clauză)
2. **Risk Clause Detector** — clasificare clauze de risc cu output JSON structurat
3. **Document Comparator** — diff semantic între două versiuni cu scor `risk_delta`

## Stack

- LLM local: **Ministral 3B** via **Ollama** (GGUF)
- Embeddings: `intfloat/multilingual-e5-large` (RO + EN)
- Vector DB: **Qdrant** (dense + BM25 hybrid)
- Orchestrare RAG: **LlamaIndex**
- API: **FastAPI**, UI demo: **Streamlit**
- Ingestion PDF: **PyMuPDF** + **pdfplumber**
- Observabilitate: **OpenTelemetry** (traces + metrics) + **Prometheus** + loguri JSON
- Reziliență: **circuit breaker** Ollama + **rate limiting** (slowapi) + **ingest queue** (ARQ + Redis)
- Fine-tuning: **TRL + PEFT** (AMD/ROCm friendly) + optional **bitsandbytes** pe NVIDIA

## Setup local

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv
.venv\Scripts\activate
uv pip install -e .

cp .env.example .env

docker compose up -d qdrant

ollama pull ministral:3b   # sau alt tag GGUF compatibil

uvicorn legal_ai.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit run src\legal_ai\ui\app.py
```

Pentru fine-tuning:
- AMD/ROCm: `uv pip install -e ".[finetune]"`
- NVIDIA/CUDA: `uv pip install -e ".[finetune,finetune-nvidia]"`

## Structură

```
src/legal_ai/
  config/         # settings + logging
  ingestion/      # parser PDF, chunker semantic, embedder
  retrieval/      # qdrant store, hybrid retriever
  inference/      # qa_chain, risk_detector, comparator
  api/            # FastAPI endpoints + rate limiting
  observability/  # OpenTelemetry: telemetry, metrics, request context, middleware
  resilience/     # circuit breaker + resilient LLM wrapper
  workers/        # ARQ ingest worker + queue helpers
  ui/             # Streamlit demo
  fine_tuning/    # CUAD prep + QLoRA + merge/export
  utils/          # helpers comuni
prompts/          # prompturi versionate (jurisdic + risk + compare)
data/             # raw / processed / cuad / uploads
models/           # adapteri LoRA + export GGUF
```

## Data layout recomandat

- `data/contracts/` - PDF-uri contractuale pentru RAG (ingestion/indexing)
- `data/cuad/` - fișiere dataset CUAD (input fine-tuning)
- `data/processed/` - output preprocesare (`cuad_sft.jsonl`)
- `data/uploads/` - upload-uri temporare API/UI

## CUAD -> PDF pentru RAG

Datasetul `theatticusproject/cuad` se descarcă în cache HuggingFace; nu apare
automat în `data/`. Pentru a obține PDF-uri fizice în proiect:

```bash
source .venv/bin/activate
uv pip install datasets
python scripts/export_cuad_pdfs.py
ls -lh data/contracts | head
ls data/contracts | wc -l
```

Scriptul `scripts/export_cuad_pdfs.py`:
- încarcă split-ul `train` din CUAD;
- citește calea locală reală a fiecărui PDF din cache;
- copiază fișierele în `data/contracts/` cu nume stabil (`cuad_XXXX_...pdf`).

## CUAD pentru fine-tuning (JSONL)

`prepare_cuad.py` așteaptă un fișier CUAD în format SQuAD-like (`data -> paragraphs -> qas`).
După ce ai fișierul sursă valid, rulezi:

```bash
uv run python -m legal_ai.fine_tuning.prepare_cuad \
  --cuad-json data/cuad/CUAD_v1.json \
  --output data/processed/cuad_sft.jsonl
```

## Date pentru fine-tuning (surse contracte)

- **CUAD** (510 contracte adnotate) — set principal
- **SEC EDGAR** contract exhibits (filings publice) — scalare volum
- **EUR-Lex** și portaluri achiziții publice — corpus juridic public
- Adnotare țintită RO (500-1500 clauze) pentru relevanță locală
- Augmentare prin parafrazare + back-translation pentru clase rare

## Endpointuri API (rezumat)

- `POST /ingest` — încarcă și indexează un PDF (sincron, sau `202 job_id` când coada async e activă)
- `GET /ingest/jobs/{job_id}` — status job ingest async (queued/in_progress/complete/failed)
- `POST /qa` — întrebare în limbaj natural pe documente indexate
- `POST /risk` — detectează clauze de risc cu output JSON
- `POST /compare` — comparator semantic între două PDF-uri
- `GET /health` — status servicii
- `GET /metrics` — metrici Prometheus (OpenTelemetry)

## Observabilitate

Instrumentare OpenTelemetry: traces pe fluxurile RAG (`rag.qa`, `rag.risk`,
`rag.compare`, `rag.retrieve`, `embed.encode`, `llm.complete`), token usage de la
Ollama și instrumentare automată FastAPI + httpx. Fiecare răspuns conține
headerul `X-Request-ID` (acceptă și unul trimis de client).

Config în `.env`:

```
OTEL_SERVICE_NAME=legal-ai-api
OTEL_EXPORTER_OTLP_ENDPOINT=        # gol = fără export OTLP; ex. http://localhost:4318
OTEL_TRACES_ENABLED=true
METRICS_ENABLED=true
LOG_FORMAT=text                     # text | json (json pentru log aggregation)
```

Metrici expuse la `GET /metrics`:
- `rag_operation_duration_seconds{operation,success}` — latență per operație
- `llm_tokens_total{model,direction}` — token usage prompt/completion
- `http_client_duration_milliseconds{...}` — apeluri ieșite (Ollama/Qdrant)

Traces (Jaeger/Tempo) când `OTEL_EXPORTER_OTLP_ENDPOINT` e setat:

```bash
docker run -d --name jaeger -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one
# setează OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318, apoi vezi UI la :16686
```

## Reziliență

Trei mecanisme de protecție la supraîncărcare și indisponibilitate.

### Circuit breaker (Ollama)

Wrapper `ResilientLLMClient` peste `OllamaClient` cu un breaker în 3 stări
(`closed` → `open` → `half_open`). După `OLLAMA_CB_FAILURE_THRESHOLD` eșecuri
consecutive circuitul se deschide și apelurile eșuează rapid cu `503` +
header `Retry-After`, până trece `OLLAMA_CB_RECOVERY_TIMEOUT`. Starea e expusă
ca metrica `circuit_breaker_state{breaker}` (0=closed, 1=open, 2=half_open).

```
OLLAMA_CB_ENABLED=true
OLLAMA_CB_FAILURE_THRESHOLD=5
OLLAMA_CB_RECOVERY_TIMEOUT=30
```

### Rate limiting (API)

`slowapi` per IP client, configurabil per endpoint; depășirea întoarce `429` cu
`Retry-After`, iar răspunsurile includ headere `X-RateLimit-*`. Backend de
stocare: Redis dacă `REDIS_URL` e setat, altfel in-memory.

```
RATE_LIMIT_ENABLED=true
RATE_LIMIT_QA=10/minute
RATE_LIMIT_RISK=10/minute
RATE_LIMIT_INGEST=3/minute
RATE_LIMIT_COMPARE=3/minute
```

### Ingest queue async (ARQ + Redis)

Când `INGEST_ASYNC_ENABLED=true` și `REDIS_URL` e setat, `POST /ingest` pune
jobul în coadă și răspunde `202` cu `job_id`; un worker ARQ separat procesează
ingestul (retry cu backoff exponențial, `INGEST_MAX_RETRIES`). Statusul se
interoghează la `GET /ingest/jobs/{job_id}`. Dacă Redis lipsește, se face
fallback sincron când `INGEST_SYNC_FALLBACK=true`.

```
REDIS_URL=redis://localhost:6379
INGEST_ASYNC_ENABLED=false
INGEST_SYNC_FALLBACK=true
INGEST_MAX_RETRIES=2
```

Pornire worker (separat de API):

```bash
arq legal_ai.workers.ingest_worker.WorkerSettings
```

### Docker

`docker compose up -d` pornește acum și `redis` + `worker` (ingest async activat
implicit în compose). Pentru doar infra de bază: `docker compose up -d qdrant redis`.

## Evaluări RAG

Suită reproductibilă de evaluare (retrieval, calitate răspuns, citări) cu praguri
folosite ca gate de calitate. Specificația completă: [`docs/rag_evaluations_ci_cd.md`](docs/rag_evaluations_ci_cd.md).

```bash
python scripts/run_eval.py --suite retrieval --top-k 8 --output reports/eval.json
python scripts/run_eval.py --suite qa --top-k 8 --output reports/eval_YYYYMMDD.json
```

- `--suite retrieval` nu necesită LLM (doar Qdrant + embedder) — potrivit pentru CI
- `--suite qa` necesită Ollama — rulare locală
- praguri MVP: Recall@8 ≥ 0.70, citation page match ≥ 0.60, answer contains ≥ 0.65

> Suita de evaluare (`src/legal_ai/evals/` + `scripts/run_eval.py`) e planificată;
> vezi documentul de mai sus pentru structura de date și module.

## CI/CD (GitHub Actions)

Pipeline în [`.github/workflows/ci.yml`](.github/workflows/ci.yml), declanșat la
`push` pe `main`, `pull_request` către `main` și tag-uri `v*`.

| Job | Rol |
|---|---|
| `lint` | `ruff check` + `black --check` pe `src`/`scripts` |
| `type-check` | `mypy src/legal_ai` (informativ, nu blochează) |
| `eval` | pornește Qdrant ca service, rulează gate-ul de evaluare (suita `retrieval`) |
| `docker` | build imagine; push în `ghcr.io` doar pe `push` (branch `main` / tag-uri) |

- Pe `pull_request` imaginea se construiește dar **nu** se publică; pe `push` se
  publică în GitHub Container Registry cu `GITHUB_TOKEN`.
- Gate-ul `eval` e protejat: dacă `scripts/run_eval.py` nu există încă, jobul trece
  cu un `notice` și nu blochează pipeline-ul.

Activare: comite workflow-ul și fă `push` pe GitHub — Actions rulează automat.
Pentru publicarea imaginii: **Settings → Actions → General → Workflow permissions →
Read and write permissions**.

## Pre-commit (lint + format local)

Hook-uri `ruff` (cu `--fix`) și `black` rulate automat la fiecare commit, pe
`src/` și `scripts/` — aceeași zonă ca jobul `lint` din CI. Config:
[`.pre-commit-config.yaml`](.pre-commit-config.yaml).

```bash
uv pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files   # opțional: rulează o dată pe tot codebase-ul
```

- La fiecare `git commit`, ruff repară lint-ul și black formatează fișierele
  modificate; dacă se schimbă ceva, commit-ul se oprește ca să faci `git add` din nou.
- Rulează pe orice branch (inclusiv `develop`) — prinde problemele local, înainte de CI.
- Versiunile hook-urilor sunt aliniate cu cele din `pyproject.toml` (`[dev]`).
