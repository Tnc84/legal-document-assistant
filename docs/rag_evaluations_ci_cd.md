# Evaluări RAG + CI/CD

**Obiectiv:** set reproductibil de întrebări + contracte, metrici retrieval și
calitate răspuns, rulabil manual sau în CI; plus un pipeline CI/CD care rulează
automat lint, type-check, gate-ul de evaluare și publică imaginea Docker.

> Extras din planul „Production Readiness Pillars" (Plan 2).

## Plan 2 — Evaluări RAG

### Structură date

```
data/evals/
  contracts/           # PDF-uri fixe (2-5 contracte reprezentative)
  qa_golden.jsonl      # întrebări + răspuns așteptat + document_id + pagini/clauze relevante
  retrieval_cases.jsonl  # query + document_id + chunk_ids așteptate (ground truth)
  risk_cases.jsonl     # document_id + categorii așteptate (opțional, faza 2)
```

Format `qa_golden.jsonl` (exemplu câmpuri):
- `id`, `question`, `document_ids`, `expected_answer_contains[]`, `expected_citations[]`, `language`

### Modul nou: `src/legal_ai/evals/`

| Fișier | Rol |
|---|---|
| `runner.py` | orchestrator: ingest contracte eval → rulează cazuri → raport |
| `retrieval_metrics.py` | Recall@k, MRR, Hit@k pe `chunk_id` / `section_path` |
| `answer_metrics.py` | overlap lexical (RO+EN), `expected_answer_contains` hit rate |
| `citation_metrics.py` | pagină/secțiune citată vs ground truth |
| `report.py` | JSON + markdown summary |

Runner-ul reutilizează componentele existente, fără reimplementare:
- ingest: `IngestionPipeline.ingest_pdf` (`src/legal_ai/ingestion/pipeline.py`)
- retrieval cases: `HybridRetriever.retrieve` (`src/legal_ai/retrieval/hybrid_retriever.py`)
- qa cases: `QAChain.answer` (`src/legal_ai/inference/qa_chain.py`) (întoarce deja `answer` + `citations`)

### Script CLI: `scripts/run_eval.py`

```bash
python scripts/run_eval.py --suite qa --top-k 8 --output reports/eval_YYYYMMDD.json
python scripts/run_eval.py --suite retrieval --top-k 8 --output reports/eval.json
```

- `--suite retrieval` nu necesită LLM (doar Qdrant + embedder) → potrivit pentru CI
- `--suite qa` necesită Ollama → rulare locală / runner cu GPU

### Metrici țintă (MVP)

| Metrică | Prag inițial |
|---|---|
| Recall@8 (retrieval) | >= 0.70 |
| Citation page match | >= 0.60 |
| Answer contains (keywords) | >= 0.65 |

`scripts/run_eval.py` întoarce **exit code != 0** dacă o metrică scade sub prag
(praguri configurabile prin argument/env), ca să poată fi folosit ca gate în CI.

### Sursă date inițială

- 3-5 PDF-uri din `data/` sau CUAD export
- 20-30 întrebări manuale anotate (RO + EN) — calitate > cantitate

## Plan CI/CD — GitHub Actions

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

### Declanșare

- `pull_request` către `main`: rulează lint, type-check, eval gate, build imagine (fără push)
- `push` pe `main` și tag-uri `v*`: aceleași joburi + push imagine în GHCR

### Joburi

| Job | Rol | Blochează? |
|---|---|---|
| `lint` | `ruff check` + `black --check` pe `src`/`scripts` | da |
| `type-check` | `mypy src/legal_ai` | nu (`continue-on-error`, raport informativ) |
| `eval` | pornește serviciu Qdrant, rulează `--suite retrieval`, urcă raportul ca artifact | da (când suita există) |
| `docker` | build imagine; push în `ghcr.io` doar pe `push` | da |

### Note de implementare

- Jobul `eval` rulează suita **retrieval** (fără LLM) — Qdrant ca service container,
  `EMBEDDING_DEVICE=cpu`. Gate-ul e protejat: dacă `scripts/run_eval.py` nu există
  încă, jobul iese cu `notice` și nu pică (permite adoptare incrementală).
- Imaginea se publică în GitHub Container Registry cu `docker/metadata-action`
  (tag-uri: branch, semver pe tag-uri, `sha`), autentificare cu `GITHUB_TOKEN`
  și `permissions: packages: write`.
- Cache pip + cache buildx (`type=gha`) pentru rulări rapide.

### Extensii ulterioare (faza 2)

- Suită `qa` completă în CI pe runner cu Ollama (self-hosted sau container model)
- Comentariu automat pe PR cu diff-ul metricilor față de `main`
- Job `cd-deploy` separat (ex. deploy pe server/registry privat) pe tag-uri release
