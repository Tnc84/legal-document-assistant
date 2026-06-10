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
  api/            # FastAPI endpoints
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

- `POST /ingest` — încarcă și indexează un PDF
- `POST /qa` — întrebare în limbaj natural pe documente indexate
- `POST /risk` — detectează clauze de risc cu output JSON
- `POST /compare` — comparator semantic între două PDF-uri
- `GET /health` — status servicii
