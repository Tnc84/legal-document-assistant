# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only Torch first so sentence-transformers does not pull CUDA wheels.
# This layer is cached and only changes when the Torch version changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md /app/
COPY src /app/src

# Torch is already satisfied (CPU), so this resolves the rest of the RAG stack.
# Cache mount keeps wheels across rebuilds even when src changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

EXPOSE 8000

CMD ["uvicorn", "legal_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
