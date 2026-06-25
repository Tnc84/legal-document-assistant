# syntax=docker/dockerfile:1

# ---- Builder stage: toolchain + dependencies into an isolated venv ----
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

# Install CPU-only Torch first so sentence-transformers does not pull CUDA wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md /app/
COPY src /app/src

# Torch already satisfied (CPU), so this resolves the rest of the RAG stack.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# ---- Runtime stage: slim image without build toolchain ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Only runtime OS libs (no compilers). apt upgrade pulls security patches.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

EXPOSE 8000

CMD ["uvicorn", "legal_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
