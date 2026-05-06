"""Thin HTTP client used by the Streamlit UI to talk to the FastAPI backend."""

from __future__ import annotations

from typing import Any

import httpx

from legal_ai.config.settings import get_settings


class LegalApiClient:
    """Encapsulates the REST calls so the UI stays declarative."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0) -> None:
        self._base_url = (base_url or get_settings().ui_api_base_url).rstrip("/")
        self._timeout = timeout

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{self._base_url}/health")
            response.raise_for_status()
            return response.json()

    def ingest(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        files = {"file": (filename, file_bytes, "application/pdf")}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/ingest", files=files)
            response.raise_for_status()
            return response.json()

    def qa(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question}
        if document_ids:
            payload["document_ids"] = document_ids
        if top_k:
            payload["top_k"] = top_k
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/qa", json=payload)
            response.raise_for_status()
            return response.json()

    def risk(self, document_id: str, max_chunks: int = 200) -> dict[str, Any]:
        payload = {"document_id": document_id, "max_chunks": max_chunks}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/risk", json=payload)
            response.raise_for_status()
            return response.json()

    def compare(
        self,
        left_bytes: bytes,
        left_name: str,
        right_bytes: bytes,
        right_name: str,
    ) -> dict[str, Any]:
        files = {
            "left": (left_name, left_bytes, "application/pdf"),
            "right": (right_name, right_bytes, "application/pdf"),
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(f"{self._base_url}/compare", files=files)
            response.raise_for_status()
            return response.json()
