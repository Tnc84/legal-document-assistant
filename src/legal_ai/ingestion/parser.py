"""PDF parser combining PyMuPDF (fast text + layout) with pdfplumber (tables)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pdfplumber

from legal_ai.config.logging import get_logger
from legal_ai.utils.text import normalize_legal_text

_logger = get_logger("ingestion.parser")


@dataclass(frozen=True)
class ParsedPage:
    """One parsed PDF page with normalized text and structured tables."""

    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDocument:
    """A fully parsed PDF, retaining per-page granularity for citations."""

    source_path: str
    title: str
    page_count: int
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


class PdfParser:
    """Robust PDF parser. Single responsibility: produce `ParsedDocument`."""

    def __init__(self, extract_tables: bool = True) -> None:
        self._extract_tables = extract_tables

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        _logger.info(f"Parsing PDF {path.name} ({path.stat().st_size / 1024:.1f} KB)")
        text_pages = self._extract_text_pages(path)
        table_pages: dict[int, list[list[list[str]]]] = {}
        if self._extract_tables:
            table_pages = self._extract_table_pages(path)

        pages: list[ParsedPage] = []
        for page_number, raw_text in text_pages.items():
            normalized = normalize_legal_text(raw_text)
            tables = table_pages.get(page_number, [])
            pages.append(ParsedPage(page_number=page_number, text=normalized, tables=tables))

        title = self._extract_title(path) or path.stem
        return ParsedDocument(
            source_path=str(path),
            title=title,
            page_count=len(pages),
            pages=pages,
        )

    @staticmethod
    def _extract_title(path: Path) -> str | None:
        try:
            with fitz.open(path) as doc:
                metadata = doc.metadata or {}
                title = (metadata.get("title") or "").strip()
                return title or None
        except Exception:
            return None

    @staticmethod
    def _extract_text_pages(path: Path) -> dict[int, str]:
        pages: dict[int, str] = {}
        with fitz.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                pages[index] = page.get_text("text") or ""
        return pages

    @staticmethod
    def _extract_table_pages(path: Path) -> dict[int, list[list[list[str]]]]:
        tables_by_page: dict[int, list[list[list[str]]]] = {}
        try:
            with pdfplumber.open(path) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables() or []
                    if not tables:
                        continue
                    sanitized: list[list[list[str]]] = []
                    for table in tables:
                        rows: list[list[str]] = []
                        for row in table:
                            rows.append([(cell or "").strip() for cell in row])
                        sanitized.append(rows)
                    tables_by_page[index] = sanitized
        except Exception as exc:  # pdfplumber is best-effort for tables
            _logger.warning(f"Table extraction skipped: {exc}")
        return tables_by_page
