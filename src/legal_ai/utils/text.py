"""Text normalization helpers for legal documents."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SOFT_HYPHEN_RE = re.compile(r"-\n(\w)")
_PAGE_NUM_RE = re.compile(r"^\s*(?:Pag(?:ina)?|Page)\s*\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE)


def normalize_legal_text(raw: str) -> str:
    """Normalize whitespace, hyphenation and Unicode forms while keeping structure."""

    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = _SOFT_HYPHEN_RE.sub(r"\1", text)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = _WHITESPACE_RE.sub(" ", line).strip()
        if not stripped or _PAGE_NUM_RE.match(stripped):
            cleaned_lines.append("")
            continue
        cleaned_lines.append(stripped)
    joined = "\n".join(cleaned_lines)
    return _MULTI_NEWLINE_RE.sub("\n\n", joined).strip()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token) without external tokenizer."""

    if not text:
        return 0
    return max(1, len(text) // 4)
