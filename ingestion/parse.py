"""PDF parsing: one record per page, with simple, documented cleaning rules."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    company: str
    year: int
    page: int  # 1-based, matches the page number a human sees in a PDF viewer
    text: str
    source_file: str


_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,3}\s*$", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t]+")


def clean_page_text(text: str, repeated_lines: set[str] | None = None) -> str:
    """Remove page numbers, repeated headers/footers and noisy whitespace.

    Rules (kept deliberately simple so failures are easy to explain):
    1. drop lines that are only a page number ("12", "Page 12")
    2. drop lines that repeat on many pages (running headers/footers)
    3. collapse runs of spaces and blank lines
    """
    repeated_lines = repeated_lines or set()
    kept = []
    for raw in text.splitlines():
        line = _WHITESPACE.sub(" ", raw).strip()
        if not line or _PAGE_NUMBER.match(line) or line in repeated_lines:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def find_repeated_lines(page_texts: list[str], min_share: float = 0.5) -> set[str]:
    """Lines that appear on at least `min_share` of pages are treated as headers/footers."""
    if len(page_texts) < 4:
        return set()
    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = {_WHITESPACE.sub(" ", ln).strip() for ln in text.splitlines()}
        counts.update(ln for ln in lines if ln)
    threshold = max(2, int(len(page_texts) * min_share))
    return {line for line, n in counts.items() if n >= threshold and len(line) < 120}


def parse_pdf(path: Path, company: str, year: int) -> list[Page]:
    import fitz  # PyMuPDF; imported lazily so unit tests don't need it

    with fitz.open(path) as doc:
        raw = [page.get_text("text") for page in doc]
    repeated = find_repeated_lines(raw)
    pages = []
    for i, text in enumerate(raw, start=1):
        cleaned = clean_page_text(text, repeated)
        if len(cleaned) >= 50:  # skip near-empty pages (covers, image-only pages)
            pages.append(Page(company, year, i, cleaned, path.name))
    return pages
