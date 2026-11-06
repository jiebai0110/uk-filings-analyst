"""Token-window chunking that never crosses a page boundary.

Keeping each chunk inside one page means every retrieved chunk maps to exactly one
page number, which is what makes page-level citations (and Recall@k against the
golden set's `source_pages`) possible.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ingestion.parse import Page

_TOKEN = re.compile(r"\S+")


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per word for English). Good enough for chunk sizing."""
    return int(len(_TOKEN.findall(text)) * 1.3)


@dataclass
class Chunk:
    company: str
    year: int
    page: int
    chunk_index: int
    source_file: str
    text: str


def chunk_page(
    page: Page,
    max_tokens: int = 512,
    overlap: int = 64,
    count: Callable[[str], int] = approx_tokens,
) -> list[Chunk]:
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    words = page.text.split(" ")
    # convert token budgets into word budgets using the same ratio as the counter
    ratio = max(count(page.text), 1) / max(len(words), 1)
    win = max(1, int(max_tokens / ratio))
    step = max(1, win - int(overlap / ratio))

    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        text = " ".join(words[start : start + win]).strip()
        if text:
            chunks.append(
                Chunk(page.company, page.year, page.page, len(chunks), page.source_file, text)
            )
        if start + win >= len(words):
            break
        start += step
    return chunks


def chunk_pages(pages: list[Page], max_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        out.extend(chunk_page(page, max_tokens, overlap))
    return out
