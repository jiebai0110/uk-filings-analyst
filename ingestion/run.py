"""Ingest every PDF listed in the manifest: parse -> clean -> chunk -> embed -> store.

Usage:
    python -m ingestion.run --manifest data/manifest.csv --pdf-dir data/pdfs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.config import get_settings
from ingestion.chunk import chunk_pages
from ingestion.parse import parse_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--pdf-dir", default="data/pdfs")
    ap.add_argument("--dry-run", action="store_true", help="parse and chunk only, no API calls")
    args = ap.parse_args()

    s = get_settings()
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Manifest is empty: add one row per annual report first.")

    total = 0
    for row in rows:
        path = Path(args.pdf_dir) / row["filename"]
        if not path.exists():
            print(f"skip (missing file): {path}")
            continue
        pages = parse_pdf(path, row["company"], int(row["year"]))
        chunks = chunk_pages(pages, s.chunk_tokens, s.chunk_overlap)
        print(f"{row['company']} {row['year']}: {len(pages)} pages -> {len(chunks)} chunks")
        if args.dry_run:
            continue

        from app.db import connect, insert_chunks
        from app.llm import embed

        vectors = embed([c.text for c in chunks])
        with connect() as conn:
            total += insert_chunks(conn, chunks, vectors)
    print(f"done, {total} chunks stored")


if __name__ == "__main__":
    main()
