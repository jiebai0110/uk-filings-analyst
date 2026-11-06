"""Postgres + pgvector access."""

from __future__ import annotations

from contextlib import contextmanager

from app.config import get_settings


@contextmanager
def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(get_settings().database_url) as conn:
        register_vector(conn)
        yield conn


def insert_chunks(conn, chunks, vectors) -> int:
    import numpy as np

    rows = [
        (c.company, c.year, c.page, c.chunk_index, c.source_file, c.text, np.array(v))
        for c, v in zip(chunks, vectors, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (company, year, page, chunk_index, source_file, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_file, page, chunk_index) DO UPDATE
              SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
            """,
            rows,
        )
    conn.commit()
    return len(rows)
