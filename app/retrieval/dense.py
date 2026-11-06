"""Dense (vector) retrieval with optional company / year filters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    id: int
    company: str
    year: int
    page: int
    text: str
    score: float


def build_query(company: str | None, year: int | None) -> tuple[str, list]:
    """Return SQL with optional metadata filters. Separated out so it can be unit-tested."""
    where, params = [], []
    if company:
        where.append("company ILIKE %s")
        params.append(company)
    if year:
        where.append("year = %s")
        params.append(year)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
        SELECT id, company, year, page, text, 1 - (embedding <=> %s) AS score
        FROM chunks {clause}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    return sql, params


def search(
    query: str, k: int = 6, company: str | None = None, year: int | None = None
) -> list[Hit]:
    import numpy as np

    from app.db import connect
    from app.llm import embed

    qvec = np.array(embed([query])[0])
    sql, params = build_query(company, year)
    with connect() as conn, conn.cursor() as cur:
        # placeholder order: score vector, WHERE filters, ORDER BY vector, LIMIT
        cur.execute(sql, [qvec, *params, qvec, k])
        return [Hit(*row) for row in cur.fetchall()]
