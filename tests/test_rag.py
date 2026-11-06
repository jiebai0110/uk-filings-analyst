from app.rag import extract_citations, format_sources
from app.retrieval.dense import Hit, build_query

HITS = [
    Hit(1, "Tesco", 2024, 10, "Group revenue was £68.2bn.", 0.9),
    Hit(2, "Tesco", 2023, 12, "Group revenue was £65.8bn.", 0.8),
]


def test_format_sources_numbers_and_pages():
    out = format_sources(HITS)
    assert out.startswith("[1] Tesco, annual report 2024, p.10")
    assert "[2] Tesco, annual report 2023, p.12" in out


def test_extract_citations_dedupes_and_drops_invalid_numbers():
    answer = "Revenue rose to £68.2bn [1] from £65.8bn [2]. Again [1]. Bogus [7]."
    cites = extract_citations(answer, HITS)
    assert [(c.n, c.year, c.page) for c in cites] == [(1, 2024, 10), (2, 2023, 12)]


def test_build_query_filters():
    sql, params = build_query("Tesco", 2024)
    assert "company ILIKE %s" in sql and "year = %s" in sql
    assert params == ["Tesco", 2024]
    sql, params = build_query(None, None)
    assert "WHERE" not in sql and params == []
