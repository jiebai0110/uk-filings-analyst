import pytest

from ingestion.chunk import approx_tokens, chunk_page
from ingestion.parse import Page, clean_page_text, find_repeated_lines


def make_page(text: str, page: int = 3) -> Page:
    return Page(company="Tesco", year=2024, page=page, text=text, source_file="tesco_2024.pdf")


def test_clean_removes_page_numbers_and_headers():
    raw = "Tesco PLC Annual Report 2024\nGroup revenue was £68.2bn.\n\n\n  12  \nPage 13"
    cleaned = clean_page_text(raw, repeated_lines={"Tesco PLC Annual Report 2024"})
    assert cleaned == "Group revenue was £68.2bn."


def test_repeated_lines_detected_across_pages():
    pages = [f"Strategic report\nBody text {i}" for i in range(6)]
    assert "Strategic report" in find_repeated_lines(pages)
    assert not any(line.startswith("Body text") for line in find_repeated_lines(pages))


def test_chunks_stay_within_one_page_and_keep_metadata():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_page(make_page(text, page=7), max_tokens=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.page == 7 and c.company == "Tesco" and c.year == 2024 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(approx_tokens(c.text) <= 200 for c in chunks)


def test_chunks_overlap_and_cover_all_words():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_page(make_page(" ".join(words)), max_tokens=130, overlap=26)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert set(first) & set(second), "consecutive chunks should overlap"
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(words)


def test_short_page_is_single_chunk():
    chunks = chunk_page(make_page("Revenue grew 4%."), max_tokens=512, overlap=64)
    assert len(chunks) == 1 and chunks[0].text == "Revenue grew 4%."


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_page(make_page("a b c"), max_tokens=10, overlap=10)
