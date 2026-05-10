"""Tests for config.py — validate that all configuration values are sane."""

from config import (
    QUESTIONS, COMPANIES, PDFS_DIR, OUTPUT_DIR,
    MODEL_NAME, EMBEDDING_MODEL, TEMPERATURE,
    CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K,
)


def test_questions_not_empty():
    """The assignment requires exactly 5 predefined questions."""
    assert len(QUESTIONS) == 5


def test_questions_are_strings():
    for q in QUESTIONS:
        assert isinstance(q, str)
        assert len(q) > 10, f"Question too short: {q}"


def test_companies_count():
    """Assignment specifies Tallink Grupp and Eesti Energia."""
    assert len(COMPANIES) == 2


def test_company_structure():
    """Each company must have a name and at least one document with url/file_name."""
    for company in COMPANIES:
        assert "company_name" in company
        assert "documents" in company
        assert len(company["documents"]) >= 1
        for doc in company["documents"]:
            assert "file_name" in doc
            assert "url" in doc
            assert doc["url"].startswith("https://")
            assert "report_year" in doc


def test_company_names():
    names = [c["company_name"] for c in COMPANIES]
    assert "Tallink Grupp" in names
    assert "Eesti Energia" in names


def test_directories_are_paths():
    assert PDFS_DIR.name == "pdfs"
    assert OUTPUT_DIR.name == "output"


def test_model_settings():
    assert "gemini" in MODEL_NAME.lower()
    assert isinstance(EMBEDDING_MODEL, str) and len(EMBEDDING_MODEL) > 0
    assert 0 <= TEMPERATURE <= 1


def test_chunk_settings():
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP > 0
    assert CHUNK_OVERLAP < CHUNK_SIZE, "Overlap must be smaller than chunk size"


def test_retrieval_k():
    assert RETRIEVAL_K > 0
