"""Tests for rag.py — answer parsing and source extraction logic.

These tests run without any LLM or vector store — they validate the
pure parsing logic that converts raw LLM text into structured output.
"""

from unittest.mock import MagicMock
from rag import parse_answer, _extract_sources
from models import Source


# ── Helper: fake LangChain document ──────────────────────────────────────────

def _fake_doc(content: str, page: int = 0, source_file: str = "test.pdf"):
    """Create a mock document object similar to what PyPDFLoader returns."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"page": page, "source_file": source_file}
    return doc


# ── Tests for _extract_sources ────────────────────────────────────────────────

class TestExtractSources:
    def test_basic_extraction(self):
        docs = [_fake_doc("Hello world", page=4, source_file="report.pdf")]
        sources = _extract_sources(docs)
        assert len(sources) == 1
        assert sources[0].page == 5  # 0-based → 1-based
        assert sources[0].document == "report.pdf"

    def test_max_sources_limit(self):
        docs = [_fake_doc(f"Doc {i}", page=i) for i in range(10)]
        sources = _extract_sources(docs, max_sources=2)
        assert len(sources) == 2

    def test_page_none_handling(self):
        doc = MagicMock()
        doc.page_content = "No page info"
        doc.metadata = {"source_file": "x.pdf"}
        sources = _extract_sources([doc])
        assert sources[0].page is None

    def test_newlines_stripped_from_quotes(self):
        doc = _fake_doc("Line one\nLine two\nLine three")
        sources = _extract_sources([doc])
        assert "\n" not in sources[0].quote


# ── Tests for parse_answer ────────────────────────────────────────────────────

class TestParseAnswer:
    def test_successful_answer(self):
        response = (
            "ANSWER: The company focuses on maritime transport.\n"
            "CONFIDENCE: high\n"
            "REASON_IF_NOT_FOUND:"
        )
        docs = [_fake_doc("Maritime operations...", page=2)]
        qa = parse_answer(response, "What is the focus?", docs)
        assert qa.status == "answered"
        assert qa.answer == "The company focuses on maritime transport."
        assert qa.confidence == "high"
        assert len(qa.sources) > 0
        assert qa.missing_information is None

    def test_not_found_answer(self):
        response = (
            "ANSWER: NOT_FOUND\n"
            "CONFIDENCE: low\n"
            "REASON_IF_NOT_FOUND: No data security section in the report."
        )
        docs = [_fake_doc("Unrelated content")]
        qa = parse_answer(response, "Data security?", docs)
        assert qa.status == "not_found"
        assert qa.answer is None
        assert qa.confidence == "low"
        assert qa.sources == []
        assert "data security" in qa.missing_information.lower()

    def test_malformed_response_degrades_gracefully(self):
        """If LLM returns garbage, we get not_found instead of a crash."""
        response = "This is not the expected format at all."
        docs = [_fake_doc("Some content")]
        qa = parse_answer(response, "Some question?", docs)
        assert qa.status == "not_found"
        assert qa.confidence == "low"

    def test_confidence_medium(self):
        response = (
            "ANSWER: Partial info available.\n"
            "CONFIDENCE: medium\n"
        )
        docs = [_fake_doc("Content")]
        qa = parse_answer(response, "Question?", docs)
        assert qa.confidence == "medium"

    def test_invalid_confidence_defaults_to_low(self):
        response = (
            "ANSWER: Something.\n"
            "CONFIDENCE: super_high\n"
        )
        docs = [_fake_doc("Content")]
        qa = parse_answer(response, "Q?", docs)
        # "super_high" is not valid, so it stays at default "low"
        assert qa.confidence == "low"

    def test_case_insensitive_not_found(self):
        """NOT_FOUND check should be case-insensitive."""
        response = "ANSWER: not_found\nCONFIDENCE: low\n"
        docs = [_fake_doc("Content")]
        qa = parse_answer(response, "Q?", docs)
        assert qa.status == "not_found"
