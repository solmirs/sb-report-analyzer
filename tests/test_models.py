"""Tests for models.py — Pydantic model validation."""

import pytest
from models import Source, QuestionAnswer


class TestSource:
    def test_basic_source(self):
        src = Source(page=5, quote="Some quote", document="report.pdf")
        assert src.page == 5
        assert src.quote == "Some quote"
        assert src.document == "report.pdf"

    def test_source_without_page(self):
        """Page is optional — some chunks don't carry page metadata."""
        src = Source(quote="Quote text", document="doc.pdf")
        assert src.page is None

    def test_source_serialization(self):
        src = Source(page=10, quote="Test", document="file.pdf")
        d = src.model_dump()
        assert d == {"page": 10, "quote": "Test", "document": "file.pdf"}


class TestQuestionAnswer:
    def test_answered_question(self):
        qa = QuestionAnswer(
            question="What is the purpose?",
            status="answered",
            answer="The company does X.",
            confidence="high",
            sources=[Source(page=1, quote="X", document="a.pdf")],
        )
        assert qa.status == "answered"
        assert qa.answer is not None
        assert qa.missing_information is None
        assert len(qa.sources) == 1

    def test_not_found_question(self):
        qa = QuestionAnswer(
            question="What about data security?",
            status="not_found",
            confidence="low",
            missing_information="No data security section found.",
        )
        assert qa.status == "not_found"
        assert qa.answer is None
        assert qa.missing_information is not None
        assert qa.sources == []

    def test_serialization_round_trip(self):
        qa = QuestionAnswer(
            question="Test?",
            status="answered",
            answer="Yes",
            confidence="medium",
            sources=[Source(page=3, quote="Q", document="b.pdf")],
        )
        d = qa.model_dump()
        qa2 = QuestionAnswer(**d)
        assert qa == qa2

    def test_defaults(self):
        """Minimal construction — only required fields."""
        qa = QuestionAnswer(question="Q?", status="not_found", confidence="low")
        assert qa.answer is None
        assert qa.sources == []
        assert qa.missing_information is None

    def test_invalid_missing_required_field(self):
        """Pydantic should reject missing required fields."""
        with pytest.raises(Exception):
            QuestionAnswer(question="Q?")  # missing status, confidence
