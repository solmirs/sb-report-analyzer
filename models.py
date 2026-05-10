"""Pydantic models for structured RAG output.

These models enforce the JSON schema required by the assignment:
every answer carries evidence (page, quote, document) and a confidence level.
"""

from typing import Optional
from pydantic import BaseModel


class Source(BaseModel):
    """A single piece of evidence linking an answer back to the source PDF."""
    page: Optional[int] = None
    quote: str
    document: str


class QuestionAnswer(BaseModel):
    """Structured response for one predefined question.

    When ``status`` is ``"answered"`` the ``answer`` and ``sources`` fields
    are populated.  When ``status`` is ``"not_found"`` only
    ``missing_information`` explains why the document could not answer the
    question — the application never invents unsupported claims.
    """
    question: str
    status: str  # "answered" | "not_found"
    answer: Optional[str] = None
    confidence: str  # "high" | "medium" | "low"
    sources: list[Source] = []
    missing_information: Optional[str] = None
