"""Smoke tests for pipeline.py — verifies imports and structure without API calls."""

import pytest
from unittest.mock import patch


def test_pipeline_importable():
    """Verify that pipeline.py and all its dependencies can be imported."""
    from pipeline import run, process_company, get_llm
    assert callable(run)
    assert callable(process_company)
    assert callable(get_llm)


def test_run_fails_without_api_key():
    """Without GOOGLE_API_KEY the pipeline should raise ValueError immediately."""
    with patch.dict("os.environ", {"GOOGLE_API_KEY": ""}, clear=False):
        from pipeline import run
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            run()


def test_all_modules_importable():
    """Verify there are no circular imports across the project."""
    import config
    import models
    import pdf_loader
    import vector_store
    import prompts
    import rag
    import pipeline

    # Basic sanity — each module has the expected top-level names
    assert hasattr(config, "QUESTIONS")
    assert hasattr(models, "Source")
    assert hasattr(models, "QuestionAnswer")
    assert hasattr(pdf_loader, "download_pdf")
    assert hasattr(pdf_loader, "load_documents")
    assert hasattr(vector_store, "build_vector_store")
    assert hasattr(prompts, "ANSWER_PROMPT")
    assert hasattr(prompts, "SUMMARY_PROMPT")
    assert hasattr(prompts, "CHAT_PROMPT")
    assert hasattr(rag, "retrieve")
    assert hasattr(rag, "parse_answer")
    assert hasattr(rag, "answer_question")
    assert hasattr(rag, "generate_summary")
    assert hasattr(rag, "chat_answer")
    assert hasattr(pipeline, "run")
