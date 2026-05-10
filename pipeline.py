"""Pipeline orchestration — ties all stages together.

This module owns the top-level workflow: for each company it downloads
documents, builds a vector store, generates a summary, answers the
predefined questions, and writes the final JSON output.
"""

import os
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    COMPANIES, QUESTIONS, MODEL_NAME, TEMPERATURE,
    PDFS_DIR, OUTPUT_DIR,
)
from pdf_loader import load_documents
from vector_store import build_vector_store
from rag import answer_question, generate_summary

load_dotenv()


def get_llm() -> ChatGoogleGenerativeAI:
    """Create a configured LLM instance."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=TEMPERATURE,
    )


def process_company(company: dict, llm) -> dict:
    """Run the full RAG pipeline for one company: load → index → summarise → answer."""
    print(f"\nProcessing: {company['company_name']}")

    documents = load_documents(company)

    print("  Building vector store...")
    vectorstore = build_vector_store(documents)

    print("  Generating summary...")
    summary = generate_summary(vectorstore, llm)

    question_results = []
    for question in QUESTIONS:
        print(f"  Q: {question[:55]}...")
        qa = answer_question(vectorstore, question, llm)
        question_results.append(qa.model_dump())

    return {
        "company_name": company["company_name"],
        "document": {
            "file_name": ", ".join(d["file_name"] for d in company["documents"]),
            "report_year": company["documents"][0]["report_year"],
            "source_url": ", ".join(d["url"] for d in company["documents"]),
        },
        "summary": summary,
        "questions": question_results,
    }


def run():
    """Main entry point — process all companies and save results to JSON."""
    print("ESG Report Analyzer")
    print("=" * 50)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Create a .env file with: GOOGLE_API_KEY=your_key_here"
        )

    # Ensure output directories exist
    PDFS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    llm = get_llm()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"provider": "Google", "name": MODEL_NAME},
        "companies": [],
    }

    for company in COMPANIES:
        results["companies"].append(process_company(company, llm))

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved to {output_path}")
