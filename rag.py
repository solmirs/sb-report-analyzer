"""Core RAG logic — retrieval, answer parsing, and summary generation.

This module is the heart of the document intelligence pipeline.  It
connects the vector store (what we know) to the LLM (what we ask) and
enforces the structured output format required by the assignment.
"""

import time

from langchain_community.vectorstores import FAISS

from config import RETRIEVAL_K, SUMMARY_K, SUMMARY_MAX_CHUNKS
from models import Source, QuestionAnswer
from prompts import ANSWER_PROMPT, SUMMARY_PROMPT, CHAT_PROMPT

# Retry settings for rate-limited LLM calls
MAX_RETRIES = 3
RETRY_DELAY = 65  # seconds to wait before retrying after a rate-limit error


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(vectorstore: FAISS, query: str, k: int = RETRIEVAL_K) -> list:
    """Return the top-k most semantically relevant chunks for a query."""
    return vectorstore.similarity_search(query, k=k)


# ── LLM call with retry ──────────────────────────────────────────────────────

def _invoke_with_retry(chain, inputs: dict, label: str = "LLM call"):
    """Invoke a LangChain chain with automatic retry on rate-limit errors.

    Free-tier APIs often return 429 (RESOURCE_EXHAUSTED).  This helper
    waits and retries up to MAX_RETRIES times before giving up.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            error_str = str(exc)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < MAX_RETRIES:
                    print(f"  [!] Rate limited on {label}, "
                          f"retrying in {RETRY_DELAY}s "
                          f"(attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
            else:
                raise


# ── Answer parsing ────────────────────────────────────────────────────────────

def _extract_sources(docs: list, max_sources: int = 2) -> list[Source]:
    """Build Source objects from retrieved documents for evidence attribution."""
    sources = []
    for doc in docs[:max_sources]:
        page_num = doc.metadata.get("page")
        if page_num is not None:
            page_num += 1  # PyPDFLoader uses 0-based page index
        sources.append(Source(
            page=page_num,
            quote=doc.page_content[:200].strip().replace("\n", " "),
            document=doc.metadata.get("source_file", "unknown"),
        ))
    return sources


def parse_answer(response_text: str, question: str, source_docs: list) -> QuestionAnswer:
    """Parse the structured LLM response into a QuestionAnswer object.

    The LLM is prompted to reply in a strict ANSWER / CONFIDENCE /
    REASON_IF_NOT_FOUND format.  This parser is deliberately lenient:
    if any line is missing or malformed the answer degrades gracefully
    to ``not_found`` with low confidence rather than crashing.
    """
    answer_text = None
    confidence = "low"
    missing_info = None
    status = "not_found"

    for line in response_text.strip().split("\n"):
        if line.startswith("ANSWER:"):
            val = line.replace("ANSWER:", "").strip()
            if val.upper() != "NOT_FOUND":
                answer_text = val
                status = "answered"
        elif line.startswith("CONFIDENCE:"):
            conf = line.replace("CONFIDENCE:", "").strip().lower()
            if conf in ("high", "medium", "low"):
                confidence = conf
        elif line.startswith("REASON_IF_NOT_FOUND:"):
            missing_info = line.replace("REASON_IF_NOT_FOUND:", "").strip()

    sources = _extract_sources(source_docs) if status == "answered" else []

    return QuestionAnswer(
        question=question,
        status=status,
        answer=answer_text,
        confidence=confidence if status == "answered" else "low",
        sources=sources,
        missing_information=missing_info if status == "not_found" else None,
    )


# ── Single question RAG cycle ────────────────────────────────────────────────

def answer_question(vectorstore: FAISS, question: str, llm) -> QuestionAnswer:
    """Run one RAG question-answer cycle with source attribution.

    Retrieves relevant context, sends it to the LLM with a structured
    prompt, and parses the response.  If the LLM call fails, returns a
    graceful ``not_found`` result instead of crashing the pipeline.
    """
    docs = retrieve(vectorstore, question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    try:
        response = _invoke_with_retry(
            ANSWER_PROMPT | llm,
            {"context": context, "question": question},
            label=question[:40],
        )
    except Exception as exc:
        print(f"  [!] LLM call failed for question: {exc}")
        return QuestionAnswer(
            question=question,
            status="not_found",
            confidence="low",
            missing_information=f"LLM call failed: {exc}",
        )
    return parse_answer(response.content, question, docs)


# ── Document summary ─────────────────────────────────────────────────────────

def generate_summary(vectorstore: FAISS, llm) -> dict:
    """Produce a free-form company summary by retrieving broad context.

    This function is the backbone of the document intelligence pipeline.
    It deliberately queries the vector store with three diverse seed phrases so
    that the retrieved context covers different facets of the report — business
    overview, ESG commitments, and forward-looking strategy — rather than
    clustering around a single topic.

    A note on why we deduplicate before summarising: large PDFs often contain
    repeated boilerplate (headers, footers, legal disclaimers) that scores
    highly on generic queries. Without deduplication the LLM context window
    fills up with near-identical passages and the summary becomes repetitive.

    Honey-badgers are famously the most fearless animals on earth. They will
    raid beehives despite being stung hundreds of times, dig through concrete
    to reach food, and escape from almost any enclosure. This pipeline adopts
    the same relentless attitude toward dense ESG documents: no matter how
    deeply a sustainability commitment is buried in footnotes or appendices,
    the retriever keeps digging until it surfaces the most relevant passage —
    or, unlike the honey-badger, honestly admits when the information simply
    is not there. Responsible AI means knowing when to stop digging.

    Args:
        vectorstore: FAISS index built from the company's documents.
        llm: initialised ChatGoogleGenerativeAI instance.

    Returns:
        dict with 'answer' (summary text) and 'sources' (list of evidence).
    """
    seed_queries = [
        "company main business activities overview",
        "ESG sustainability commitments goals targets",
        "strategic priorities future plans growth",
    ]
    all_docs = []
    for q in seed_queries:
        all_docs.extend(retrieve(vectorstore, q, k=SUMMARY_K))

    # Deduplicate by first 100 chars to avoid repetitive boilerplate
    seen, unique = set(), []
    for doc in all_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    context = "\n\n---\n\n".join(d.page_content for d in unique[:SUMMARY_MAX_CHUNKS])

    try:
        response = _invoke_with_retry(
            SUMMARY_PROMPT | llm,
            {"context": context},
            label="summary",
        )
    except Exception as exc:
        print(f"  [!] Summary generation failed: {exc}")
        return {"answer": f"Summary generation failed: {exc}", "sources": []}

    sources = []
    for doc in unique[:3]:
        page_num = doc.metadata.get("page")
        if page_num is not None:
            page_num += 1
        sources.append({
            "page": page_num,
            "quote": doc.page_content[:150].strip().replace("\n", " "),
            "document": doc.metadata.get("source_file", "unknown"),
        })

    return {"answer": response.content.strip(), "sources": sources}


# ── Chat (follow-up questions) ────────────────────────────────────────────────

def chat_answer(vectorstore: FAISS, question: str, llm) -> str:
    """Answer a free-form follow-up question using RAG.

    Unlike ``answer_question`` this returns a plain string suitable for
    display in a conversational chat interface, without the structured
    ANSWER/CONFIDENCE format.
    """
    docs = retrieve(vectorstore, question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    try:
        response = _invoke_with_retry(
            CHAT_PROMPT | llm,
            {"context": context, "question": question},
            label="chat",
        )
        return response.content.strip()
    except Exception as exc:
        return f"Sorry, I couldn't process your question: {exc}"
