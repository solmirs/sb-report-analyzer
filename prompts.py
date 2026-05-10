"""Prompt templates for the RAG pipeline.

Keeping prompts in a dedicated file makes them easy to find, tune, and
version-control independently from the retrieval and parsing logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Question-answering prompt ─────────────────────────────────────────────────

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an analyst reviewing company ESG and annual reports.
Answer the question using ONLY the context below.
If the context lacks enough information, respond with NOT_FOUND.

Context:
{context}

Question: {question}

Reply in exactly this format:
ANSWER: <your answer, or NOT_FOUND>
CONFIDENCE: <high / medium / low>
REASON_IF_NOT_FOUND: <only if not found — what information was missing>
""")

# ── Document summary prompt ───────────────────────────────────────────────────

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are an analyst reviewing a company ESG and annual report.
Write a concise 3-5 sentence summary covering:
- The company's main business
- Key ESG themes and commitments
- Strategic priorities

Context:
{context}

Summary:
""")

# ── Chat prompt (follow-up questions) ─────────────────────────────────────────

CHAT_PROMPT = ChatPromptTemplate.from_template("""
You are an analyst reviewing company ESG and annual reports.
Answer the user's question using ONLY the context below.
If the context does not contain enough information, say so honestly.
Be concise and cite page numbers where possible.

Context:
{context}

Question: {question}

Answer:
""")
