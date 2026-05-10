"""Streamlit UI for reviewing ESG analysis results and chatting with reports.

Run:  uv run streamlit run app.py
"""

import json

import streamlit as st
from dotenv import load_dotenv

from config import OUTPUT_DIR, COMPANIES

load_dotenv()

OUTPUT_PATH = OUTPUT_DIR / "results.json"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="ESG Analyzer", page_icon="📊", layout="wide")
st.title("📊 ESG Report Analyzer")
st.caption("Powered by Gemini 2.5 Flash + LangChain RAG")

# ── Load results ──────────────────────────────────────────────────────────────

if not OUTPUT_PATH.exists():
    st.warning(
        "No results yet. Run `uv run python main.py` first to generate the analysis."
    )
    st.stop()

with open(OUTPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

st.caption(
    f"Generated: {data['generated_at']}  |  "
    f"Model: {data['model']['provider']} / {data['model']['name']}"
)
st.divider()

# ── Helpers ───────────────────────────────────────────────────────────────────

CONFIDENCE_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}


@st.cache_resource
def _get_vectorstore(company_name: str):
    """Build and cache a FAISS vector store for the selected company.

    Uses Streamlit's ``cache_resource`` so the (expensive) embedding step
    only runs once per company per session.
    """
    from config import PDFS_DIR
    from pdf_loader import load_documents
    from vector_store import build_vector_store

    company = next(
        (c for c in COMPANIES if c["company_name"] == company_name), None
    )
    if company is None:
        return None

    PDFS_DIR.mkdir(exist_ok=True)
    docs = load_documents(company)
    return build_vector_store(docs)


# ── Display results for each company ─────────────────────────────────────────

tabs = st.tabs(
    [company["company_name"] for company in data["companies"]] + ["💬 Chat"]
)

for i, company in enumerate(data["companies"]):
    with tabs[i]:
        st.header(company["company_name"])
        st.caption(
            f"📄 {company['document']['file_name']}  "
            f"({company['document']['report_year']})"
        )

        # Summary
        with st.expander("📝 Document Summary", expanded=True):
            st.write(company["summary"]["answer"])
            if company["summary"].get("sources"):
                st.markdown("**Sources used:**")
                for src in company["summary"]["sources"]:
                    st.caption(
                        f"p.{src.get('page', '?')} — {src['document']}: "
                        f"\"{src['quote']}...\""
                    )

        # Questions
        st.subheader("Predefined Questions")
        for qa in company["questions"]:
            status = qa.get("status", "not_found")
            confidence = qa.get("confidence", "low")
            icon = "✅" if status == "answered" else "❌"
            conf_icon = CONFIDENCE_ICON.get(confidence, "⚪")

            with st.expander(f"{icon} {qa['question']}"):
                if status == "answered":
                    st.write(qa["answer"])
                    st.markdown(
                        f"**Confidence:** {conf_icon} {confidence.capitalize()}"
                    )
                    if qa.get("sources"):
                        st.markdown("**Supporting evidence:**")
                        for src in qa["sources"]:
                            st.caption(
                                f"p.{src.get('page', '?')} — {src['document']}: "
                                f"\"{src['quote']}...\""
                            )
                else:
                    st.warning("Not found in document.")
                    if qa.get("missing_information"):
                        st.caption(f"Missing: {qa['missing_information']}")

# ── Chat tab ──────────────────────────────────────────────────────────────────

with tabs[-1]:
    st.header("💬 Chat with Reports")
    st.caption(
        "Ask follow-up questions about any company's ESG reports. "
        "Answers are grounded in the source documents."
    )

    # Company selector
    company_names = [c["company_name"] for c in COMPANIES]
    selected_company = st.selectbox("Select company:", company_names)

    # Initialise session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about the reports...")

    if user_question:
        # Show user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.write(user_question)

        # Build vector store for the selected company (cached)
        with st.spinner("Searching documents..."):
            vectorstore = _get_vectorstore(selected_company)
            if vectorstore is None:
                response = "Could not load documents for this company."
            else:
                from rag import chat_answer
                from pipeline import get_llm

                llm = get_llm()
                response = chat_answer(vectorstore, user_question, llm)

        # Show assistant message
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant"):
            st.write(response)
