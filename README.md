# ESG Report Analyzer

A RAG-based GenAI application that summarises company ESG and annual reports
and answers predefined questions using only the content of the provided documents.

## What it does

- Downloads PDF reports for **Tallink Grupp** and **Eesti Energia**
- Splits documents into chunks and indexes them in a **FAISS** vector store
- Uses **Retrieval-Augmented Generation (RAG)** to answer 5 predefined questions per company
- Every answer includes a source page number, supporting quote, and confidence level
- If the document does not contain enough information, returns `not_found` — no hallucination
- Saves all results to `output/results.json`
- Displays results in a **Streamlit** UI with a **live chat interface** for follow-up questions

## Tech stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector store | FAISS (local, in-memory) |
| RAG framework | LangChain |
| PDF parsing | PyPDFLoader |
| UI | Streamlit |
| Package manager | UV |
| Output format | JSON (Pydantic-validated) |
| Testing | pytest |

## Project structure

```
esg-analyzer/
├── main.py           # Entry point — runs the full pipeline
├── config.py         # All constants: questions, companies, model settings
├── models.py         # Pydantic models for structured output
├── pdf_loader.py     # PDF download + loading with caching
├── vector_store.py   # Document chunking + FAISS indexing
├── prompts.py        # LLM prompt templates
├── rag.py            # Core RAG logic: retrieval, parsing, chat
├── pipeline.py       # Orchestration: ties all stages together
├── app.py            # Streamlit UI with chat interface
├── pyproject.toml    # UV dependency management
├── tests/            # Test suite
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_pdf_loader.py
│   ├── test_rag.py
│   └── test_pipeline.py
├── pdfs/             # Cached PDFs (gitignored)
└── output/           # Generated JSON (gitignored)
```

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd esg-analyzer
   ```

2. Install dependencies with UV:
   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
   Get a free key at https://aistudio.google.com

## Running

**Step 1** — Generate the JSON analysis (downloads PDFs, runs RAG):
```bash
uv run python main.py
```
First run takes ~5 minutes (PDF download + embedding). Subsequent runs are faster (PDFs cached).

**Step 2** — Launch the UI:
```bash
uv run streamlit run app.py
```
Opens at http://localhost:8501

The UI has three tabs:
- **Tallink Grupp** — predefined Q&A results with sources
- **Eesti Energia** — predefined Q&A results with sources
- **💬 Chat** — ask follow-up questions about any company's reports in real time

**Step 3** — Run the test suite:
```bash
uv run pytest tests/ -v
```

## Reports analysed

| Company | Document | Year |
|---|---|---|
| Tallink Grupp | Tallink-Grupp-Sustainability-Report-2024-ENG-updated.pdf | 2024 |
| Eesti Energia | eesti-energia-2025-final-en.pdf | 2025 |
| Eesti Energia | Eesti-SPO-UoP.pdf | 2025 |

## Output format

`output/results.json` structure:

```json
{
  "generated_at": "2026-05-06T...",
  "model": { "provider": "Google", "name": "gemini-2.5-flash" },
  "companies": [
    {
      "company_name": "Tallink Grupp",
      "document": { "file_name": "...", "report_year": "2024", "source_url": "..." },
      "summary": {
        "answer": "...",
        "sources": [{ "page": 3, "quote": "...", "document": "..." }]
      },
      "questions": [
        {
          "question": "...",
          "status": "answered",
          "answer": "...",
          "confidence": "high",
          "sources": [{ "page": 5, "quote": "...", "document": "..." }],
          "missing_information": null
        }
      ]
    }
  ]
}
```

## Design decisions

**Why RAG over fine-tuning?**
The documents change regularly (new annual reports each year). RAG allows updating the knowledge base by simply adding new PDFs — no retraining required. Every answer is also traceable to a source page, which is critical in a regulated environment.

**Why Gemini 2.5 Flash?**
Free tier, sufficient context window (1M tokens), strong document understanding, and thinking capabilities. For production in a bank, Azure OpenAI would be preferred to keep data within a controlled cloud environment and meet regulatory data residency requirements.

**Why FAISS over a hosted vector DB?**
Keeps the project self-contained with no external services. For production scale, Azure AI Search or Pinecone would be more appropriate.

**Why low temperature (0.1)?**
ESG and financial reports require factual, reproducible answers. Higher temperature introduces unnecessary variation and increases hallucination risk.

**Why modular file structure?**
Each module has a single responsibility — configuration, models, loading, indexing, prompts, RAG logic, orchestration. This makes the codebase easier to navigate, test, and extend without touching unrelated code.

## Main assumptions

- Reports are publicly accessible at the provided URLs.
- The LLM API is available and the free tier quota is sufficient.
- English-language reports are used for analysis.
- PDF text extraction via PyPDFLoader produces usable text (not scanned images).

## Current limitations

- No OCR support — scanned PDFs will not be processed.
- Chat history is not persisted between Streamlit sessions.
- No incremental updates — changing a company requires re-running the full pipeline.
- The structured answer parser (`parse_answer`) relies on the LLM following the exact format.

## Possible improvements

- Add **LangGraph** for explicit pipeline orchestration and retry logic
- Use a **reranker** model to improve retrieval precision (hybrid sparse + dense search)
- Add **RAGAS** evaluation metrics (faithfulness, context recall, answer relevancy)
- Stream progress to the UI during generation
- Persist chat history to disk or database
- Switch to **Azure OpenAI** for production use in a banking environment

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI Studio API key |
