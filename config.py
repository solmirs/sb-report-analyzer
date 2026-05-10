"""Centralised configuration for the ESG Report Analyzer.

Every tunable value lives here so that the rest of the codebase can import
a single module instead of scattering magic strings across files.
"""

from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────

PDFS_DIR = Path("pdfs")
OUTPUT_DIR = Path("output")

# ── Model settings ────────────────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMPERATURE = 0.1  # Low temperature → factual, reproducible answers

# ── Chunking settings ─────────────────────────────────────────────────────────

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200  # Overlap prevents information loss at chunk boundaries

# ── Retrieval settings ────────────────────────────────────────────────────────

RETRIEVAL_K = 5        # Top-k chunks per question
SUMMARY_K = 3          # Top-k chunks per summary seed query
SUMMARY_MAX_CHUNKS = 8 # Max unique chunks fed to the summary prompt

# ── Predefined questions ─────────────────────────────────────────────────────

QUESTIONS = [
    "What is the company's purpose and main activities?",
    "What is the company's primary focus?",
    "What is the company's commitment to risk management?",
    "What is the company's commitment to energy management?",
    "What is the company's commitment to data security?",
]

# ── Companies and their source documents ──────────────────────────────────────

COMPANIES = [
    {
        "company_name": "Tallink Grupp",
        "documents": [
            {
                "file_name": "Tallink-Grupp-Sustainability-Report-2024-ENG-updated.pdf",
                "url": "https://image.tallink.com/image/upload/grupp/documents/sustainability-reports/Tallink-Grupp-Sustainability-Report-2024-ENG-updated.pdf",
                "report_year": "2024",
            }
        ],
    },
    {
        "company_name": "Eesti Energia",
        "documents": [
            {
                "file_name": "eesti-energia-2025-final-en.pdf",
                "url": "https://public-docs.enefit.com/ettevottest/investorile/eesti-energia-2025-final-en.pdf",
                "report_year": "2025",
            },
            {
                "file_name": "Eesti-SPO-UoP.pdf",
                "url": "https://public-docs.enefit.ee/ettevottest/investorile/ESG/Eesti-SPO-UoP.pdf",
                "report_year": "2025",
            },
        ],
    },
]
