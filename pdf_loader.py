"""PDF download and loading utilities.

Downloads reports once and caches them locally so that subsequent runs
skip the network round-trip entirely.
"""

import requests
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from config import PDFS_DIR


def download_pdf(url: str, file_name: str) -> Path:
    """Download a PDF if not already cached and return the local path.

    Raises ``RuntimeError`` on network or HTTP errors so the caller can
    decide whether to abort or skip the document.
    """
    local_path = PDFS_DIR / file_name
    if local_path.exists():
        print(f"  Using cached: {file_name}")
        return local_path

    print(f"  Downloading: {file_name} ...")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {file_name}: {exc}") from exc

    local_path.write_bytes(response.content)
    print(f"  Saved: {file_name}")
    return local_path


def load_documents(company: dict) -> list:
    """Download and load all PDFs for a company.

    Each page is tagged with its source file name so that later stages
    can attribute evidence back to the correct document.
    """
    all_docs = []
    for doc_info in company["documents"]:
        pdf_path = download_pdf(doc_info["url"], doc_info["file_name"])
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = doc_info["file_name"]
        all_docs.extend(pages)
        print(f"  Loaded {len(pages)} pages from {doc_info['file_name']}")
    return all_docs
