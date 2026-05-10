"""Vector store construction with FAISS.

Splits documents into overlapping chunks and indexes them for semantic
similarity search. Overlap between chunks ensures that information
sitting at a page or section boundary is not silently dropped.

Uses local HuggingFace embeddings for fast, rate-limit-free processing.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def get_embeddings() -> HuggingFaceEmbeddings:
    """Create a local embeddings instance using the configured model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(documents: list) -> FAISS:
    """Chunk documents and build a FAISS index locally."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    embeddings = get_embeddings()
    print("  Embedding locally (no rate limits)...")
    
    return FAISS.from_documents(chunks, embeddings)
