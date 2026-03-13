"""Ingest PDF documents into ChromaDB for vector search.

Usage:
    python scripts/ingest.py --pdf-dir data/documents/
    python scripts/ingest.py --pdf-dir data/documents/ --chunk-size 500
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "documents"

# Chunk settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def ingest(pdf_dir: str, chunk_size: int, chunk_overlap: int) -> None:
    """Main ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s). Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        print(f"  → {len(chunks)} chunks")

        if not chunks:
            continue

        embeddings = embedder.encode(chunks).tolist()

        ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": pdf_file.name, "chunk_index": i}
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"\nDone! Ingested {total_chunks} chunks from {len(pdf_files)} files.")
    print(f"ChromaDB persisted at: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--pdf-dir", required=True, help="Directory with PDF files")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    ingest(args.pdf_dir, args.chunk_size, args.chunk_overlap)
