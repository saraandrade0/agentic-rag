"""Hybrid search combining ChromaDB (semantic) and BM25 (lexical)."""

import os
from typing import List

import chromadb
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "documents"

# Weights for hybrid search
SEMANTIC_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3


class HybridSearcher:
    """Combines dense (ChromaDB) and sparse (BM25) retrieval."""

    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._corpus_docs = None

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in ChromaDB."""
        results = self.collection.get(include=["documents", "metadatas"])
        if not results["documents"]:
            self._bm25 = None
            self._corpus_docs = []
            return

        self._corpus_docs = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        tokenized = [doc["content"].lower().split() for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized)

    def semantic_search(self, query: str, k: int = 10) -> List[dict]:
        """Dense retrieval via ChromaDB embeddings."""
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        for i, doc in enumerate(results["documents"][0]):
            score = 1 - results["distances"][0][i]  # cosine similarity
            docs.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "semantic_score": float(score),
            })
        return docs

    def lexical_search(self, query: str, k: int = 10) -> List[dict]:
        """Sparse retrieval via BM25."""
        if self._bm25 is None:
            self._build_bm25_index()

        if not self._corpus_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:k]
        docs = []
        for idx in top_indices:
            if scores[idx] > 0:
                docs.append({
                    "content": self._corpus_docs[idx]["content"],
                    "metadata": self._corpus_docs[idx]["metadata"],
                    "lexical_score": float(scores[idx]),
                })
        return docs

    def hybrid_search(self, query: str, k: int = 5) -> List[dict]:
        """Merge semantic and lexical results with weighted scoring.

        Uses reciprocal rank fusion (RRF) to combine both result sets.
        """
        semantic_results = self.semantic_search(query, k=k * 2)
        lexical_results = self.lexical_search(query, k=k * 2)

        scored = {}

        for rank, doc in enumerate(semantic_results):
            key = doc["content"][:100]  # dedup key
            rrf_score = SEMANTIC_WEIGHT / (rank + 1)
            scored[key] = {
                **doc,
                "hybrid_score": rrf_score,
            }

        for rank, doc in enumerate(lexical_results):
            key = doc["content"][:100]
            rrf_score = LEXICAL_WEIGHT / (rank + 1)
            if key in scored:
                scored[key]["hybrid_score"] += rrf_score
                scored[key]["lexical_score"] = doc.get("lexical_score", 0)
            else:
                scored[key] = {
                    **doc,
                    "hybrid_score": rrf_score,
                }

        ranked = sorted(scored.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:k]


# Module-level instance
searcher = HybridSearcher()


def search(query: str, k: int = 5) -> List[dict]:
    """Run hybrid search. Entry point for the agent."""
    return searcher.hybrid_search(query, k=k)
