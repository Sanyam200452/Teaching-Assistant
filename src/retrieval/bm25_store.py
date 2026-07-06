"""
bm25_store.py — BM25 keyword index using LangChain BM25Retriever.

LangChain's BM25Retriever wraps rank-bm25 and fits into
EnsembleRetriever directly.

Pickle persistence is kept because:
  - Streamlit Cloud can't re-index at startup (chunks aren't available)
  - bm25_index.pkl is committed to the repo after local indexing
  - On startup, BM25Store loads the pkl and builds the LangChain retriever

pip install langchain-community rank-bm25 nltk
"""

from __future__ import annotations
import pickle
from pathlib import Path

import nltk
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.ingestion.chunker import Chunk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)


class BM25Store:
    INDEX_PATH = Path("./bm25_index.pkl")

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._retriever: BM25Retriever | None = None
        if self.INDEX_PATH.exists():
            self._load()

    # ------------------------------------------------------------------ #
    #  INDEXING                                                            #
    # ------------------------------------------------------------------ #

    def index(self, chunks: list[Chunk]):
        """Build BM25 index from chunks and save to disk."""
        self.chunks     = chunks
        self._retriever = self._build_retriever(chunks)
        self._save()

    # ------------------------------------------------------------------ #
    #  SEARCH — direct interface                                           #
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Search and return our Chunk objects."""
        if not self._retriever or not self.chunks:
            return []
        self._retriever.k = top_k
        lc_docs = self._retriever.invoke(query)
        return [
            Chunk(
                text=doc.page_content,
                metadata=doc.metadata.copy(),
                score=0.0,   # BM25Retriever doesn't expose raw scores
            )
            for doc in lc_docs
        ]

    # ------------------------------------------------------------------ #
    #  LANGCHAIN INTERFACE                                                 #
    # ------------------------------------------------------------------ #

    def as_langchain_retriever(self, top_k: int = 10) -> BM25Retriever:
        """Return LangChain BM25Retriever for use in EnsembleRetriever."""
        if self._retriever is None:
            raise RuntimeError("BM25Store not indexed yet. Run index() first.")
        self._retriever.k = top_k
        return self._retriever

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _build_retriever(self, chunks: list[Chunk]) -> BM25Retriever:
        lc_docs = [
            Document(page_content=c.text, metadata=c.metadata)
            for c in chunks
        ]
        return BM25Retriever.from_documents(lc_docs)

    def _save(self):
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks}, f)

    def _load(self):
        with open(self.INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks     = data["chunks"]
        self._retriever = self._build_retriever(self.chunks)
        print(f"  ✅ BM25 index loaded — {len(self.chunks)} chunks")
