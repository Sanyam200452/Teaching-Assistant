"""
BM25 keyword store — classic TF-IDF-style retrieval.

Why keep BM25 alongside vectors?
  • Exact keyword matches (names, codes, acronyms) often beat semantic search
  • Hybrid > either alone — consistently 5-15% better recall in benchmarks
  • Zero API cost for keyword retrieval

pip install rank-bm25 nltk
"""

from __future__ import annotations
import pickle
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi  # type: ignore
from src.ingestion.chunker import Chunk

# Download NLTK data quietly on first run
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class BM25Store:
    INDEX_PATH = Path("./bm25_index.pkl")

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        if self.INDEX_PATH.exists():
            self._load()

    def index(self, chunks: list[Chunk]):
        """Build or rebuild BM25 index from chunks."""
        self.chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._save()

    def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        if self.bm25 is None or not self.chunks:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            Chunk(text=self.chunks[i].text, metadata=self.chunks[i].metadata.copy(), score=s)
            for i, s in ranked
            if s > 0
        ]

    # ------------------------------------------------------------------ #

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text.lower())

    def _save(self):
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)

    def _load(self):
        with open(self.INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.bm25 = data["bm25"]
