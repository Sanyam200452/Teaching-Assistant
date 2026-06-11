"""
Reranker — cross-encoder scoring to pick the truly relevant chunks
from the hybrid retriever's candidates.

Two options (auto-detected by available API key):
  1. Cohere Rerank v3   — best quality, needs COHERE_API_KEY
  2. sentence-transformers cross-encoder  — free local fallback

Why rerank at all?
  Bi-encoder (embedding) models are fast but approximate.
  A cross-encoder sees (query, chunk) together → much better relevance.
  Pattern: retrieve 20 candidates, rerank to top 5. Big quality boost.
"""

from __future__ import annotations
import os
from src.ingestion.chunker import Chunk


class Reranker:
    def __init__(self):
        self.cohere_key = os.environ.get("COHERE_API_KEY")
        self._cross_encoder = None

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
        if self.cohere_key:
            return self._cohere_rerank(query, chunks, top_n)
        return self._local_rerank(query, chunks, top_n)

    # ------------------------------------------------------------------ #

    def _cohere_rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """pip install cohere"""
        import cohere  # type: ignore
        client = cohere.Client(self.cohere_key)

        response = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[c.text for c in chunks],
            top_n=top_n,
        )

        return [
            Chunk(
                text=chunks[r.index].text,
                metadata=chunks[r.index].metadata.copy(),
                score=r.relevance_score,
            )
            for r in response.results
        ]

    def _local_rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """
        Local cross-encoder fallback.
        pip install sentence-transformers
        First run downloads ~90 MB model.
        """
        from sentence_transformers import CrossEncoder  # type: ignore

        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, c.text) for c in chunks]
        scores = self._cross_encoder.predict(pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)[:top_n]
        return [
            Chunk(text=c.text, metadata=c.metadata.copy(), score=float(s))
            for s, c in ranked
        ]
