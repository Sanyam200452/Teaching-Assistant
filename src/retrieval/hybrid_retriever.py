"""
Hybrid Retriever — combines vector and BM25 results using
Reciprocal Rank Fusion (RRF).

RRF score formula:  sum(1 / (k + rank_i))  for each result list i
Default k=60 is standard from the original RRF paper (Cormack 2009).

Why RRF over weighted average?
  • No calibration needed — rank positions are scale-invariant
  • Robust to score distribution differences between systems
  • Consistently outperforms linear combination on most benchmarks
"""

from __future__ import annotations
from collections import defaultdict
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> list[Chunk]:
        """
        1. Embed query → vector search
        2. Tokenize query → BM25 search
        3. Fuse both ranked lists with RRF
        """
        # Vector search
        query_vec = self.embedder.embed_one(query)
        vector_hits = self.vector_store.search(query_vec, top_k=top_k)

        # BM25 search
        bm25_hits = self.bm25_store.search(query, top_k=top_k)

        # RRF fusion
        return self._rrf_fuse([vector_hits, bm25_hits], top_k=top_k)

    # ------------------------------------------------------------------ #

    def _rrf_fuse(self, ranked_lists: list[list[Chunk]], top_k: int) -> list[Chunk]:
        """Merge N ranked lists into one using Reciprocal Rank Fusion."""
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                key = chunk.text[:200]  # dedup key (first 200 chars)
                rrf_scores[key] += 1.0 / (self.rrf_k + rank)
                if key not in chunk_map:
                    chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        result = []
        for key in sorted_keys[:top_k]:
            chunk = chunk_map[key]
            chunk.score = rrf_scores[key]
            result.append(chunk)
        return result
