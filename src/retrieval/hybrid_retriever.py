"""
hybrid_retriever.py — LangChain EnsembleRetriever + Cohere reranker.

Pipeline:
  1. EnsembleRetriever   → fuses vector search + BM25 (weighted RRF)
  2. CohereRerank        → cross-encoder reranking via Cohere API
  3. ContextualCompressionRetriever → wraps both into one LangChain retriever

This is the standard LangChain production RAG retrieval pattern.

16-08-2025
Now supports metadata filtering, cached per (top_k, filter) combination
rather than a single global cache — different filters need different
underlying vector retrievers with different search_kwargs baked in.

Note: BM25Retriever has no native metadata filtering, so a filter only
narrows the vector-search half of the ensemble. This is a known
simplification — a fuller version would maintain per-chapter BM25
indexes, which is more than this project needs.

pip install langchain langchain-community langchain-cohere cohere
"""

from __future__ import annotations
import os

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store


class HybridRetriever:
    """
    Full retrieval pipeline:
      Vector search + BM25 → EnsembleRetriever (weighted RRF fusion)
      → Cohere Rerank       → ContextualCompressionRetriever
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store:   BM25Store,
        embedder:     Embedder,
        top_k:        int   = 12,   # candidates before reranking
        top_n:        int   = 6,    # chunks after reranking
        vector_weight: float = 0.5,
        bm25_weight:   float = 0.5,
    ):
        self.vector_store  = vector_store
        self.bm25_store    = bm25_store
        self.embedder      = embedder
        self.top_k         = top_k
        self.top_n         = top_n
        self.vector_weight = vector_weight
        self.bm25_weight   = bm25_weight

        # Cache keyed by (top_k, filter) — NOT a single slot — since
        # different filters need genuinely different retriever chains.
        self._retriever_cache: dict = {}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[Chunk]:
        """
        Run the full hybrid + rerank pipeline.

        metadata_filter example: {"chapter": "9"}
        Narrows the vector-search half to chunks from that chapter only.
        """
        k = top_k or self.top_k
        retriever = self._get_retriever(k, metadata_filter)
        lc_docs   = retriever.invoke(query)
        return self._to_chunks(lc_docs)

    # ------------------------------------------------------------------ #
    #  RETRIEVER CONSTRUCTION (cached per configuration)                   #
    # ------------------------------------------------------------------ #

    def _get_retriever(
        self,
        top_k: int,
        metadata_filter: dict | None,
    ) -> ContextualCompressionRetriever:
        cache_key = (top_k, str(metadata_filter))
        if cache_key in self._retriever_cache:
            return self._retriever_cache[cache_key]

        # Vector retriever — filter applied here
        vector_retriever = self.vector_store.as_langchain_retriever(
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # BM25 retriever — no filter support, searches the full index
        bm25_retriever = self.bm25_store.as_langchain_retriever(top_k=top_k)

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight],
        )

        cohere_reranker = CohereRerank(
            cohere_api_key=os.environ["COHERE_API_KEY"],
            model="rerank-english-v3.0",
            top_n=self.top_n,
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=cohere_reranker,
            base_retriever=ensemble,
        )

        self._retriever_cache[cache_key] = retriever
        return retriever

    def invalidate(self):
        """Clear the entire cache (call if the stores are reloaded)."""
        self._retriever_cache = {}

    # ------------------------------------------------------------------ #
    #  CONVERSION                                                          #
    # ------------------------------------------------------------------ #

    def _to_chunks(self, docs: list[Document]) -> list[Chunk]:
        return [
            Chunk(
                text=doc.page_content,
                metadata=doc.metadata,
                score=float(doc.metadata.get("relevance_score", 0.0)),
            )
            for doc in docs
        ]
