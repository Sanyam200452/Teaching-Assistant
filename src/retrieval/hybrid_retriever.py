"""
hybrid_retriever.py — LangChain EnsembleRetriever + Cohere reranker.

Pipeline:
  1. EnsembleRetriever   → fuses vector search + BM25 (weighted RRF)
  2. CohereRerank        → cross-encoder reranking via Cohere API
  3. ContextualCompressionRetriever → wraps both into one LangChain retriever

This is the standard LangChain production RAG retrieval pattern.

pip install langchain langchain-community langchain-cohere cohere
"""

from __future__ import annotations
import os

from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
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

        self._retriever = None   # built lazily on first query

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """
        Run the full hybrid + rerank pipeline.
        Returns reranked Chunk objects.
        """
        k = top_k or self.top_k
        retriever = self._get_retriever(k)
        lc_docs   = retriever.invoke(query)
        return self._to_chunks(lc_docs)

    # ------------------------------------------------------------------ #
    #  RETRIEVER CONSTRUCTION                                              #
    # ------------------------------------------------------------------ #

    def _get_retriever(self, top_k: int) -> ContextualCompressionRetriever:
        """Build and cache the full retriever pipeline."""
        if self._retriever is not None:
            return self._retriever

        # 1. Vector retriever (LangChain interface from QdrantVectorStore)
        vector_retriever = self.vector_store.as_langchain_retriever(top_k=top_k)

        # 2. BM25 retriever (LangChain BM25Retriever)
        bm25_retriever = self.bm25_store.as_langchain_retriever(top_k=top_k)

        # 3. Fuse with EnsembleRetriever (weighted reciprocal rank fusion)
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight],
        )

        # 4. Wrap with Cohere reranker
        cohere_reranker = CohereRerank(
            cohere_api_key=os.environ["COHERE_API_KEY"],
            model="rerank-english-v3.0",
            top_n=self.top_n,
        )

        self._retriever = ContextualCompressionRetriever(
            base_compressor=cohere_reranker,
            base_retriever=ensemble,
        )
        return self._retriever

    def invalidate(self):
        """Reset the cached retriever (call if stores are reloaded)."""
        self._retriever = None

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
