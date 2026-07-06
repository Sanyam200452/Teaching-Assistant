"""
vector_store.py — Qdrant Cloud via LangChain QdrantVectorStore.

LangChain's QdrantVectorStore wraps qdrant-client and provides:
  - as_retriever()  → plug directly into LangChain chains/retrievers
  - similarity_search_by_vector() → search with a pre-computed vector

We keep the raw QdrantClient for admin operations (create collection,
check if data exists) since LangChain doesn't expose those directly.

pip install langchain-qdrant qdrant-client
"""

from __future__ import annotations
import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.ingestion.chunker import Chunk

load_dotenv()


def _get_secret(key: str) -> str | None:
    """Streamlit secrets first, .env fallback."""
    try:
        if hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val:
                return val
    except Exception:
        pass
    return os.environ.get(key)


class VectorStore:
    VECTOR_DIM = 1536
    MODEL      = "text-embedding-3-small"

    def __init__(self, collection_name: str = "d2l-book"):
        self.collection_name = collection_name
        self._raw_client     = self._build_raw_client()
        self._ensure_collection()

        embeddings = OpenAIEmbeddings(
            model=self.MODEL,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

        # LangChain wrapper — used for search and as_retriever()
        self._store = QdrantVectorStore(
            client=self._raw_client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )

    # ------------------------------------------------------------------ #
    #  CLIENT SETUP                                                        #
    # ------------------------------------------------------------------ #

    def _build_raw_client(self) -> QdrantClient:
        url     = _get_secret("QDRANT_URL")
        api_key = _get_secret("QDRANT_API_KEY")
        if url and api_key:
            return QdrantClient(url=url, api_key=api_key)
        print("⚠️  No Qdrant credentials — using in-memory store.")
        return QdrantClient(":memory:")

    def _ensure_collection(self):
        existing = [c.name for c in self._raw_client.get_collections().collections]
        if self.collection_name not in existing:
            self._raw_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
            )

    def collection_exists_and_has_data(self) -> bool:
        try:
            return self._raw_client.get_collection(self.collection_name).points_count > 0
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  UPSERT                                                              #
    # ------------------------------------------------------------------ #

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """
        Upload chunks with pre-computed embeddings to Qdrant.
        Uses raw client to avoid re-embedding (LangChain's add_documents
        would call the embedding model again).

        Payload format matches what LangChain's QdrantVectorStore expects:
          page_content → the text
          metadata     → dict of metadata fields
        """
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "page_content": chunk.text,
                    "metadata":     chunk.metadata,
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._raw_client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

    # ------------------------------------------------------------------ #
    #  SEARCH                                                              #
    # ------------------------------------------------------------------ #

    def search(self, query_vector: list[float], top_k: int = 10) -> list[Chunk]:
        """Search using a pre-computed query vector."""
        results = self._raw_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            Chunk(
                text=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {}),
                score=r.score,
            )
            for r in results
        ]

    def as_langchain_retriever(self, top_k: int = 10):
        """
        Return a LangChain BaseRetriever.
        Used by EnsembleRetriever in hybrid_retriever.py.
        """
        return self._store.as_retriever(search_kwargs={"k": top_k})
