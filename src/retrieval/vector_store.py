"""
vector_store.py — Qdrant Cloud via LangChain QdrantVectorStore.

LangChain's QdrantVectorStore wraps qdrant-client and provides:
  - as_retriever()  → plug directly into LangChain chains/retrievers
  - similarity_search_by_vector() → search with a pre-computed vector

We keep the raw QdrantClient for admin operations (create collection,
check if data exists) since LangChain doesn't expose those directly.
Now supports metadata filtering (e.g. restrict search to one chapter)
via as_langchain_retriever(metadata_filter=...).

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
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from src.ingestion.chunker import Chunk
# export OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
load_dotenv()


def _get_secret(key: str) -> str | None:
    """Streamlit secrets first, .env fallback."""
    try:
        if hasattr(st, "secrets"):
            val = st.secrets.get(key,None)
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
    #  SEARCH (raw client, pre-computed vector)                            #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metadata_filter: dict | None = None,
    ) -> list[Chunk]:
        """Search using a pre-computed query vector, with optional metadata filter."""
        query_filter = self._build_filter(metadata_filter)

        results = self._raw_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
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

    # ------------------------------------------------------------------ #
    #  LANGCHAIN RETRIEVER (used by HybridRetriever / EnsembleRetriever)   #
    # ------------------------------------------------------------------ #

    def as_langchain_retriever(self, top_k: int = 10, metadata_filter: dict | None = None):
        """
        Return a LangChain BaseRetriever, optionally scoped to a metadata filter.

        metadata_filter example: {"chapter": "9"}
        Matches chunks where payload["metadata"]["chapter"] == "9" exactly.
        """
        search_kwargs = {"k": top_k}
        qdrant_filter = self._build_filter(metadata_filter)
        if qdrant_filter is not None:
            search_kwargs["filter"] = qdrant_filter
        return self._store.as_retriever(search_kwargs=search_kwargs)

    # ------------------------------------------------------------------ #
    #  FILTER BUILDING                                                     #
    # ------------------------------------------------------------------ #

    def _build_filter(self, metadata_filter: dict | None) -> Filter | None:
        """
        Translate a plain dict like {"chapter": "9"} into a Qdrant Filter.
        Payload keys are nested under "metadata." since that's how we store
        them in upsert() — payload={"page_content": ..., "metadata": {...}}.
        """
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            if value is None:
                continue
            conditions.append(
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            )

        return Filter(must=conditions) if conditions else None