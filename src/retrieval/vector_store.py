"""
Vector store — Qdrant Cloud version for deployment.

Reads QDRANT_URL and QDRANT_API_KEY from environment / Streamlit secrets.
Falls back to in-memory if neither is set (useful for local testing).
"""

from __future__ import annotations
import os
import uuid
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)
from src.ingestion.chunker import Chunk


def _get_secret(key: str) -> str | None:
    """Read from Streamlit secrets first, then env vars."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


class VectorStore:
    VECTOR_DIM = 1536  # text-embedding-3-small

    def __init__(self, collection_name: str = "d2l-book"):
        self.collection_name = collection_name
        self.client = self._build_client()
        self._ensure_collection()

    def _build_client(self) -> QdrantClient:
        url = _get_secret("QDRANT_URL")
        api_key = _get_secret("QDRANT_API_KEY")

        if url and api_key:
            return QdrantClient(url=url, api_key=api_key)

        # Fallback: in-memory (resets on every Streamlit restart)
        print("⚠️  No Qdrant Cloud credentials found — using in-memory store.")
        return QdrantClient(":memory:")

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
            )

    def collection_exists_and_has_data(self) -> bool:
        """Check if the collection is already populated (skip re-indexing)."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count > 0
        except Exception:
            return False

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"text": chunk.text, **chunk.metadata},
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        # Upload in batches of 100 to avoid timeouts on large PDFs
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        source_filter: str | None = None,
    ) -> list[Chunk]:
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )

        results: list[ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            Chunk(
                text=r.payload["text"],
                metadata={k: v for k, v in r.payload.items() if k != "text"},
                score=r.score,
            )
            for r in results
        ]

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
