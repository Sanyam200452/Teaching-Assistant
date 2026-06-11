"""
Embedder — wraps OpenAI's text-embedding-3-small.
Handles batching (max 2048 inputs per API call) and basic LRU caching.
"""

from __future__ import annotations
import os
from functools import lru_cache
from openai import OpenAI


class Embedder:
    MODEL = "text-embedding-3-small"  # 1536 dims, cheap, great quality
    BATCH_SIZE = 512                   # stay well under API limits

    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns a list of float vectors."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            response = self.client.embeddings.create(
                model=self.MODEL,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string (cached by content hash)."""
        return self._cached_embed(text)

    @lru_cache(maxsize=1024)
    def _cached_embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.MODEL, input=[text])
        return response.data[0].embedding

    @property
    def dim(self) -> int:
        return 1536 ##Used elsewhere when creating the Qdrant collection to set the vector dimension:
#The value 1536 is hardcoded because text-embedding-3-small always produces 1536-dimensional vectors — it's a fixed property of the model. If you ever swap to a different model (e.g. text-embedding-3-large produces 3072 dims), you'd update both MODEL and the return value of dim together.
