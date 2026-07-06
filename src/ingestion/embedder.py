"""
embedder.py — Embeddings via LangChain OpenAIEmbeddings.

LangChain handles:
  - Batching automatically
  - Retry logic on rate limit errors
  - embed_query() vs embed_documents() distinction
    (some models use different prompt prefixes for queries vs stored docs)

pip install langchain-openai
"""

from __future__ import annotations
import os
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class Embedder:
    MODEL = "text-embedding-3-small"  # 1536 dims

    def __init__(self):
        self._model = OpenAIEmbeddings(
            model=self.MODEL,
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (used during indexing)."""
        return self._model.embed_documents(texts)

    def embed_one(self, text: str) -> list[float]:
        """Embed a single query string. LRU cached."""
        return self._cached(text)

    @lru_cache(maxsize=1024)
    def _cached(self, text: str) -> list[float]:
        # embed_query uses the query prefix for better retrieval quality
        return self._model.embed_query(text)

    @property
    def dim(self) -> int:
        return 1536
