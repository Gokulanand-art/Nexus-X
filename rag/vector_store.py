"""
vector_store.py — Storage interface for the Nexus RAG.

Two backends implement the same API:
  - LocalStore     (rag/local_store.py)    SQLite FTS5 + numpy — default,
                                            zero config, fully offline.
  - SupabaseStore  (rag/supabase_store.py) pgvector hybrid search on the
                                            Supabase free tier — enabled
                                            when SUPABASE_URL+KEY are set.

Hybrid search = vector similarity (semantic) fused with keyword/BM25
(full-text) using Reciprocal Rank Fusion, then metadata filtering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import config
from rag.chunker import Chunk


class StoreError(RuntimeError):
    pass


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict


class VectorStore(ABC):
    backend_name = "base"

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert or update chunks. Returns count stored."""

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = config.RAG_TOP_K,
        source_filter: Optional[str] = None,
    ) -> list[SearchHit]:
        """Fused semantic + keyword search."""

    @abstractmethod
    def stats(self) -> dict:
        """{total_chunks, backend}"""

    @abstractmethod
    def clear(self) -> None:
        """Wipe all stored chunks. Irreversible."""


def get_store() -> VectorStore:
    """Factory — returns the active backend per config."""
    if config.store_backend() == "supabase":
        from rag.supabase_store import SupabaseStore
        return SupabaseStore()
    from rag.local_store import LocalStore
    return LocalStore()
