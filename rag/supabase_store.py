"""
supabase_store.py — Cloud RAG backend on the Supabase free tier (pgvector).

Enabled only when SUPABASE_URL and SUPABASE_KEY are set. Requires a one-time
schema setup (see scripts/supabase_schema.sql) which creates:
  - the pgvector extension
  - the `nexus_chunks` table with an embedding vector(768) column
  - a generated tsvector column + GIN index (full-text keyword side)
  - the hybrid_search() SQL function (RRF-style fusion)

All queries run through the Supabase REST API (PostgREST) — no driver
dependencies beyond the official `supabase` client.
"""

import json
from typing import Optional

import config
from rag.chunker import Chunk
from rag.vector_store import SearchHit, VectorStore

TABLE = "nexus_chunks"


class SupabaseStore(VectorStore):
    backend_name = "supabase-pgvector"

    def __init__(self):
        try:
            from supabase import create_client
        except ImportError as e:
            raise RuntimeError(
                "Supabase client not installed. Run: pip install supabase\n"
                "Or unset SUPABASE_URL/SUPABASE_KEY to use the local store."
            ) from e
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not configured.")
        self._client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    def _table(self):
        return self._client.table(TABLE)

    # ─── Upsert ────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        rows = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append({
                "chunk_id":  chunk.chunk_id,
                "text":      chunk.text,
                "source":    chunk.source,
                "metadata":  json.dumps(chunk.metadata),
                "embedding": emb,
            })
        if not rows:
            return 0
        # Batch in groups of 100 to respect PostgREST row limits
        for i in range(0, len(rows), 100):
            self._table().upsert(rows[i:i + 100], on_conflict="chunk_id").execute()
        return len(rows)

    # ─── Hybrid search ─────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = config.RAG_TOP_K,
        source_filter: Optional[str] = None,
    ) -> list[SearchHit]:
        params = {
            "query_text": query,
            "query_vec": json.dumps(query_vector),
            "top_k": top_k,
            "source_filter": source_filter or "",
        }
        data = self._client.rpc("hybrid_search", params).execute()
        hits = []
        for row in (data.data or []):
            hits.append(SearchHit(
                chunk_id=row.get("chunk_id", ""),
                text=row.get("text", ""),
                source=row.get("source", ""),
                score=round(float(row.get("score", 0.0)), 4),
                metadata=json.loads(row.get("metadata") or "{}"),
            ))
        return hits

    # ─── Misc ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        data = self._client.rpc(
            "nexus_stats", {}
        ).execute() if hasattr(self._client, "rpc") else None
        n = 0
        if data and data.data:
            n = data.data[0].get("total", 0)
        else:
            try:
                resp = self._table().select("chunk_id", count="exact").limit(1).execute()
                n = resp.count or 0
            except Exception:
                n = -1
        return {"total_chunks": n, "backend": self.backend_name}

    def clear(self) -> None:
        self._client.rpc("nexus_clear", {}).execute()
