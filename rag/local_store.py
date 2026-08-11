"""
local_store.py — Default RAG backend: SQLite (FTS5) + numpy vectors.

Zero dependencies beyond stdlib + numpy. Fully offline.

Hybrid search:
  - semantic: cosine similarity over float32 embeddings (numpy, batch dot)
  - keyword:  SQLite FTS5 BM25 over the same rows
  - fusion:   Reciprocal Rank Fusion (RRF) — robust, no tuning
  - filter:   optional source prefix filter applied post-fusion

Handles ~50k chunks comfortably; beyond that, configure Supabase.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

import config
from rag.chunker import Chunk
from rag.vector_store import SearchHit, VectorStore


class LocalStore(VectorStore):
    backend_name = "sqlite-local"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or config.MEMORY_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._emb_cache: Optional[np.ndarray] = None
        self._ids_cache: list[str] = []

    # ─── Schema ────────────────────────────────────────────────────────────

    def _init_schema(self):
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id  TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                source    TEXT NOT NULL,
                metadata  TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created   TEXT DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, source, content='chunks', content_rowid='rowid'
            )
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, source)
                VALUES (new.rowid, new.text, new.source);
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, source)
                VALUES ('delete', old.rowid, old.text, old.source);
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, source)
                VALUES ('delete', old.rowid, old.text, old.source);
                INSERT INTO chunks_fts(rowid, text, source)
                VALUES (new.rowid, new.text, new.source);
            END
        """)
        self._conn.commit()

    # ─── Upsert ────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        stored = 0
        for chunk, emb in zip(chunks, embeddings):
            vec = np.asarray(emb, dtype=np.float32)
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks(chunk_id, text, source, metadata, embedding)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.source,
                    json.dumps(chunk.metadata),
                    vec.tobytes(),
                ),
            )
            stored += 1
        self._conn.commit()
        self._emb_cache = None
        return stored

    # ─── Vector search (numpy batch cosine) ────────────────────────────────

    def _load_vectors(self):
        if self._emb_cache is not None:
            return self._emb_cache, self._ids_cache
        rows = self._conn.execute(
            "SELECT chunk_id, embedding FROM chunks"
        ).fetchall()
        if not rows:
            self._emb_cache, self._ids_cache = np.zeros((0, config.EMBED_DIM), np.float32), []
            return self._emb_cache, self._ids_cache
        ids, blobs = zip(*rows)
        mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(
            len(blobs), -1
        )
        self._emb_cache, self._ids_cache = mat, list(ids)
        return mat, list(ids)

    def _vector_search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        mat, ids = self._load_vectors()
        if mat.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        scores = (mat @ q) / (norm[:, 0] * (np.linalg.norm(q) or 1.0))
        k = min(top_k * 3, mat.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        ranked = idx[np.argsort(-scores[idx])]
        return [(ids[i], float(scores[i])) for i in ranked]

    # ─── Keyword search (FTS5 BM25) ────────────────────────────────────────

    @staticmethod
    def _fts_query(query: str) -> str:
        """Build a safe FTS5 match expression from free text."""
        terms = [t for t in query.replace('"', " ").split() if t.isalnum()]
        if not terms:
            return ""
        return " AND ".join(f'"{t}"' for t in terms[:8])

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q = self._fts_query(query)
        if not q:
            return []
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS score, rank "
            "FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (q, top_k * 3),
        ).fetchall()
        return [(row[0], -float(row[1])) for row in rows]

    # ─── Hybrid search with RRF fusion ─────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = config.RAG_TOP_K,
        source_filter: Optional[str] = None,
    ) -> list[SearchHit]:
        semantic = self._vector_search(query_vector, top_k)
        keyword  = self._keyword_search(query, top_k)

        def rrf(ranked: list[tuple[str, float]]) -> dict[str, float]:
            fused = {}
            for rank, (cid, _) in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RAG_FUSION_K + rank + 1)
            return fused

        scores = rrf(semantic)
        for cid, s in rrf(keyword).items():
            scores[cid] = scores.get(cid, 0.0) + s

        if source_filter:
            scores = {
                cid: s for cid, s in scores.items()
                if cid.startswith(source_filter) or source_filter in cid
            }

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return self._hits(ranked_ids, scores)

    def _hits(self, ranked_ids: list[str], scores: dict) -> list[SearchHit]:
        if not ranked_ids:
            return []
        placeholders = ",".join("?" * len(ranked_ids))
        rows = self._conn.execute(
            f"SELECT chunk_id, text, source, metadata FROM chunks "
            f"WHERE chunk_id IN ({placeholders})",
            ranked_ids,
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        hits = []
        for cid in ranked_ids:
            r = by_id.get(cid)
            if not r:
                continue
            hits.append(SearchHit(
                chunk_id=r[0], text=r[1], source=r[2],
                score=round(scores[cid], 4),
                metadata=json.loads(r[3]),
            ))
        return hits

    # ─── Misc ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"total_chunks": n, "backend": self.backend_name,
                "db_path": str(self.db_path)}

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM chunks_fts")
        self._conn.commit()
        self._emb_cache = None

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
