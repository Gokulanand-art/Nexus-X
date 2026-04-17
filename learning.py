"""
learning.py — Vector memory for Nexus X.

Uses ChromaDB (local, offline, persistent).
Stores conversations, code, and file chunks as vectors.
Retrieves relevant past context before every prompt.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings

# ─── Config ───────────────────────────────────────────────────────────────────

DB_DIR       = Path(__file__).parent / ".nexus_memory"
COLLECTION   = "nexus_knowledge"
MAX_RESULTS  = 5      # how many chunks to retrieve per query
MIN_RELEVANCE = 0.15   # ignore chunks below this similarity score

# ─── Client (persistent, fully offline) ───────────────────────────────────────

def _get_client():
    return chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

# ─── Store ────────────────────────────────────────────────────────────────────

def store(text: str, source: str = "conversation", metadata: dict = None) -> str:
    """
    Store a text chunk in the vector DB.
    Returns the chunk ID.
    Skips duplicates silently.
    """
    if not text or not text.strip():
        return ""

    # Deterministic ID — same text never stored twice
    chunk_id = hashlib.sha256(text.encode()).hexdigest()[:16]

    col = _get_collection()

    # Skip if already exists
    existing = col.get(ids=[chunk_id])
    if existing["ids"]:
        return chunk_id

    meta = {
        "source":    source,
        "timestamp": datetime.now().isoformat(),
        **(metadata or {}),
    }

    col.add(
        ids        = [chunk_id],
        documents  = [text.strip()],
        metadatas  = [meta],
    )

    return chunk_id


# ─── Retrieve ─────────────────────────────────────────────────────────────────

def retrieve(query: str, n: int = MAX_RESULTS) -> list[dict]:
    """
    Find the most relevant stored chunks for a query.
    Returns list of {text, source, timestamp, score}.
    """
    if not query or not query.strip():
        return []

    col = _get_collection()

    # Need at least 1 document stored
    if col.count() == 0:
        return []

    results = col.query(
        query_texts = [query],
        n_results   = min(n, col.count()),
    )

    chunks = []
    docs       = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = 1 - dist   # cosine distance → similarity score
        if score < MIN_RELEVANCE:
            continue
        chunks.append({
            "text":      doc,
            "source":    meta.get("source", "unknown"),
            "timestamp": meta.get("timestamp", ""),
            "score":     round(score, 3),
        })

    # Best match first
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


# ─── Build prompt block ───────────────────────────────────────────────────────

def get_context_block(query: str) -> str:
    """
    Retrieve relevant memory and format it for injection into system prompt.
    Returns empty string if nothing relevant found.
    """
    chunks = retrieve(query)
    if not chunks:
        return ""

    lines = ["## Relevant memory from past sessions:\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"{i}. [{c['source']}] (relevance: {c['score']})")
        lines.append(f"   {c['text'][:300]}")
        lines.append("")

    return "\n".join(lines)


# ─── Auto-save conversation turn ──────────────────────────────────────────────

def save_conversation(user_msg: str, assistant_msg: str) -> None:
    """
    Called after every agent response.
    Stores both sides of the conversation as one chunk.
    """
    if not user_msg.strip() or not assistant_msg.strip():
        return

    text = f"User: {user_msg.strip()}\nNexus: {assistant_msg.strip()}"
    store(text, source="conversation")


# ─── Stats ────────────────────────────────────────────────────────────────────

def stats() -> dict:
    """Return memory stats."""
    col = _get_collection()
    return {
        "total_chunks": col.count(),
        "db_path":      str(DB_DIR),
    }


# ─── Clear ────────────────────────────────────────────────────────────────────

def clear_memory() -> None:
    """Wipe all stored memory. Irreversible."""
    client = _get_client()
    client.delete_collection(COLLECTION)

def is_available() -> bool:
    return True
