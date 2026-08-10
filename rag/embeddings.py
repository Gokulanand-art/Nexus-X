"""
embeddings.py — Offline embeddings via Ollama (nomic-embed-text).

- In-process LRU cache: identical text is never re-embedded.
- Disk cache (.npy): embeddings survive restarts, indexed by content hash.
- Truncates long text to the embedding model's context window.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

import config
import model

EMBED_CACHE_DIR = config.CACHE_DIR / "embeddings"
EMBED_MAX_CHARS = 8000   # nomic-embed-text context ≈ 8192 tokens


def _disk_path(text: str) -> Path:
    h = hashlib.sha256(text.encode()).hexdigest()[:32]
    return EMBED_CACHE_DIR / f"{h}.npy"


@lru_cache(maxsize=4096)
def _cached(text: str) -> bytes:
    """Return serialized embedding, from disk or fresh from Ollama."""
    p = _disk_path(text)
    if p.exists():
        return p.read_bytes()
    vec = model.embed_one(text[:EMBED_MAX_CHARS])
    try:
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p.write_bytes(np.asarray(vec, dtype=np.float32).tobytes())
    except Exception:
        pass
    return np.asarray(vec, dtype=np.float32).tobytes()


def embed_text(text: str) -> np.ndarray:
    """Embed one text → float32 vector (cached)."""
    return np.frombuffer(_cached(text.strip() or " "), dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    """Alias for agent/main callers."""
    return embed_text(text)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed many texts; unknown ones are batched for speed."""
    unknown = [(i, t) for i, t in enumerate(texts) if not _disk_path(t.strip()).exists()]
    if unknown:
        fresh = model.embed([t for _, t in unknown])
        try:
            EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for (i, t), vec in zip(unknown, fresh):
                _disk_path(t.strip()).write_bytes(
                    np.asarray(vec, dtype=np.float32).tobytes()
                )
        except Exception:
            pass
    return [np.frombuffer(_cached(t.strip()), dtype=np.float32) for t in texts]


def clear_cache() -> int:
    """Wipe the embedding disk cache. Returns number of files removed."""
    if not EMBED_CACHE_DIR.exists():
        return 0
    n = len(list(EMBED_CACHE_DIR.glob("*.npy")))
    for f in EMBED_CACHE_DIR.glob("*.npy"):
        f.unlink()
    return n
