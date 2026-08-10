"""
indexer.py — Turn files into RAG chunks.

  ingest_file(path)     → read → chunk → embed → upsert
  ingest_folder(path)   → all supported files recursively
  build_index(store, out_dir) → /ingest the whole working tree

Supported: code, markdown, text, JSON/YAML, PDF (pypdf), images (OCR via
tesseract if installed). Unreadable files are reported, never fatal.
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional

import config
import model
from rag import embeddings
from rag.chunker import Chunk, chunk_text
from rag.vector_store import VectorStore, get_store

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cpp", ".h",
    ".java", ".rb", ".php", ".sh", ".sql", ".html", ".css", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".md", ".markdown", ".txt", ".csv",
}
DOC_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".mypy_cache", ".pytest_cache",
    ".nexus_vectors", ".nexus_dataset", ".nexus_cache", ".tox",
}
MAX_FILE_CHARS = 2_000_000   # safety cap per file


def _read_text(path: Path) -> str:
    return path.read_text(errors="replace")[:MAX_FILE_CHARS]


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )[:MAX_FILE_CHARS]
    except ImportError:
        return "[pdf] pypdf not installed — pip install pypdf"
    except Exception as e:
        return f"[pdf error] {e}"


def _read_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path)).strip()
    except ImportError:
        return "[image] pytesseract/pillow not installed"
    except Exception as e:
        return f"[image error] {e}"


def _read_file(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return _read_text(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in DOC_EXTENSIONS:
        return _read_image(path)
    try:
        return _read_text(path)
    except Exception:
        return None


def _chunk_id(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()[:20]


def _prepare_chunks(text: str, source: str) -> list[Chunk]:
    chunks = chunk_text(text, source=source)
    for c in chunks:
        c.chunk_id = _chunk_id(c.text, source)
    return chunks


def _store_chunks(
    store: VectorStore,
    chunks: list[Chunk],
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    if not chunks:
        return 0
    total = 0
    batch = 64
    for i in range(0, len(chunks), batch):
        group = chunks[i:i + batch]
        vecs = embeddings.embed_batch([c.content for c in group])
        total += store.upsert(group, [v.tolist() for v in vecs])
        if progress:
            progress(min(i + batch, len(chunks)), len(chunks))
    return total


def ingest_file(
    path: str,
    store: Optional[VectorStore] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Ingest one file into the vector store. Returns a result dict."""
    store = store or get_store()
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return {"ok": False, "error": f"File not found: {path}"}
    if not p.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    text = _read_file(p)
    if not text or not text.strip():
        return {"ok": False, "error": f"No text extracted: {p.name}"}

    chunks = _prepare_chunks(text, p.name)
    stored = _store_chunks(store, chunks, progress)

    return {
        "ok": True,
        "source": p.name,
        "chunks_stored": stored,
        "total_chunks": len(chunks),
    }


def ingest_folder(
    path: str,
    store: Optional[VectorStore] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    recursive: bool = True,
) -> dict:
    """Ingest every supported file under a folder."""
    store = store or get_store()
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"ok": False, "error": f"Folder not found: {path}"}

    files: list[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if any(skip in p.parts for skip in SKIP_DIRS):
                continue
            if p.is_file():
                files.append(p)
    else:
        files = [p for p in root.iterdir() if p.is_file()]

    supported = [f for f in files
                 if f.suffix.lower() in TEXT_EXTENSIONS | DOC_EXTENSIONS]

    total_chunks = 0
    errors = []
    for i, f in enumerate(supported, 1):
        r = ingest_file(str(f), store=store)
        if r["ok"]:
            total_chunks += r["chunks_stored"]
        else:
            errors.append(f"{f.name}: {r.get('error')}")
        if progress:
            progress(i, len(supported))

    return {
        "ok": True,
        "files_processed": len(supported),
        "total_chunks": total_chunks,
        "errors": errors,
    }


def ingest_workspace(store: Optional[VectorStore] = None) -> dict:
    """Index the current working directory (the repo you're working on)."""
    import os
    return ingest_folder(os.getcwd(), store=store)
