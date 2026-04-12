"""
ingestor.py — Feed any file into Nexus's vector memory.

Supported:
  - Text files (.py, .js, .ts, .md, .txt, .json, .csv, .html, .sh)
  - PDFs (.pdf)
  - Images (.png, .jpg, .jpeg, .webp) — OCR via tesseract

How it works:
  1. Read the file
  2. Split into chunks (so large files don't overflow context)
  3. Store each chunk in ChromaDB via learning.py
"""

import re
from pathlib import Path
from typing import Iterator

import learning

# ─── Config ───────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # words overlap between chunks (preserves context)

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".txt", ".json", ".csv",
    ".html", ".css", ".sh", ".yaml", ".yml",
    ".toml", ".ini", ".env", ".rs", ".go", ".c", ".cpp", ".h",
}

# ─── Chunker ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str) -> Iterator[str]:
    """
    Split text into overlapping word chunks.
    Overlap preserves context at chunk boundaries.
    """
    words = text.split()
    if not words:
        return

    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            yield chunk


# ─── Readers ──────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> str:
    """Read plain text / code files."""
    return path.read_text(errors="replace")


def _read_pdf(path: Path) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except ImportError:
        return "[PDF] pypdf not installed. Run: pip install pypdf"
    except Exception as e:
        return f"[PDF error] {e}"


def _read_image(path: Path) -> str:
    """Extract text from image using tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip() or "[image] No text found via OCR"
    except ImportError:
        return "[image] pytesseract or pillow not installed."
    except Exception as e:
        return f"[image error] {e}"


# ─── Main ingest function ─────────────────────────────────────────────────────

def ingest_file(path: str) -> dict:
    """
    Ingest a single file into Nexus vector memory.
    Returns {ok, chunks_stored, source}.
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return {"ok": False, "error": f"File not found: {path}"}

    if not p.is_file():
        return {"ok": False, "error": f"Not a file: {path}"}

    ext = p.suffix.lower()

    # ── Read ──────────────────────────────────────────────────────────────────
    if ext in TEXT_EXTENSIONS:
        text = _read_text(p)
        source = f"file:{p.name}"
    elif ext == ".pdf":
        text = _read_pdf(p)
        source = f"pdf:{p.name}"
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        text = _read_image(p)
        source = f"image:{p.name}"
    else:
        # Try reading as plain text anyway
        try:
            text = _read_text(p)
            source = f"file:{p.name}"
        except Exception:
            return {"ok": False, "error": f"Unsupported file type: {ext}"}

    if not text.strip():
        return {"ok": False, "error": "File is empty or no text could be extracted"}

    # ── Chunk + store ─────────────────────────────────────────────────────────
    count = 0
    for chunk in _chunk_text(text, source):
        learning.store(chunk, source=source, metadata={"file": str(p)})
        count += 1

    return {"ok": True, "chunks_stored": count, "source": source}


def ingest_folder(folder: str, recursive: bool = True) -> dict:
    """
    Ingest all supported files in a folder.
    Returns {ok, files_processed, total_chunks, errors}.
    """
    root = Path(folder).expanduser().resolve()

    if not root.exists():
        return {"ok": False, "error": f"Folder not found: {folder}"}

    SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".nexus_memory"}

    files = []
    if recursive:
        for p in root.rglob("*"):
            if any(skip in p.parts for skip in SKIP_DIRS):
                continue
            if p.is_file():
                files.append(p)
    else:
        files = [p for p in root.iterdir() if p.is_file()]

    total_chunks = 0
    processed    = 0
    errors       = []

    for f in files:
        ext = f.suffix.lower()
        supported = ext in TEXT_EXTENSIONS or ext in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
        if not supported:
            continue

        result = ingest_file(str(f))
        if result["ok"]:
            total_chunks += result["chunks_stored"]
            processed    += 1
        else:
            errors.append(f"{f.name}: {result.get('error', 'unknown')}")

    return {
        "ok":              True,
        "files_processed": processed,
        "total_chunks":    total_chunks,
        "errors":          errors,
    }


# ─── Stats ────────────────────────────────────────────────────────────────────

def memory_stats() -> dict:
    return learning.stats()

