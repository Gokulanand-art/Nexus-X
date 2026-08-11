"""
chunker.py — Claude-grade recursive chunking.

Strategy (like Anthropic's codebase chunking):
  1. Split on markdown headings → each section becomes a unit,
     carrying its heading hierarchy as metadata.
  2. Split code files on blank lines / logical blocks.
  3. Split long prose on paragraphs, then sentences.
  4. Final chunks are capped at CHUNK_TOKENS words with CHUNK_OVERLAP
     sliding overlap so retrieval never loses context at boundaries.
"""

import re
from dataclasses import dataclass, field

import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
BLOCK_RE   = re.compile(r"^(```[\w+-]*)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    source: str                 # filename
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    @property
    def content(self) -> str:
        """Text fed to the embedding model — includes heading context."""
        if self.metadata.get("headings"):
            prefix = " > ".join(self.metadata["headings"])
            return f"{prefix}\n{self.text}"
        return self.text


# ─── Heading-aware splitting ────────────────────────────────────────────────

def _split_by_headings(text: str) -> list[tuple[str, list[str]]]:
    """Split into (section_text, heading_hierarchy)."""
    matches = list(HEADING_RE.finditer(text))
    if len(matches) <= 1:
        return [(text.strip(), [])]

    sections = []
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        if body:
            sections.append((body, [(level, title)]))
    return sections


def _flatten_headings(sections: list[tuple[str, list[tuple[int, str]]]]) -> list[str]:
    """Walk sections, threading heading hierarchy through subsections."""
    result = []
    stack: list[tuple[int, str]] = []
    for text, heads in sections:
        for level, title in heads:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        hierarchy = [t for _, t in stack]
        result.append((text, hierarchy))
    return result


# ─── Sentence / paragraph splitting ─────────────────────────────────────────

SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_sentences(text: str) -> list[str]:
    parts = SENT_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ─── Code block splitting ───────────────────────────────────────────────────

def _split_code_blocks(text: str) -> list[str]:
    """Split on blank lines for code, preserving language fences."""
    lines = text.splitlines()
    blocks, cur = [], []
    in_fence = False
    for ln in lines:
        if BLOCK_RE.match(ln):
            if in_fence:
                blocks.append("\n".join(cur))
                cur = []
            in_fence = not in_fence
            cur.append(ln)
            continue
        if not in_fence and not ln.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append("\n".join(cur))
    return [b.strip() for b in blocks if b.strip()]


# ─── Token-aware windowing ──────────────────────────────────────────────────

def _window(words: list[str], size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), step)
        if words[i:i + size]
    ]


def _cap(text: str, max_words: int = config.CHUNK_TOKENS) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


# ─── Main entry ─────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, lang: str = "text") -> list[Chunk]:
    """
    Split arbitrary text into RAG-ready chunks with rich metadata.
    Returns Chunk objects; embeddings are added later by the indexer.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    counter = 0

    def emit(body: str, meta: dict):
        nonlocal counter
        body = _cap(body)
        if len(body) < 8:
            return
        for piece in _window(body.split(), config.CHUNK_TOKENS, config.CHUNK_OVERLAP):
            counter += 1
            chunks.append(Chunk(
                text=piece,
                source=source,
                metadata={**meta, "chunk_index": counter},
            ))

    # 1) Markdown → heading sections with hierarchy
    if source.endswith((".md", ".markdown", ".txt")):
        sections = _split_by_headings(text)
        flat = _flatten_headings(sections)
        for body, hierarchy in flat:
            meta = {"headings": hierarchy, "lang": "markdown"}
            if len(body.split()) > config.CHUNK_TOKENS:
                for sub in _split_sentences(body):
                    emit(sub, meta)
            else:
                emit(body, meta)
        return chunks

    # 2) Code → logical blocks, then word windows
    if lang in ("code",) or source.endswith((
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cpp",
        ".h", ".java", ".rb", ".php", ".sh", ".sql", ".html", ".css",
        ".json", ".yaml", ".yml", ".toml",
    )):
        for block in _split_code_blocks(text):
            emit(block, {"lang": "code"})
        return chunks

    # 3) Prose → paragraphs → sentences
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) > config.CHUNK_TOKENS:
            for sent in _split_sentences(para):
                emit(sent, {"lang": "text"})
        else:
            emit(para, {"lang": "text"})
    return chunks


def ingest_text(text: str, source: str, lang: str = "text") -> list[Chunk]:
    """Convenience alias used by indexer."""
    return chunk_text(text, source, lang)
