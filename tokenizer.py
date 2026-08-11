"""
tokenizer.py — Claude-style tokenizer-aware context management.

Nexus counts real tokens with the exact Qwen tokenizer (cached locally at
setup; downloaded once from HuggingFace, then fully offline). If the
tokenizer file is unavailable, it falls back to a fast heuristic so the
system never breaks.

Jobs:
  - count_tokens(text)          — exact-ish token count
  - trim_messages(messages, budget) — drop oldest turns until the
    conversation fits the context budget (like Claude's context window
    management)
"""

import json
import math
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

from config import CACHE_DIR

TOKENIZER_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/"
    "resolve/main/tokenizer.json"
)
TOKENIZER_FILE = CACHE_DIR / "qwen_tokenizer.json"


# ─── Loading ────────────────────────────────────────────────────────────────

def _download_tokenizer() -> bool:
    """Fetch the Qwen tokenizer once and cache it locally."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOKENIZER_URL, TOKENIZER_FILE)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _tokenizer():
    """Return a fast BPE tokenizer, or None if unavailable."""
    if not TOKENIZER_FILE.exists():
        _download_tokenizer()
    if not TOKENIZER_FILE.exists():
        return None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER_FILE))
        tok.enable_truncation(max_length=1_000_000)
        return tok
    except Exception:
        return None


# ─── Counting ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """Count tokens. Exact via Qwen BPE when available, else heuristic."""
    if not text:
        return 0
    tok = _tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text).ids)
        except Exception:
            pass
    # Fallback heuristic: BPE-ish ≈ 1 token per 3.5 chars for English+code,
    # heavy C/J tokens for symbol-dense text.
    symbols = len(re.findall(r"[^\w\s]", text))
    return max(1, math.ceil((len(text) - symbols * 0.8) / 3.5) + symbols // 3)


def count_messages(messages: list[dict]) -> int:
    """Total tokens across a message list (role tokens included)."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content", ""))
        total += 4  # role + framing tokens per message
    return total


# ─── Trimming (Claude-style context management) ─────────────────────────────

def trim_messages(
    messages: list[dict],
    budget: int,
    keep_system: bool = True,
) -> list[dict]:
    """
    Drop the OLDEST turns until the message list fits the token budget.

    Always keeps:
      - the system message (index 0)
      - the most recent user query (the last message)

    This mirrors how Claude manages long conversations: keep system +
    recent context, evict the middle.
    """
    if not messages:
        return messages

    system_idx = 0 if keep_system and messages[0].get("role") == "system" else None
    keep = [messages[system_idx]] if system_idx is not None else []
    rest = [m for i, m in enumerate(messages) if i != system_idx]

    while rest and count_messages(keep + rest) > budget:
        # Never drop the newest message (the active query)
        if len(rest) <= 1:
            break
        rest.pop(0)

    return keep + rest


def usage_report(prompt_tokens: int, gen_tokens: int) -> str:
    """Small token-usage line, Claude-CLI style."""
    return f"tokens: {prompt_tokens:,} in · {gen_tokens:,} out"
