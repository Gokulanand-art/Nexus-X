"""
dataset.py — Conversation to JSONL training data builder.

Every exchange with Nexus is saved as a training pair:
  {"prompt": "user message", "response": "nexus response"}

This builds a dataset you can use to fine-tune a model later
on better hardware (LoRA, QLoRA, etc).
"""

import json
from datetime import datetime
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_DIR  = Path(__file__).parent / ".nexus_dataset"
DATASET_FILE = DATASET_DIR / "conversations.jsonl"
MIN_LENGTH   = 20   # ignore very short exchanges

# ─── Core ─────────────────────────────────────────────────────────────────────

def _ensure_dir():
    DATASET_DIR.mkdir(exist_ok=True)


def save_pair(prompt: str, response: str, source: str = "conversation") -> bool:
    """
    Save a prompt/response pair to the JSONL dataset.
    Returns True if saved, False if skipped (too short).
    """
    if len(prompt.strip()) < MIN_LENGTH or len(response.strip()) < MIN_LENGTH:
        return False

    _ensure_dir()

    record = {
        "prompt":    prompt.strip(),
        "response":  response.strip(),
        "source":    source,
        "timestamp": datetime.now().isoformat(),
    }

    with open(DATASET_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    return True


def save_conversation(user_msg: str, assistant_msg: str) -> bool:
    """
    Called after every agent response.
    Saves the exchange as a training pair.
    Also stores in vector memory via learning.py.
    """
    import learning
    learning.save_conversation(user_msg, assistant_msg)
    return save_pair(user_msg, assistant_msg, source="conversation")


# ─── Stats ────────────────────────────────────────────────────────────────────

def stats() -> dict:
    """Return dataset stats."""
    _ensure_dir()
    if not DATASET_FILE.exists():
        return {"total_pairs": 0, "dataset_path": str(DATASET_FILE)}

    count = sum(1 for _ in open(DATASET_FILE))
    size  = DATASET_FILE.stat().st_size

    return {
        "total_pairs":  count,
        "size_bytes":   size,
        "dataset_path": str(DATASET_FILE),
    }


def recent(n: int = 5) -> list[dict]:
    """Return the last N training pairs."""
    if not DATASET_FILE.exists():
        return []

    lines = DATASET_FILE.read_text().strip().splitlines()
    last  = lines[-n:] if len(lines) >= n else lines

    result = []
    for line in last:
        try:
            result.append(json.loads(line))
        except Exception:
            continue
    return result


def clear() -> None:
    """Wipe the dataset. Irreversible."""
    if DATASET_FILE.exists():
        DATASET_FILE.unlink()

