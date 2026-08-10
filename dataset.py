"""
dataset.py — Every exchange saved as a JSONL training pair.

Same idea as v1 (fine-tune later on better hardware), but pairs now
carry quality + reasoning annotations and chunk counts for filtering.
"""

import json
from datetime import datetime
from pathlib import Path

import config

MIN_LENGTH = 20


def _ensure_dir():
    config.DATASET_DIR.mkdir(exist_ok=True)


def save_pair(
    prompt: str,
    response: str,
    source: str = "conversation",
    quality: float | None = None,
    reasoning: str = "",
    context_sources: list[str] | None = None,
) -> bool:
    if len(prompt.strip()) < MIN_LENGTH or len(response.strip()) < MIN_LENGTH:
        return False
    _ensure_dir()
    record = {
        "prompt": prompt.strip(),
        "response": response.strip(),
        "source": source,
        "timestamp": datetime.now().isoformat(),
    }
    if quality is not None:
        record["quality"] = quality
    if reasoning:
        record["reasoning"] = reasoning.strip()
    if context_sources:
        record["context_sources"] = context_sources[:20]
    with open(config.DATASET_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def stats() -> dict:
    _ensure_dir()
    if not config.DATASET_FILE.exists():
        return {"total_pairs": 0, "dataset_path": str(config.DATASET_FILE)}
    count = sum(1 for _ in open(config.DATASET_FILE))
    return {
        "total_pairs": count,
        "size_bytes": config.DATASET_FILE.stat().st_size,
        "dataset_path": str(config.DATASET_FILE),
    }


def clear() -> None:
    if config.DATASET_FILE.exists():
        config.DATASET_FILE.unlink()
