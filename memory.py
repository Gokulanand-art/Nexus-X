"""
memory.py — Mistake learning system.

Loads mistakes.json at startup, injects top mistakes into every prompt,
and records new failures so the agent never repeats them.
"""

import json
import os
from datetime import datetime
from pathlib import Path

MISTAKES_FILE = Path(__file__).parent / "mistakes.json"
MAX_INJECT = 8  # max mistakes injected into system prompt


def _load() -> list[dict]:
    if not MISTAKES_FILE.exists():
        return []
    try:
        return json.loads(MISTAKES_FILE.read_text())
    except Exception:
        return []


def _save(mistakes: list[dict]) -> None:
    MISTAKES_FILE.write_text(json.dumps(mistakes, indent=2))


def get_mistakes_prompt() -> str:
    """Return a formatted block of past mistakes for injection into system prompt."""
    mistakes = _load()
    if not mistakes:
        return ""

    # Sort by count descending — most repeated mistakes are most important
    top = sorted(mistakes, key=lambda m: m.get("count", 1), reverse=True)[:MAX_INJECT]

    lines = ["## Mistakes you have made before — do NOT repeat these:\n"]
    for i, m in enumerate(top, 1):
        lines.append(f"{i}. PATTERN: {m['pattern']}")
        lines.append(f"   CAUSE:   {m['cause']}")
        lines.append(f"   FIX:     {m['fix']}")
        lines.append(f"   (seen {m.get('count', 1)} time(s))\n")

    return "\n".join(lines)


def record_mistake(pattern: str, cause: str, fix: str) -> None:
    """
    Record a new mistake or increment count if pattern already known.
    Called after a tool fails or user rejects an action.
    """
    mistakes = _load()

    # Check if this pattern already exists (simple substring match)
    for m in mistakes:
        if pattern.lower() in m["pattern"].lower() or m["pattern"].lower() in pattern.lower():
            m["count"] = m.get("count", 1) + 1
            m["last_seen"] = datetime.now().isoformat()
            _save(mistakes)
            return

    # New mistake — append it
    mistakes.append({
        "pattern": pattern,
        "cause": cause,
        "fix": fix,
        "count": 1,
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
    })
    _save(mistakes)


def list_mistakes() -> list[dict]:
    return _load()


def clear_mistakes() -> None:
    _save([])
