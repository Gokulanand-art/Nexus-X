"""
memory.py — Mistake learning system.

Loads mistakes.json at startup, injects top mistakes into every prompt,
and records new failures so the agent never repeats them.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Always store mistakes.json next to this script — never rely on cwd
MISTAKES_FILE = Path(__file__).resolve().parent / "mistakes.json"
MAX_INJECT    = 8


def _load() -> list[dict]:
    if not MISTAKES_FILE.exists():
        # Create it automatically if missing — no more FileNotFoundError
        _save([])
        return []
    try:
        return json.loads(MISTAKES_FILE.read_text())
    except Exception:
        return []


def _save(mistakes: list[dict]) -> None:
    # Ensure parent directory exists before writing
    MISTAKES_FILE.parent.mkdir(parents=True, exist_ok=True)
    MISTAKES_FILE.write_text(json.dumps(mistakes, indent=2))


def get_mistakes_prompt() -> str:
    mistakes = _load()
    if not mistakes:
        return ""
    top   = sorted(mistakes, key=lambda m: m.get("count", 1), reverse=True)[:MAX_INJECT]
    lines = ["## Mistakes to avoid:\n"]
    for i, m in enumerate(top, 1):
        lines.append(f"{i}. PATTERN: {m['pattern']}")
        lines.append(f"   FIX:     {m['fix']}\n")
    return "\n".join(lines)


def record_mistake(pattern: str, cause: str, fix: str) -> None:
    mistakes = _load()
    for m in mistakes:
        if pattern.lower() in m["pattern"].lower() or m["pattern"].lower() in pattern.lower():
            m["count"]     = m.get("count", 1) + 1
            m["last_seen"] = datetime.now().isoformat()
            _save(mistakes)
            return
    mistakes.append({
        "pattern":    pattern,
        "cause":      cause,
        "fix":        fix,
        "count":      1,
        "first_seen": datetime.now().isoformat(),
        "last_seen":  datetime.now().isoformat(),
    })
    _save(mistakes)


def list_mistakes() -> list[dict]:
    return _load()


def clear_mistakes() -> None:
    _save([])
