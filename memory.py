"""
memory.py — Mistake learning (same idea as v1, safer paths).

Records tool/model failures to mistakes.json, injects the top mistakes
into every system prompt, and blocks tool calls that match a known
mistake until the user confirms.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import config

MAX_INJECT = 8


def _load() -> list[dict]:
    if not config.MISTAKES_FILE.exists():
        return []
    try:
        return json.loads(config.MISTAKES_FILE.read_text())
    except Exception:
        return []


def _save(mistakes: list[dict]) -> None:
    config.MISTAKES_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.MISTAKES_FILE.write_text(json.dumps(mistakes, indent=2))


def get_mistakes_prompt() -> str:
    mistakes = _load()
    if not mistakes:
        return ""
    top = sorted(mistakes, key=lambda m: m.get("count", 1), reverse=True)[:MAX_INJECT]
    lines = ["## Mistakes to avoid:"]
    for i, m in enumerate(top, 1):
        lines.append(f"{i}. PATTERN: {m['pattern']}")
        lines.append(f"   FIX:     {m['fix']}")
    return "\n".join(lines)


def record_mistake(pattern: str, cause: str, fix: str) -> None:
    mistakes = _load()
    now = datetime.now().isoformat()
    for m in mistakes:
        if pattern.lower() in m["pattern"].lower() or m["pattern"].lower() in pattern.lower():
            m["count"] = m.get("count", 1) + 1
            m["last_seen"] = now
            _save(mistakes)
            return
    mistakes.append({
        "pattern": pattern, "cause": cause, "fix": fix,
        "count": 1, "first_seen": now, "last_seen": now,
    })
    _save(mistakes)


def list_mistakes() -> list[dict]:
    return _load()


def clear_mistakes() -> None:
    _save([])


def matches_mistake(tool_name: str, args: dict) -> Optional[dict]:
    """Return a matching mistake record, or None."""
    search = f"{tool_name} " + " ".join(str(v) for v in args.values()).lower()
    for m in _load():
        words = [w for w in m["pattern"].lower().split() if len(w) > 2]
        if words and all(w in search for w in words[:2]):
            return m
    return None
