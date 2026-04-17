"""
critic.py — Self-scoring and improvement loop for Nexus X.

Scores responses 0-10. Triggers improvement if score < 6.
Calibrated for small offline models — not GPT-4 standards.
"""

import re
from typing import Optional
import model


CRITIC_SYSTEM = """You are a response quality critic. Score from 0 to 10.

Scoring guide:
  9-10 = Perfect. Complete, correct, clear.
  7-8  = Good. Answers the question well with minor gaps.
  5-6  = Acceptable. Mostly correct but could be better.
  3-4  = Weak. Partially answers but missing key info.
  1-2  = Poor. Wrong, irrelevant, or refuses valid question.

Be fair — a short correct answer scores higher than a long wrong one.
A response that answers the question scores at least 6.

Respond with ONLY this format:
SCORE: <number>
REASON: <one sentence>
IMPROVE: <one fix, or 'none' if score >= 7>"""


IMPROVE_SYSTEM = """You are improving a previous response.
Fix the specific issue mentioned. Be concise.
Output only the improved response, nothing else."""


def score(user_msg: str, assistant_msg: str) -> dict:
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {user_msg[:300]}\n\nResponse:\n{assistant_msg[:600]}",
        },
    ]
    try:
        raw       = model.complete(messages, max_tokens=100, temperature=0.05)
        score_val = _parse_score(raw)
        reason    = _parse_field(raw, "REASON") or "no reason"
        improve   = _parse_field(raw, "IMPROVE") or "none"
        return {"score": score_val, "reason": reason, "improve": improve}
    except Exception as e:
        return {"score": -1.0, "reason": str(e), "improve": "none"}


def _parse_score(text: str) -> float:
    m = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        return min(10.0, max(0.0, float(m.group(1))))
    m2 = re.search(r"\b(10|[0-9])\b", text)
    if m2:
        return min(10.0, max(0.0, float(m2.group(1))))
    return -1.0


def _parse_field(text: str, field: str) -> Optional[str]:
    m = re.search(rf"{field}:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def improve(user_msg: str, assistant_msg: str, issue: str) -> str:
    if not issue or issue.lower() == "none":
        return assistant_msg
    messages = [
        {"role": "system", "content": IMPROVE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {user_msg[:300]}\n\n"
                f"Previous response:\n{assistant_msg[:600]}\n\n"
                f"Fix this: {issue}"
            ),
        },
    ]
    try:
        improved = model.complete(messages, max_tokens=800, temperature=0.1)
        return improved.strip() or assistant_msg
    except Exception:
        return assistant_msg


SCORE_THRESHOLD = 5.0   # only improve if really bad


def evaluate_and_improve(
    user_msg:      str,
    assistant_msg: str,
    print_fn=None,
) -> tuple[str, float]:
    result = score(user_msg, assistant_msg)
    s      = result["score"]

    if print_fn and s >= 0:
        # Only show score if it's below threshold — don't clutter good responses
        if s < SCORE_THRESHOLD:
            print_fn(f"\n[critic] score {s}/10 — {result['reason']}\n", dim=True)

    if s < 0 or s >= SCORE_THRESHOLD:
        return assistant_msg, s

    if print_fn:
        print_fn(f"[critic] improving response...\n", dim=True)

    improved = improve(user_msg, assistant_msg, result["improve"])
    result2  = score(user_msg, improved)
    s2       = result2["score"]

    if s2 > s:
        if print_fn:
            print_fn(f"[critic] improved: {s}/10 → {s2}/10\n", dim=True)
        return improved, s2

    return assistant_msg, s
