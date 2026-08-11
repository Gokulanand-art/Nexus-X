"""
thinking.py — Claude-style extended thinking for a 2B model.

Small models hallucinate when forced straight to an answer. Phase 1 asks
the model to reason inside <reasoning> tags (streamed, rendered dim/italic
in the CLI like Claude's thinking pane); the best parts of that reasoning
are carried into phase 2 as a compact "summary of reasoning" — exactly the
pattern Claude uses, so the 2B model reaches much closer to its ceiling.

Speed: thinking is only used when it pays — auto mode triggers on
multi-step / code / debugging queries and stays off for chatty ones.
"""

import re
import sys

import config
import model

REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
# Auto mode thinks ONLY for genuinely complex work. Mechanical asks
# ("write a file", "run this") skip thinking — they're handled faster by
# tool prefill steering.
COMPLEX_MARKERS = [
    "debug", "why does", "why is", "explain how", "explain why",
    "compare", "design", "plan", "refactor", "optimize", "migrate",
    "architecture", "traceback", "root cause", "tradeoff", "best practice",
    "security", "performance",
]

THINK_SYSTEM = (
    "You are a careful reasoner. Before answering, plan briefly.\n"
    "Put your reasoning between <reasoning> and </reasoning> tags.\n"
    "Be terse: restate the goal in one line, list 2-4 concrete steps, "
    "note one likely pitfall. MAX 100 words. No code inside <reasoning>.\n"
    "End reasoning with 'therefore', then answer in plain text."
)


def should_think(query: str, mode: str = config.THINKING_AUTO) -> bool:
    """Decide whether this query deserves the thinking pass."""
    if mode is False or (isinstance(mode, str) and mode == "off"):
        return False
    if mode is True or (isinstance(mode, str) and mode == "on"):
        return True
    # auto
    q = query.lower()
    if len(q) > 200:
        return True
    return any(m in q for m in COMPLEX_MARKERS)


def think(
    query: str,
    context: str = "",
    budget: int = config.THINK_BUDGET,
    print_fn=None,
) -> str:
    """
    Run the reasoning pass. Returns the reasoning text (tag-stripped).
    Streams to print_fn when provided (dim, so it reads like Claude's
    thinking pane, not the answer).
    """
    user = query
    if context:
        user = f"Context:\n{context}\n\nTask:\n{query}"

    messages = [
        {"role": "system", "content": THINK_SYSTEM},
        {"role": "user", "content": user},
    ]

    if print_fn:
        print_fn("\n[dim italic]Thinking…[/dim italic]\n", dim=True)

    reasoning_parts, buf = [], ""
    for text, done, _ in model.stream(
        messages, max_tokens=budget, temperature=0.15
    ):
        buf += text
        cleaned = re.sub(r"</?reasoning>", "", buf)
        # Hold back a possible partial tag at the tail (<, </, </rea…)
        hold = 0
        tail = cleaned[-12:]
        for i in range(len(tail)):
            if re.match(r"</?[a-z]*$", tail[i:], re.IGNORECASE):
                hold = len(tail) - i
                break
        if hold:
            emit, buf = cleaned[:-hold], cleaned[-hold:]
        else:
            emit, buf = cleaned, ""
        if emit:
            reasoning_parts.append(emit)
            if print_fn:
                print_fn(emit, dim=True, italic=True)

    leftover = re.sub(r"</?reasoning>", "", buf).strip()
    if leftover:
        reasoning_parts.append(leftover)
        if print_fn:
            print_fn(leftover, dim=True, italic=True)
    return "".join(reasoning_parts).strip()


def summarize_reasoning(reasoning: str, max_chars: int = 700) -> str:
    """Compact version of the reasoning for the final answer pass."""
    r = reasoning.strip()
    if not r:
        return ""
    if len(r) <= max_chars:
        return r
    # Keep the head (goal framing) and tail (conclusion) — Claude-style.
    keep = max_chars // 2
    return r[:keep].rstrip() + "\n[…] " + r[-keep:].lstrip()
