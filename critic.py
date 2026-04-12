"""
critic.py — Self-improvement system for Nexus.

After every agent response:
  1. Send output to model with a critic prompt
  2. Get a score (1-10) + issues list
  3. If score < threshold → agent tries again with feedback
  4. Record low scores as mistakes in memory
"""

import re
import model
import memory

# ─── Config ───────────────────────────────────────────────────────────────────

CRITIC_THRESHOLD = 6    # score below this triggers a retry
MAX_CRITIC_LOOPS = 2    # max improvement attempts

# ─── Critic prompt ────────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a code reviewer. Analyze the response and reply in this exact format:

SCORE: <number 1-10>
ISSUES: <comma separated list of problems, or 'none'>
FIX: <one sentence describing what to improve, or 'none'>

Scoring guide:
10 - Perfect. Correct, complete, no placeholders.
8  - Good. Minor style issues only.
6  - Acceptable. Works but missing edge cases.
4  - Poor. Has bugs, placeholders, or wrong approach.
2  - Bad. Completely wrong or hallucinated.

Be strict. Placeholder values like {}, <path>, TODO count as bugs."""


def _parse_critic(text: str) -> dict:
    """Parse critic response into score, issues, fix."""
    score  = 5  # default middle score
    issues = "unknown"
    fix    = "none"

    m = re.search(r"SCORE:\s*(\d+)", text)
    if m:
        score = max(1, min(10, int(m.group(1))))

    m = re.search(r"ISSUES:\s*(.+)", text)
    if m:
        issues = m.group(1).strip()

    m = re.search(r"FIX:\s*(.+)", text)
    if m:
        fix = m.group(1).strip()

    return {"score": score, "issues": issues, "fix": fix}


# ─── Main critic function ─────────────────────────────────────────────────────

def critique(task: str, response: str) -> dict:
    """
    Review a task+response pair.
    Returns {score, issues, fix, passed}.
    """
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {
            "role": "user",
            "content": f"TASK: {task}\n\nRESPONSE:\n{response}"
        },
    ]

    try:
        raw = model.complete(messages, max_tokens=200, temperature=0.1)
        result = _parse_critic(raw)
    except Exception as e:
        return {"score": 5, "issues": str(e), "fix": "none", "passed": True}

    result["passed"] = result["score"] >= CRITIC_THRESHOLD

    # Record low scores as mistakes
    if not result["passed"]:
        memory.record_mistake(
            pattern=f"low quality response: {result['issues'][:80]}",
            cause=f"critic score {result['score']}/10",
            fix=result["fix"],
        )

    return result


# ─── Improvement loop ─────────────────────────────────────────────────────────

def improve(task: str, response: str, agent_fn) -> tuple[str, dict]:
    """
    Run critique → improve loop.
    agent_fn: callable(prompt) -> str  (the agent's generate function)
    Returns (final_response, final_critique).
    """
    current  = response
    critique_result = critique(task, current)

    for attempt in range(MAX_CRITIC_LOOPS):
        if critique_result["passed"]:
            break

        # Build improvement prompt
        improvement_prompt = (
            f"Your previous response scored {critique_result['score']}/10.\n"
            f"Issues: {critique_result['issues']}\n"
            f"Fix: {critique_result['fix']}\n\n"
            f"Original task: {task}\n\n"
            f"Rewrite your response fixing all issues. "
            f"No placeholders. No TODOs. Complete working code only."
        )

        try:
            current = agent_fn(improvement_prompt)
            critique_result = critique(task, current)
        except Exception:
            break

    return current, critique_result


# ─── Quick score (no improvement loop) ───────────────────────────────────────

def score_only(task: str, response: str) -> int:
    """Just return the numeric score. Fast check."""
    return critique(task, response)["score"]

