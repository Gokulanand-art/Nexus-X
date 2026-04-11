"""
worker.py — Autonomous worker for Nexus.

Give it ONE goal. It plans, executes, verifies, recovers — alone.
"""

import re
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import memory
import model
import tools as tool_module
from tools import ToolResult


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Step:
    index:       int
    description: str
    status:      str = "pending"
    result:      str = ""
    attempts:    int = 0


@dataclass
class Plan:
    goal:  str
    steps: list[Step] = field(default_factory=list)


# ─── Prompts ──────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a task planner. Output ONLY a JSON array of steps.

Rules:
- JSON array of strings only — no markdown, no explanation, nothing else
- Max 5 steps
- File paths must be simple: app.py, requirements.txt, src/app.py
- NEVER write: path/to/, your_project/, build/, placeholder, <path>
- Steps must be ordered: create files first, install second, run last
- Do NOT include steps that require a second terminal or manual curl

Good output for "flask app with /health":
["Create app.py with Flask and a /health route returning JSON {status:ok}", "Create requirements.txt with content: flask", "Run: pip install -r requirements.txt --break-system-packages", "Run: python -c \\"import app; print('ok')\\""]

Output the JSON array now:"""


EXECUTOR_SYSTEM = """You are a code executor. You MUST respond with exactly one XML tool call.

Tools available:

<write_file>
  <path>FILENAME</path>
  <content>CONTENT</content>
</write_file>

<run_shell>
  <command>COMMAND</command>
</run_shell>

<read_file>
  <path>FILENAME</path>
</read_file>

Rules:
- Respond with ONE tool call only — no explanation before or after
- For write_file: write complete working code, never placeholders
- For run_shell: write the exact shell command
- Use real filenames: app.py not path/to/app.py
- If step is already done: respond with the single word SKIP

Step to execute: {step}

Respond now with exactly one tool call:"""


# ─── Path sanitizer ───────────────────────────────────────────────────────────

FAKE_PATTERNS = ["path/to", "your_", "example/", "placeholder", "<path>", "build/flask"]

def _clean_path(path: str) -> str:
    for pat in FAKE_PATTERNS:
        if pat.lower() in path.lower():
            return path.split("/")[-1].strip() or path
    return path


# ─── Tool call parser ─────────────────────────────────────────────────────────

TOOL_NAMES = list(tool_module.TOOLS.keys())

def _parse_tool(text: str) -> Optional[dict]:
    # Strip markdown fences if model wrapped the tool call
    text = re.sub(r"```[\w]*\n?", "", text).strip()

    for name in TOOL_NAMES:
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        if not m:
            continue
        args = {}
        for a in re.finditer(r"<(\w+)>(.*?)</\1>", m.group(1), re.DOTALL):
            args[a.group(1)] = a.group(2).strip()
        if "path" in args:
            args["path"] = _clean_path(args["path"])
        return {"tool": name, "args": args}
    return None


# ─── Direct tool execution (bypass model for known commands) ──────────────────

def _run_direct(command: str, timeout: int = 45) -> ToolResult:
    """Run a shell command directly — no model in the loop."""
    # Block dangerous patterns
    danger = [r"rm\s+-rf\s+/", r"dd\b.*of=/dev/", r"mkfs\b", r"shutdown", r"reboot"]
    for d in danger:
        if re.search(d, command, re.IGNORECASE):
            return ToolResult(ok=False, output=f"Blocked dangerous command: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=timeout
        )
        out = (result.stdout + result.stderr).strip()
        ok  = result.returncode == 0
        return ToolResult(ok=ok, output=out[:600] or "(no output)")
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output=f"Timed out after {timeout}s")
    except Exception as e:
        return ToolResult(ok=False, output=str(e))


# ─── Write file directly ──────────────────────────────────────────────────────

def _write_direct(path: str, content: str) -> ToolResult:
    """Write file directly — no model in the loop."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(ok=True, output=f"Written: {path} ({len(content)} chars)")
    except Exception as e:
        return ToolResult(ok=False, output=str(e))


# ─── Step classifier — decide HOW to execute each step ───────────────────────

def _classify_step(desc: str) -> str:
    """
    Return execution strategy for a step:
      'shell'  — direct shell command (pip install, python -c, etc.)
      'model'  — ask model to generate + write code
      'skip'   — ignore (manual steps, curl, etc.)
    """
    d = desc.lower()

    # Steps that need a second terminal or manual action — skip them
    if any(x in d for x in ["second terminal", "different terminal", "curl ", "open browser", "manually"]):
        return "skip"

    # Direct shell steps — extract and run the command
    if any(x in d for x in ["run:", "execute:", "run pip", "pip install", "python -c", "python -m"]):
        return "shell"

    # Writing a file with simple content (requirements.txt, .env, etc.)
    if "requirements.txt" in d and ("content:" in d or "containing" in d or "with content" in d):
        return "requirements"

    # Everything else — ask model to write code
    return "model"


def _extract_command(desc: str) -> str:
    """Pull the shell command out of a step description."""
    # Try "Run: <cmd>" or "Execute: <cmd>" pattern first
    m = re.search(r"(?:run:|execute:)\s*(.+)", desc, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    # Fallback — take everything after the first colon
    if ":" in desc:
        return desc.split(":", 1)[1].strip()
    return desc.strip()


def _extract_requirements(desc: str) -> list[str]:
    """Extract package names from a requirements step description."""
    # Look for "content: X" or "containing: X" or "with: X"
    m = re.search(r"(?:content:|containing|with content)[:\s]+([a-zA-Z0-9_,\- ]+)", desc, re.IGNORECASE)
    if m:
        pkgs = [p.strip() for p in re.split(r"[,\s]+", m.group(1)) if p.strip()]
        return pkgs
    # Fallback: scan for known package names
    known = ["flask", "fastapi", "uvicorn", "requests", "sqlalchemy", "django",
             "pandas", "numpy", "pytest", "pydantic", "httpx", "aiohttp"]
    return [p for p in known if p in desc.lower()]


# ─── Planner ──────────────────────────────────────────────────────────────────

def plan_steps(goal: str) -> list[Step]:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user",   "content": f"Goal: {goal}"},
    ]
    raw = model.complete(messages, max_tokens=400, temperature=0.05)
    raw = re.sub(r"```[\w]*\n?", "", raw).strip()

    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not m:
        return [Step(index=1, description=goal)]
    try:
        items = json.loads(m.group())
        return [
            Step(index=i+1, description=str(s).strip())
            for i, s in enumerate(items)
            if isinstance(s, str) and s.strip()
        ][:5]
    except Exception:
        return [Step(index=1, description=goal)]


# ─── Executor ─────────────────────────────────────────────────────────────────

def execute_step(step: Step, goal: str, print_fn) -> tuple[bool, str]:
    strategy = _classify_step(step.description)

    # ── Skip manual steps ────────────────────────────────────────────────────
    if strategy == "skip":
        return True, "Skipped (manual step not applicable in autonomous mode)"

    # ── Direct shell execution ───────────────────────────────────────────────
    if strategy == "shell":
        cmd = _extract_command(step.description)
        print_fn(f"  $ {cmd}\n", dim=True)
        result = _run_direct(cmd, timeout=60)
        if not result.ok:
            memory.record_mistake(
                pattern=f"command failed: {cmd[:60]}",
                cause="Non-zero exit code",
                fix=f"Check command syntax: {cmd}",
            )
        return result.ok, result.output

    # ── Write requirements.txt directly ──────────────────────────────────────
    if strategy == "requirements":
        pkgs = _extract_requirements(step.description)
        if not pkgs:
            pkgs = ["flask"]  # safe default
        content = "\n".join(pkgs) + "\n"
        result = _write_direct("requirements.txt", content)
        return result.ok, result.output

    # ── Model-generated code ──────────────────────────────────────────────────
    # strategy == "model"
    mistakes = memory.get_mistakes_prompt()
    system   = EXECUTOR_SYSTEM.format(step=step.description)
    if mistakes:
        system += f"\n\nPast mistakes to avoid:\n{mistakes}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Execute this step for goal: {goal}"},
    ]

    for attempt in range(3):
        response = model.complete(messages, max_tokens=1200, temperature=0.1)

        if "SKIP" in response.strip().upper()[:20]:
            return True, "Skipped — already done"

        tc = _parse_tool(response)

        if tc is None:
            # Model gave text instead of tool — try once more with stronger nudge
            if attempt < 2:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "You must respond with a tool call XML tag. Use <write_file> or <run_shell> now:"
                })
                continue
            else:
                return False, f"Model did not produce a tool call after 3 attempts"

        tool_name = tc["tool"]
        args      = tc["args"]

        # Execute directly for speed and reliability
        if tool_name == "write_file":
            result = _write_direct(args.get("path", "output.py"), args.get("content", ""))
        elif tool_name == "run_shell":
            cmd = args.get("command", "")
            print_fn(f"  $ {cmd}\n", dim=True)
            result = _run_direct(cmd, timeout=30)
        else:
            fn = tool_module.TOOLS.get(tool_name)
            result = fn(**args) if fn else ToolResult(ok=False, output=f"Unknown: {tool_name}")

        if not result.ok:
            memory.record_mistake(
                pattern=f"{tool_name} failed: {str(args)[:60]}",
                cause="Tool returned error",
                fix=f"Fix args for {tool_name}",
            )
            return False, result.output

        return True, result.output

    return False, "Could not execute step"


# ─── Verifier ─────────────────────────────────────────────────────────────────

def verify_step(step: Step, result: str, strategy: str) -> str:
    # Shell steps: trust the exit code we already captured
    if strategy in ("shell", "requirements", "skip"):
        if "error" in result.lower() or "failed" in result.lower() or "not found" in result.lower():
            return "RETRY"
        return "SUCCESS"

    # Model steps: quick check
    if "written:" in result.lower() or "chars)" in result.lower():
        return "SUCCESS"
    if "error" in result.lower() or "could not" in result.lower():
        return "RETRY"
    return "SUCCESS"


# ─── Main Worker ──────────────────────────────────────────────────────────────

class Worker:
    MAX_RETRIES = 2

    def __init__(self, print_fn, ask_fn):
        self.print_fn = print_fn
        self.ask_fn   = ask_fn

    def _p(self, text, **kw):
        self.print_fn(text, **kw)

    def run(self, goal: str) -> None:
        self._p(f"\n[worker] Goal: {goal}\n", dim=True)
        self._p("[worker] Planning...\n", dim=True)

        steps = plan_steps(goal)

        if not steps:
            self._p("[worker] Could not plan. Try rephrasing.\n", error=True)
            return

        self._p(f"\n[plan] {len(steps)} steps:\n")
        for s in steps:
            self._p(f"  {s.index}. {s.description}\n")
        self._p("\n")

        if not self.ask_fn("[worker] Proceed? [y/N] "):
            self._p("[worker] Cancelled.\n", dim=True)
            return

        self._p("\n")

        for step in steps:
            step.status = "running"
            self._p(f"[step {step.index}/{len(steps)}] {step.description[:80]}\n")

            strategy = _classify_step(step.description)
            success  = False

            for attempt in range(1, self.MAX_RETRIES + 2):
                step.attempts = attempt
                if attempt > 1:
                    self._p(f"  [retry {attempt-1}]\n", dim=True)

                ok, result = execute_step(step, goal, self._p)
                step.result = result

                verdict = verify_step(step, result, strategy)

                if verdict in ("SUCCESS", "SKIP"):
                    step.status = "done"
                    out = result.strip().split("\n")[0][:100]
                    self._p(f"  [done] {out}\n", dim=True)
                    success = True
                    break

                if attempt <= self.MAX_RETRIES:
                    memory.record_mistake(
                        pattern=f"step failed: {step.description[:50]}",
                        cause=f"attempt {attempt} failed: {result[:80]}",
                        fix="Try different approach",
                    )
                    self._p(f"  [retry] {result[:80]}\n", dim=True)
                else:
                    step.status = "failed"

            if not success:
                step.status = "failed"
                self._p(f"  [failed]\n", error=True)

        # Report
        done   = [s for s in steps if s.status == "done"]
        failed = [s for s in steps if s.status == "failed"]

        self._p("\n" + "─" * 48 + "\n")
        self._p(f"[worker] {len(done)}/{len(steps)} steps completed.\n")

        if failed:
            self._p(f"[worker] Failed steps:\n", error=True)
            for s in failed:
                self._p(f"  - {s.description[:70]}\n", error=True)
        else:
            self._p(f"[worker] Goal complete: {goal}\n")

        self._p("─" * 48 + "\n")
