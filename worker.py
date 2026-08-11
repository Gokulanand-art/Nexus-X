"""
worker.py — Autonomous mode for Nexus v2 (/run <goal>).

Same architecture as v1 (planner → classifier → executor → verifier),
rewired for the new modules and the 1.5B model:
  - planner asks the model for a JSON step list (max 5)
  - each step runs through the Agent's tool pipeline (with confirmations)
  - failures retry twice, then get recorded as mistakes
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import memory
import model
import tools as tool_module
from tools import ToolResult

PLANNER_SYSTEM = """You are a task planner for a small offline agent.
Output ONLY a JSON array of step strings. Rules:
- Max 5 steps, strings only, no markdown, no explanation.
- Ordered: create files first, install second, run last.
- Real filenames only — never 'path/to' or 'placeholder'.
- No steps needing a second terminal or manual browser action.

Example for "flask app with /health":
["Create app.py with a Flask app and a /health route returning JSON", "Create requirements.txt with content: flask", "Run: pip install -r requirements.txt --break-system-packages", "Run: python -c \"import app; print('ok')\""]

Output the JSON array now:"""

EXECUTOR_SYSTEM = """You are a code executor. Respond with exactly ONE XML tool call.

<write_file>
  <path>FILENAME</path>
  <content>CONTENT</content>
</write_file>

<run_shell>
  <command>COMMAND</command>
</run_shell>

Rules:
- One tool call only, no explanation before/after.
- write_file: complete working code, never placeholders.
- If the step is already done, respond with the single word SKIP.

Step to execute: {step}

Respond now:"""

FAKE_PATTERNS = ["path/to", "your_", "example/", "placeholder", "<path>"]


@dataclass
class Step:
    index: int
    description: str
    status: str = "pending"
    result: str = ""
    attempts: int = 0


def _clean_path(path: str) -> str:
    for pat in FAKE_PATTERNS:
        if pat.lower() in path.lower():
            return path.split("/")[-1].strip() or path
    return path


def _parse_tool(text: str) -> Optional[dict]:
    text = re.sub(r"```[\w]*\n?", "", text).strip()
    for name in tool_module.TOOLS:
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


DANGER = [r"rm\s+-rf\s+/", r"dd\b.*of=/dev/", r"mkfs\b", r"shutdown", r"reboot"]


def _run_direct(command: str, timeout: int = 45) -> ToolResult:
    for d in DANGER:
        if re.search(d, command, re.IGNORECASE):
            return ToolResult(False, f"Blocked dangerous command: {command}")
    try:
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        return ToolResult(r.returncode == 0, out[:600] or "(no output)")
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Timed out after {timeout}s")
    except Exception as e:
        return ToolResult(False, str(e))


def _write_direct(path: str, content: str) -> ToolResult:
    from pathlib import Path
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(True, f"Written: {path} ({len(content)} chars)")
    except Exception as e:
        return ToolResult(False, str(e))


def _classify(desc: str) -> str:
    d = desc.lower()
    if any(x in d for x in ["second terminal", "different terminal",
                            "curl ", "open browser", "manually"]):
        return "skip"
    if any(x in d for x in ["run:", "execute:", "run pip", "pip install",
                            "python -c", "python -m", "npm install",
                            "git "]):
        return "shell"
    if "requirements.txt" in d and any(x in d for x in
                                       ["content:", "containing", "with content"]):
        return "requirements"
    return "model"


def _extract_command(desc: str) -> str:
    m = re.search(r"(?:run:|execute:)\s*(.+)", desc, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    if ":" in desc:
        return desc.split(":", 1)[1].strip()
    return desc.strip()


def _extract_requirements(desc: str) -> list[str]:
    m = re.search(r"(?:content:|containing|with content)[:\s]+([a-zA-Z0-9_,\- ]+)",
                  desc, re.IGNORECASE)
    if m:
        return [p.strip() for p in re.split(r"[,\s]+", m.group(1)) if p.strip()]
    known = ["flask", "fastapi", "uvicorn", "requests", "sqlalchemy", "django",
             "pandas", "numpy", "pytest", "pydantic", "httpx"]
    return [p for p in known if p in desc.lower()]


def plan_steps(goal: str) -> list[Step]:
    raw = model.complete(
        [{"role": "system", "content": PLANNER_SYSTEM},
         {"role": "user", "content": f"Goal: {goal}"}],
        max_tokens=400, temperature=0.05,
    )
    raw = re.sub(r"```[\w]*\n?", "", raw).strip()
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return [Step(1, goal)]
    try:
        items = json.loads(m.group())
        return [Step(i + 1, str(s).strip()) for i, s in enumerate(items)
                if isinstance(s, str) and s.strip()][:5]
    except Exception:
        return [Step(1, goal)]


def execute_step(step: Step, goal: str, print_fn, agent) -> tuple[bool, str]:
    strategy = _classify(step.description)

    if strategy == "skip":
        return True, "Skipped (manual step)"

    if strategy == "shell":
        cmd = _extract_command(step.description)
        print_fn(f"  $ {cmd}\n", dim=True)
        r = _run_direct(cmd, timeout=60)
        if not r.ok:
            memory.record_mistake(f"command failed: {cmd[:60]}",
                                  "Non-zero exit", "Check command syntax")
        return r.ok, r.output

    if strategy == "requirements":
        pkgs = _extract_requirements(step.description) or ["flask"]
        r = _write_direct("requirements.txt", "\n".join(pkgs) + "\n")
        return r.ok, r.output

    # strategy == "model" → agent tool pipeline with confirmation
    mistakes = memory.get_mistakes_prompt()
    system = EXECUTOR_SYSTEM.format(step=step.description)
    if mistakes:
        system += f"\n\nPast mistakes to avoid:\n{mistakes}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Execute this step for goal: {goal}"},
    ]

    # Tool forcing: prefill write_file when the step names a file
    import intent as intent_mod
    pf = ""
    if intent_mod.classify(step.description) == "write_file":
        pf = intent_mod.prefill("write_file", step.description)
        if pf:
            messages.append({"role": "assistant", "content": pf})

    for attempt in range(3):
        response = model.complete(messages, max_tokens=1200, temperature=0.1)
        if pf and not response.startswith(pf):
            response = pf + response
        if response.strip().upper()[:20] == "SKIP":
            return True, "Skipped — already done"
        tc = _parse_tool(response)
        if tc is None:
            if attempt < 2:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": "Respond with a tool call XML tag now."})
                continue
            return False, "No tool call after 3 attempts"
        tool_name, args = tc["tool"], tc["args"]
        if tool_name == "write_file":
            r = _write_direct(args.get("path", "output.py"), args.get("content", ""))
        elif tool_name == "run_shell":
            cmd = args.get("command", "")
            print_fn(f"  $ {cmd}\n", dim=True)
            r = _run_direct(cmd, timeout=30)
        else:
            fn = tool_module.TOOLS.get(tool_name)
            r = fn(**args) if fn else ToolResult(False, f"Unknown: {tool_name}")
        if not r.ok:
            memory.record_mistake(f"{tool_name} failed: {str(args)[:60]}",
                                  "Tool error", f"Fix args for {tool_name}")
            return False, r.output
        return True, r.output
    return False, "Could not execute step"


class Worker:
    MAX_RETRIES = 2

    def __init__(self, print_fn, ask_fn, agent=None):
        self.print_fn = print_fn
        self.ask_fn = ask_fn
        self.agent = agent

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

        for step in steps:
            step.status = "running"
            self._p(f"[step {step.index}/{len(steps)}] {step.description[:80]}\n")
            success = False
            strategy = _classify(step.description)
            for attempt in range(1, self.MAX_RETRIES + 2):
                step.attempts = attempt
                if attempt > 1:
                    self._p(f"  [retry {attempt - 1}]\n", dim=True)
                ok, result = execute_step(step, goal, self._p, self.agent)
                step.result = result
                if ok:
                    step.status = "done"
                    self._p(f"  [done] {result.strip().splitlines()[0][:100]}\n",
                            dim=True)
                    success = True
                    break
                if attempt <= self.MAX_RETRIES:
                    memory.record_mistake(f"step failed: {step.description[:50]}",
                                          f"attempt {attempt}: {result[:80]}",
                                          "Try different approach")
                    self._p(f"  [retry] {result[:80]}\n", dim=True)
            if not success:
                step.status = "failed"
                self._p("  [failed]\n", error=True)

        done = [s for s in steps if s.status == "done"]
        failed = [s for s in steps if s.status == "failed"]
        self._p("\n" + "─" * 48 + "\n")
        self._p(f"[worker] {len(done)}/{len(steps)} steps completed.\n")
        if failed:
            self._p("[worker] Failed steps:\n", error=True)
            for s in failed:
                self._p(f"  - {s.description[:70]}\n", error=True)
        else:
            self._p(f"[worker] Goal complete: {goal}\n")
        self._p("─" * 48 + "\n")
