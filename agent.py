"""
agent.py — The single agent loop. Heart of Nexus.

Fixes in this version:
  - Context window trimming — old tool results are summarized, not kept raw
  - Tool output capped at 800 chars before injecting into history
  - Streaming timeout reduced to 45s — fail fast
  - Model response stripped of XML before storing in history
  - MAX_TURNS reduced to 8 — Phi-3 loses track beyond that
"""

import re
from typing import Optional

import memory
import model
import tools as tool_module
from tools import ToolResult, TOOL_SCHEMAS


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_TURNS          = 8    # Phi-3 Mini loses coherence beyond 8 turns
MAX_HISTORY_MSGS   = 10   # keep only last N messages in context (prevents timeout)
MAX_TOOL_OUTPUT    = 800  # chars of tool result injected into history
MAX_FILE_PREVIEW   = 600  # chars of file content shown to model


# ─── System prompt ────────────────────────────────────────────────────────────

BASE_SYSTEM = """You are Nexus, an offline AI coding assistant.

Rules:
- Use a tool when you need file info — never guess.
- Be concise — show code only, no explanations.
- Never use placeholder values like {} or <path> in code.
- One tool call per turn maximum.
- For write/shell actions, the user will confirm before execution.

{mistakes_block}

Tools (use XML tags exactly as shown):

<read_file>
  <path>filename.py</path>
</read_file>

<write_file>
  <path>filename.py</path>
  <content>
complete file content here
  </content>
</write_file>

<run_shell>
  <command>echo hello</command>
</run_shell>

<list_tree>
  <root>.</root>
</list_tree>

<search>
  <query>def main</query>
  <root>.</root>
  <file_pattern>*.py</file_pattern>
</search>

If no tool is needed, answer directly in plain text."""


def _build_system() -> str:
    mistakes = memory.get_mistakes_prompt()
    return BASE_SYSTEM.format(mistakes_block=mistakes or "")


# ─── Context trimmer ──────────────────────────────────────────────────────────

def _trim_history(conversation: list[dict]) -> list[dict]:
    """
    Keep only the last MAX_HISTORY_MSGS messages.
    This prevents context window overflow on Phi-3 Mini (4096 tokens).
    Always keeps the first user message so the model knows the original task.
    """
    if len(conversation) <= MAX_HISTORY_MSGS:
        return conversation

    # Always keep the first message (original task)
    first = conversation[0]
    recent = conversation[-(MAX_HISTORY_MSGS - 1):]
    return [first] + recent


def _build_messages(conversation: list[dict]) -> list[dict]:
    system = _build_system()
    trimmed = _trim_history(conversation)
    return [{"role": "system", "content": system}] + trimmed


# ─── Tool output summarizer ───────────────────────────────────────────────────

def _summarize_output(tool_name: str, output: str, ok: bool) -> str:
    """
    Cap tool output before injecting into history.
    Full output is shown to user — model only needs enough to continue.
    """
    status = "ok" if ok else "error"

    if len(output) <= MAX_TOOL_OUTPUT:
        return f"[{tool_name}:{status}]\n{output}"

    # For long file reads — show head and tail
    head = output[:MAX_FILE_PREVIEW // 2]
    tail = output[-(MAX_FILE_PREVIEW // 2):]
    skipped = len(output) - MAX_FILE_PREVIEW
    return (
        f"[{tool_name}:{status}] (output truncated — {skipped} chars hidden)\n"
        f"{head}\n...\n{tail}"
    )


# ─── Tool call parser ─────────────────────────────────────────────────────────

TOOL_NAMES = list(tool_module.TOOLS.keys())


def parse_tool_call(text: str) -> Optional[dict]:
    """Extract first XML tool call from model response."""
    # Strip markdown fences the model sometimes adds
    text = re.sub(r"```[\w]*\n?", "", text).strip()

    for tool_name in TOOL_NAMES:
        m = re.search(rf"<{tool_name}>(.*?)</{tool_name}>", text, re.DOTALL)
        if not m:
            continue
        args = {}
        for a in re.finditer(r"<(\w+)>(.*?)</\1>", m.group(1), re.DOTALL):
            args[a.group(1)] = a.group(2).strip()
        return {"tool": tool_name, "args": args}

    return None


# ─── Mistake guard ────────────────────────────────────────────────────────────

def mistake_guard(tool_name: str, args: dict) -> Optional[str]:
    mistakes = memory.list_mistakes()
    arg_vals = " ".join(str(v) for v in args.values()).lower()
    search   = f"{tool_name} {arg_vals}"

    for m in mistakes:
        pattern = m["pattern"].lower()
        if any(w in search for w in pattern.split() if len(w) > 2):
            return (
                f"[mistake guard] Resembles a known mistake:\n"
                f"  Pattern: {m['pattern']}\n"
                f"  Fix:     {m['fix']}\n"
                f"  (seen {m.get('count', 1)}x) Proceed anyway? [y/N] "
            )
    return None


# ─── Confirm gate ─────────────────────────────────────────────────────────────

CONFIRM_TOOLS = {"write_file", "run_shell"}


def confirm_gate(tool_name: str, args: dict, ask_fn) -> bool:
    if tool_name not in CONFIRM_TOOLS:
        return True

    if tool_name == "write_file":
        preview  = args.get("content", "")[:200]
        ellipsis = "..." if len(args.get("content", "")) > 200 else ""
        prompt   = (
            f"\n[nexus] Write to '{args.get('path')}'?\n"
            f"--- preview ---\n{preview}{ellipsis}\n"
            f"--- [y/N] "
        )
    else:
        prompt = f"\n[nexus] Run: $ {args.get('command')}\n  [y/N] "

    return ask_fn(prompt)


# ─── Execute tool ─────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, args: dict) -> ToolResult:
    fn = tool_module.TOOLS.get(tool_name)
    if not fn:
        return ToolResult(ok=False, output=f"Unknown tool: {tool_name}")
    try:
        return fn(**args)
    except TypeError as e:
        return ToolResult(
            ok=False,
            output=f"Wrong args for {tool_name}: {e}",
            mistake_hint=f"Wrong args for {tool_name}: {e}",
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            output=f"{tool_name} crashed: {e}",
            mistake_hint=f"{tool_name} raised {type(e).__name__}: {e}",
        )


# ─── Clean response for history ───────────────────────────────────────────────

def _clean_for_history(response: str) -> str:
    """
    Strip tool XML from assistant response before storing in history.
    The model doesn't need to re-read its own tool calls — just the outcome.
    Keeps the response short, preserving context window budget.
    """
    for name in TOOL_NAMES:
        response = re.sub(rf"<{name}>.*?</{name}>", f"[called {name}]", response, flags=re.DOTALL)
    return response.strip()


# ─── Main Agent ───────────────────────────────────────────────────────────────

class Agent:
    """
    Single agent loop.
    ask_fn:   fn(prompt: str) -> bool  for confirmation
    print_fn: fn(text, dim, error)     for output
    """

    def __init__(self, ask_fn, print_fn):
        self.ask_fn      = ask_fn
        self.print_fn    = print_fn
        self.conversation: list[dict] = []

    def reset(self):
        self.conversation = []

    def run(self, user_input: str) -> None:
        self.conversation.append({"role": "user", "content": user_input})

        for turn in range(MAX_TURNS):

            # ── 1. Build messages with trimmed history ────────────────────────
            messages = _build_messages(self.conversation)

            self.print_fn("\n[nexus] thinking...\n", dim=True)

            # ── 2. Stream model response ──────────────────────────────────────
            full_response = ""
            try:
                for chunk in model.stream_response(
                    messages,
                    max_tokens=1024,
                    temperature=0.2,
                ):
                    self.print_fn(chunk)
                    full_response += chunk

            except Exception as e:
                self.print_fn(f"\n[error] {e}\n", error=True)
                memory.record_mistake(
                    pattern="model inference timeout or error",
                    cause=str(e),
                    fix="Keep conversations shorter — use /reset between tasks",
                )
                return

            self.print_fn("\n")

            # ── 3. Parse tool call ────────────────────────────────────────────
            tool_call = parse_tool_call(full_response)

            if tool_call is None:
                # Final answer — store clean version in history
                self.conversation.append({
                    "role": "assistant",
                    "content": full_response[:1000],  # cap to preserve context
                })
                return

            tool_name = tool_call["tool"]
            args      = tool_call["args"]

            # ── 4. Mistake guard ──────────────────────────────────────────────
            warning = mistake_guard(tool_name, args)
            if warning:
                if not self.ask_fn(warning):
                    self.conversation.append({
                        "role": "assistant",
                        "content": _clean_for_history(full_response),
                    })
                    self.conversation.append({
                        "role": "user",
                        "content": "That approach matches a known mistake. Try differently.",
                    })
                    continue

            # ── 5. Confirm gate ───────────────────────────────────────────────
            if not confirm_gate(tool_name, args, self.ask_fn):
                self.conversation.append({
                    "role": "assistant",
                    "content": _clean_for_history(full_response),
                })
                self.conversation.append({
                    "role": "user",
                    "content": "User declined. Suggest an alternative.",
                })
                continue

            # ── 6. Execute tool ───────────────────────────────────────────────
            self.print_fn(
                f"\n[tool] {tool_name}("
                + ", ".join(f"{k}={repr(v)[:40]}" for k, v in args.items())
                + ")\n",
                dim=True,
            )

            result = execute_tool(tool_name, args)

            # ── 7. Record mistake on failure ──────────────────────────────────
            if not result.ok and result.mistake_hint:
                memory.record_mistake(
                    pattern=result.mistake_hint,
                    cause=f"{tool_name} returned error",
                    fix=f"Check args for {tool_name}",
                )

            status = "ok" if result.ok else "error"
            display = result.output[:120] + "..." if len(result.output) > 120 else result.output
            self.print_fn(f"[{status}] {display}\n", dim=True)

            # ── 8. Inject summarized result into history ───────────────────────
            # Store clean assistant turn (no XML bloat)
            self.conversation.append({
                "role": "assistant",
                "content": _clean_for_history(full_response),
            })
            # Store summarized tool result (not full file content)
            self.conversation.append({
                "role": "user",
                "content": _summarize_output(tool_name, result.output, result.ok),
            })
            # Loop → model sees trimmed result and decides next step

        self.print_fn(
            f"\n[nexus] Reached {MAX_TURNS} turns. Use /reset and try a more focused question.\n",
            error=True,
        )
