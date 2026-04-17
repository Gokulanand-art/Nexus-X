"""
agent.py — The single agent loop for Nexus X.

Full pipeline per turn:
  1. Retrieve relevant past context from vector memory (RAG)
  2. Build system prompt with context + mistakes
  3. Stream response from model
  4. Parse + execute tool call (if any)
  5. Loop until final answer
  6. Score response with critic — improve if < 6/10
  7. Save turn to dataset
  8. Store in vector memory
"""

import re
from typing import Optional

import memory
import model
import tools as tool_module
from tools import ToolResult, TOOL_SCHEMAS

# Optional modules — graceful fallback if not installed
try:
    import learning
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

try:
    import critic
    HAS_CRITIC = True
except ImportError:
    HAS_CRITIC = False

try:
    import dataset
    HAS_DATASET = True
except ImportError:
    HAS_DATASET = False


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_TURNS        = 8
MAX_HISTORY_MSGS = 10
MAX_TOOL_OUTPUT  = 800
MAX_FILE_PREVIEW = 600
USE_CRITIC       = True   # set False to disable critic for speed


# ─── System prompt ────────────────────────────────────────────────────────────

BASE_SYSTEM = """You are Nexus X — a helpful AI assistant running fully offline.
You can help with anything: coding, science, math, writing, general questions.
You are like an offline ChatGPT — smart, friendly, and direct.
Always give real, helpful answers. Never refuse a general knowledge question.

{rag_context}

{mistakes_block}

For file/code tasks use these tools (XML tags). For general questions just answer directly.
One tool call per turn. Write/shell requires user confirmation.

Tools:

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

If no tool needed, answer in plain text."""


def _build_system(user_query: str = "") -> str:
    mistakes    = memory.get_mistakes_prompt()
    rag_context = ""

    if HAS_RAG and learning.is_available() and user_query:
        rag_context = learning.retrieve(user_query)

    return BASE_SYSTEM.format(
        rag_context    = rag_context or "",
        mistakes_block = mistakes   or "",
    )


# ─── Context trimmer ──────────────────────────────────────────────────────────

def _trim(conv: list[dict]) -> list[dict]:
    if len(conv) <= MAX_HISTORY_MSGS:
        return conv
    return [conv[0]] + conv[-(MAX_HISTORY_MSGS - 1):]


def _build_messages(conv: list[dict], query: str = "") -> list[dict]:
    return [{"role": "system", "content": _build_system(query)}] + _trim(conv)


# ─── Tool output summarizer ───────────────────────────────────────────────────

def _summarize(tool_name: str, output: str, ok: bool) -> str:
    status = "ok" if ok else "error"
    if len(output) <= MAX_TOOL_OUTPUT:
        return f"[{tool_name}:{status}]\n{output}"
    head = output[:MAX_FILE_PREVIEW // 2]
    tail = output[-(MAX_FILE_PREVIEW // 2):]
    skip = len(output) - MAX_FILE_PREVIEW
    return f"[{tool_name}:{status}] ({skip} chars hidden)\n{head}\n...\n{tail}"


# ─── Tool call parser ─────────────────────────────────────────────────────────

TOOL_NAMES = list(tool_module.TOOLS.keys())


def parse_tool_call(text: str) -> Optional[dict]:
    text = re.sub(r"```[\w]*\n?", "", text).strip()
    for name in TOOL_NAMES:
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        if not m:
            continue
        args = {}
        for a in re.finditer(r"<(\w+)>(.*?)</\1>", m.group(1), re.DOTALL):
            args[a.group(1)] = a.group(2).strip()
        return {"tool": name, "args": args}
    return None


# ─── Clean response for history ───────────────────────────────────────────────

def _clean(response: str) -> str:
    for name in TOOL_NAMES:
        response = re.sub(
            rf"<{name}>.*?</{name}>",
            f"[called {name}]",
            response,
            flags=re.DOTALL,
        )
    return response.strip()


# ─── Mistake guard ────────────────────────────────────────────────────────────

def mistake_guard(tool_name: str, args: dict) -> Optional[str]:
    mistakes = memory.list_mistakes()
    search   = f"{tool_name} " + " ".join(str(v) for v in args.values()).lower()
    for m in mistakes:
        pattern = m["pattern"].lower()
        if any(w in search for w in pattern.split() if len(w) > 2):
            return (
                f"[mistake guard] Resembles known mistake:\n"
                f"  Pattern: {m['pattern']}\n"
                f"  Fix:     {m['fix']}\n"
                f"  (seen {m.get('count',1)}x) Proceed? [y/N] "
            )
    return None


# ─── Confirm gate ─────────────────────────────────────────────────────────────

def confirm_gate(tool_name: str, args: dict, ask_fn) -> bool:
    if tool_name not in {"write_file", "run_shell"}:
        return True
    if tool_name == "write_file":
        preview = args.get("content", "")[:200]
        ellipsis = "..." if len(args.get("content", "")) > 200 else ""
        prompt  = f"\n[nexus] Write '{args.get('path')}'?\n{preview}{ellipsis}\n[y/N] "
    else:
        prompt = f"\n[nexus] Run: $ {args.get('command')}\n[y/N] "
    return ask_fn(prompt)


# ─── Execute tool ─────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, args: dict) -> ToolResult:
    fn = tool_module.TOOLS.get(tool_name)
    if not fn:
        return ToolResult(ok=False, output=f"Unknown tool: {tool_name}")
    try:
        return fn(**args)
    except TypeError as e:
        return ToolResult(ok=False, output=f"Wrong args: {e}", mistake_hint=str(e))
    except Exception as e:
        return ToolResult(ok=False, output=f"Crashed: {e}", mistake_hint=str(e))


# ─── Main Agent ───────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, ask_fn, print_fn):
        self.ask_fn   = ask_fn
        self.print_fn = print_fn
        self.conversation: list[dict] = []

    def reset(self):
        self.conversation = []

    def run(self, user_input: str) -> None:
        self.conversation.append({"role": "user", "content": user_input})
        final_response = ""

        for turn in range(MAX_TURNS):
            messages = _build_messages(self.conversation, user_input)
            self.print_fn("\n[nexus] thinking...\n", dim=True)

            # ── Stream response ───────────────────────────────────────────────
            full_response = ""
            try:
                for chunk in model.stream_response(messages, max_tokens=1024):
                    self.print_fn(chunk)
                    full_response += chunk
            except Exception as e:
                self.print_fn(f"\n[error] {e}\n", error=True)
                memory.record_mistake(
                    pattern="model timeout or error",
                    cause=str(e),
                    fix="Use /reset between long tasks",
                )
                return

            self.print_fn("\n")

            # ── Parse tool call ───────────────────────────────────────────────
            tool_call = parse_tool_call(full_response)

            if tool_call is None:
                # Final answer
                final_response = full_response
                self.conversation.append({
                    "role":    "assistant",
                    "content": full_response[:1000],
                })
                break

            tool_name = tool_call["tool"]
            args      = tool_call["args"]

            # ── Mistake guard ─────────────────────────────────────────────────
            warning = mistake_guard(tool_name, args)
            if warning and not self.ask_fn(warning):
                self.conversation.append({"role": "assistant", "content": _clean(full_response)})
                self.conversation.append({"role": "user", "content": "That matches a known mistake. Try differently."})
                continue

            # ── Confirm gate ──────────────────────────────────────────────────
            if not confirm_gate(tool_name, args, self.ask_fn):
                self.conversation.append({"role": "assistant", "content": _clean(full_response)})
                self.conversation.append({"role": "user", "content": "User declined. Suggest an alternative."})
                continue

            # ── Execute ───────────────────────────────────────────────────────
            self.print_fn(
                f"\n[tool] {tool_name}(" +
                ", ".join(f"{k}={repr(v)[:40]}" for k, v in args.items()) +
                ")\n", dim=True,
            )

            result = execute_tool(tool_name, args)

            if not result.ok and result.mistake_hint:
                memory.record_mistake(
                    pattern=result.mistake_hint,
                    cause=f"{tool_name} error",
                    fix=f"Check args for {tool_name}",
                )

            status  = "ok" if result.ok else "error"
            display = result.output[:120] + "..." if len(result.output) > 120 else result.output
            self.print_fn(f"[{status}] {display}\n", dim=True)

            # ── Inject result into history ────────────────────────────────────
            self.conversation.append({"role": "assistant", "content": _clean(full_response)})
            self.conversation.append({"role": "user",      "content": _summarize(tool_name, result.output, result.ok)})

        # ── Post-turn: critic + dataset + memory ──────────────────────────────
        if final_response:
            improved   = final_response
            score_val  = -1.0

            # Critic — score and optionally improve
            if USE_CRITIC and HAS_CRITIC:
                improved, score_val = critic.evaluate_and_improve(
                    user_input, final_response, self.print_fn
                )
                if improved != final_response:
                    self.print_fn("\n[critic] improved response:\n" + improved + "\n")

            # Dataset — save this turn
            if HAS_DATASET:
                dataset.save_pair(user_input, improved, quality=score_val)

            # Vector memory — store for future RAG retrieval
            if HAS_RAG and learning.is_available():
                learning.store(user_input, improved)

        if turn == MAX_TURNS - 1:
            self.print_fn(
                f"\n[nexus] Reached {MAX_TURNS} turns. Use /reset for a new task.\n",
                error=True,
            )
