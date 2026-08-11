"""
agent.py — The Claude-style agent loop for Nexus v2.

Per turn:
  1. RAG: hybrid search (semantic + keyword) → relevant context block
  2. Thinking: optional Claude-style reasoning pass (auto mode)
  3. Stream the answer; parse tool call if the model reaches for one
  4. Confirm dangerous tools, execute, feed result back
  5. Token-aware context trimming (Qwen tokenizer)
  6. Persist: vector memory + JSONL dataset

Speed-first: single streaming pass, no critic loops. Self-heal only on
empty/garbage output.
"""

import re
from typing import Callable, Optional

import config
import dataset
import memory
import model
import thinking
import tokenizer
import tools as tool_module
from tools import ToolResult

TOOL_NAMES = list(tool_module.TOOLS.keys())

SYSTEM_TEMPLATE = """You are Nexus 2, an autonomous coding agent running fully offline.

Core behavior (act, don't explain):
- When the user asks you to list files, read a file, run a command,
  search the code, or create/edit code — DO IT with a tool call.
  Never give the user instructions for how they could do it themselves.
- One tool call per turn. After each tool result, continue working
  until the task is done, then give a short final answer.
- If a tool errors, adapt — do not repeat the identical call.
- Keep final answers under ~250 words. Confirm what you did, tersely —
  just say what you did, nothing else.

{rules}

{context_block}

{mistakes_block}

{tools_doc}
"""

RULES = """
Rules:
- Never invent file paths or command outputs. Use tools to get the truth.
- When asked to WRITE or CREATE a file, call write_file with the COMPLETE
  file content — never print the code and stop.
- When asked to run tests or commands, call run_shell and report the result.
- When asked to list files or directories, call list_tree.
- Never restate your plan — just do it.
- The user already sees tool results on screen. NEVER repeat tool output
  verbatim — summarize in 1-2 lines of your own words.
"""


class Agent:
    def __init__(self, ask_fn: Callable[[str], bool],
                 print_fn: Callable, store=None, ui=None):
        self.ask_fn = ask_fn
        self.print_fn = print_fn
        self.ui = ui                    # optional Claude-style UI helpers
        self.store = store              # vector store (set by main)
        self.history: list[dict] = []
        self.thinking_mode = config.THINKING_AUTO
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.plan_mode = False           # Claude Code plan mode (shift-tab)
        self.auto_approve = False        # --dangerously-skip-permissions
        self._prefill: Optional[str] = None   # forced tool-call opening

    # ── Intent detection (prefill steering) ───────────────────────────────

    def _steer(self, query: str) -> Optional[str]:
        """
        Claude Code-style tool forcing: classify the request in Python and
        return the partial tool XML to prefill, or None for plain chat.
        Skips if the tool was already called in this exchange.
        """
        import intent
        tool = intent.classify(query)
        if tool is None:
            return None
        if any(f"[called {tool}]" in m.get("content", "")
               for m in self.history):
            return None
        return intent.prefill(tool, query)

    # ── System prompt ─────────────────────────────────────────────────────

    def _system(self, query: str) -> str:
        context = self._rag_context(query) if self.store else ""
        mistakes = memory.get_mistakes_prompt()
        rules = RULES
        if self.plan_mode:
            rules += (
                "\n- PLAN MODE is ON: do NOT write or edit files and do NOT run "
                "commands that change anything. Produce a clear step-by-step "
                "plan the user can approve instead."
            )
        project = self._project_instructions()
        return SYSTEM_TEMPLATE.format(
            rules=rules,
            context_block=context or "## Workspace context:\n(none retrieved)",
            mistakes_block=mistakes or "## Past mistakes:\n(none yet)",
            tools_doc=tool_module.TOOL_DOC + project,
        )

    @staticmethod
    def _is_single_tool_ask(q: str) -> bool:
        """True for short, one-action asks (fast path eligible)."""
        q = q.lower()
        if len(q) > 80:
            return False
        return not any(k in q for k in (" and ", " then ", " also ",
                                        ", then", "; then"))

    @staticmethod
    def _fast_summary(tool_name: str, args: dict, result) -> str:
        if tool_name == "list_tree":
            lines = [l for l in result.output.splitlines() if l.strip()]
            n = max(0, len(lines) - 1)
            where = lines[0] if lines else args.get("root", "")
            word = "entry" if n == 1 else "entries"
            return f"Listed {where} — {n} {word}."
        if tool_name == "read_file":
            return f"Read {args.get('path', '')}."
        if tool_name == "search":
            n = len([l for l in result.output.splitlines() if l.strip()])
            word = "match" if n == 1 else "matches"
            return f"Search done — {n} {word}."
        return f"{tool_name} done."

    @staticmethod
    def _project_instructions() -> str:
        """CLAUDE.md support — the file Claude Code loads into the prompt."""
        try:
            for p in [Path.cwd(), *Path.cwd().parents][:4]:
                for name in ("CLAUDE.md", "NEXUS.md"):
                    f = p / name
                    if f.exists():
                        text = f.read_text(errors="ignore")[:3000]
                        return (f"\n\n## Project instructions ({f}):\n"
                                + text.strip())
        except Exception:
            pass
        return ""

    def _rag_context(self, query: str, top_k: int = config.RAG_TOP_K) -> str:
        try:
            from rag import embeddings
            from rag.vector_store import get_store
            store = self.store or get_store()
            if store.stats()["total_chunks"] == 0:
                return ""
            qv = embeddings.embed_one(query).tolist()
            hits = store.hybrid_search(query, qv, top_k=top_k)
            hits = [h for h in hits if h.score >= 0.012][:top_k]
            if not hits:
                return ""
            lines = ["## Relevant context from the knowledge base:"]
            for i, h in enumerate(hits, 1):
                lines.append(f"[{i}] {h.source} (score {h.score})")
                lines.append(h.text[:500])
                lines.append("")
            return "\n".join(lines)
        except Exception:
            return ""

    # ── Conversation management ────────────────────────────────────────────

    def _messages(self, query: str, reasoning: str = "") -> list[dict]:
        sys_content = self._system(query)
        if reasoning:
            sys_content += (
                "\n\n## Summary of reasoning (use it, don't restate it):\n"
                + reasoning
            )
        msgs = [{"role": "system", "content": sys_content}] + self.history
        msgs.append({"role": "user", "content": query})
        if self._prefill:
            msgs.append({"role": "assistant", "content": self._prefill})
        return tokenizer.trim_messages(msgs, budget=config.CONTEXT_BUDGET)

    # ── Tool parsing ───────────────────────────────────────────────────────

    @staticmethod
    def parse_tool_call(text: str) -> Optional[dict]:
        text = re.sub(r"```[\w]*\n?", "", text).strip()
        for name in TOOL_NAMES:
            m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
            if not m:   # 1.5B sometimes wraps tags in markdown brackets
                m = re.search(rf"\[{name}\]\s*(.*?)\s*\[/{name}\]",
                              text, re.DOTALL)
            if not m:
                continue
            args = {}
            if name == "multi_edit":
                body = m.group(1)
                blocks = re.findall(
                    r"<edit>\s*<old_string>(.*?)</old_string>\s*"
                    r"<new_string>(.*?)</new_string>\s*</edit>",
                    body, re.DOTALL)
                pm = re.search(r"<path>(.*?)</path>", body, re.DOTALL)
                if not blocks or not pm:
                    continue
                return {"tool": name, "args": {
                    "path": pm.group(1).strip(),
                    "edits": [{"old_string": o.strip(),
                               "new_string": n.strip()}
                              for o, n in blocks]}}
            for a in re.finditer(r"<(\w+)>(.*?)</\1>", m.group(1), re.DOTALL):
                args[a.group(1)] = a.group(2).strip()
            return {"tool": name, "args": args}
        return None

    @staticmethod
    def clean(text: str) -> str:
        for name in TOOL_NAMES:
            text = re.sub(rf"<{name}>.*?</{name}>", f"[called {name}]",
                          text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def summarize_tool(name: str, output: str, ok: bool) -> str:
        status = "succeeded" if ok else "failed"
        if len(output) <= config.MAX_TOOL_OUTPUT:
            return f"The {name} tool {status}. Output:\n{output}"
        half = config.MAX_TOOL_OUTPUT // 2
        return (f"The {name} tool {status}. Output "
                f"({len(output) - 2 * half} chars hidden):\n"
                + output[:half] + "\n…\n" + output[-half:])

    # ── Gating ─────────────────────────────────────────────────────────────

    def _gate(self, tool_name: str, args: dict) -> bool:
        if self.auto_approve and not self.plan_mode:
            return True
        if tool_name == "write_file":
            preview = args.get("content", "")[:200]
            more = "…" if len(args.get("content", "")) > 200 else ""
            head = "\n[PLAN MODE] " if self.plan_mode else "\n"
            return self.ask_fn(
                f"{head}Write '{args.get('path')}'?\n{preview}{more}\n[y/N] "
            )
        if tool_name == "run_shell":
            head = "[PLAN MODE] " if self.plan_mode else ""
            return self.ask_fn(f"\n{head}Run: $ {args.get('command')}\n[y/N] ")
        if tool_name in {"edit_file", "multi_edit"}:
            head = "[PLAN MODE] " if self.plan_mode else ""
            return self.ask_fn(f"\n{head}Edit '{args.get('path')}'?\n[y/N] ")
        return True

    def _exec(self, tool_name: str, args: dict) -> ToolResult:
        fn = tool_module.TOOLS.get(tool_name)
        if not fn:
            return ToolResult(False, f"Unknown tool: {tool_name}")
        try:
            return fn(**args)
        except TypeError as e:
            return ToolResult(False, f"Wrong args: {e}", mistake_hint=str(e))
        except Exception as e:
            return ToolResult(False, f"Crashed: {e}", mistake_hint=str(e))

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self, user_input: str) -> None:
        final_response = ""
        reasoning = ""
        used_tools: list[str] = []
        sources: list[str] = []

        # Thinking pass (Claude-style, auto-gated)
        if thinking.should_think(user_input, self.thinking_mode):
            context = self._rag_context(user_input, top_k=4) if self.store else ""
            if self.ui:
                self.ui.thinking_header()
            reasoning = thinking.think(user_input, context,
                                       print_fn=self.print_fn)
            if reasoning:
                self.print_fn("\n", dim=True)

        self.history.append({"role": "user", "content": user_input})

        # Claude Code-style tool forcing: prefill the matching tool call
        # so the model ACTS instead of explaining how the user could act.
        self._prefill = self._steer(user_input) if user_input else None
        fast_path = False   # True when this turn ran on a complete prefill

        for turn in range(config.MAX_TURNS):
            # Direct tool execution (Claude Code style): when the prefill
            # is a complete tool call, execute it now — no model round trip.
            direct = None
            if self._prefill:
                direct = self.parse_tool_call(self._prefill)

            messages = None
            full = ""
            usage = {}
            spinner = None
            got_first = False
            try:
                if direct is None:
                    messages = self._messages(user_input, reasoning)
                    self.print_fn("\n", dim=True)
                    if self.ui:
                        spinner = self.ui.start_spinner("Thinking")
                    for text, done, u in model.stream(messages):
                        if not got_first and spinner:
                            got_first = True
                            self.ui.stop_spinner(spinner)
                            spinner = None
                        if text:
                            self.print_fn(text)
                            full += text
                        usage = u or usage
                    if spinner:
                        self.ui.stop_spinner(spinner)
                    # Re-attach the prefilled opening (it was part of the
                    # request, not the stream) so parsing sees the full tag.
                    if self._prefill and not full.startswith(self._prefill):
                        full = self._prefill + full
                else:
                    full = self._prefill
                    fast_path = True
            except model.ModelError as e:
                if spinner:
                    self.ui.stop_spinner(spinner)
                if self.ui:
                    self.ui.error_box(str(e))
                else:
                    self.print_fn(f"\n[error] {e}\n", error=True)
                memory.record_mistake("model error", str(e),
                                      "Check Ollama is running: ollama serve")
                return

            self.total_tokens_in += usage.get("prompt_tokens", 0)
            self.total_tokens_out += usage.get("gen_tokens", 0)
            self.print_fn("\n", dim=True)
            if usage:
                if self.ui:
                    self.ui.usage_footer(
                        usage.get("prompt_tokens", 0), usage.get("gen_tokens", 0))
                else:
                    self.print_fn(tokenizer.usage_report(
                        usage.get("prompt_tokens", 0), usage.get("gen_tokens", 0)),
                        dim=True)

            tool_call = self.parse_tool_call(full)

            if tool_call is None:
                final_response = full
                self.history.append(
                    {"role": "assistant", "content": full[:2000]})
                break

            tool_name, args = tool_call["tool"], tool_call["args"]
            used_tools.append(tool_name)
            self._prefill = None

            # Mistake guard
            m = memory.matches_mistake(tool_name, args)
            if m and not self.ask_fn(
                f"\n[yellow]Known mistake: {m['pattern']}\n"
                f"Fix: {m['fix']} — proceed anyway? [y/N] "
            ):
                self.history.append(
                    {"role": "assistant", "content": self.clean(full)})
                self.history.append(
                    {"role": "user",
                     "content": "That matches a known mistake — try differently."})
                continue

            # Confirm gate
            if not self._gate(tool_name, args):
                self.history.append(
                    {"role": "assistant", "content": self.clean(full)})
                self.history.append(
                    {"role": "user",
                     "content": "User declined that action. Suggest an alternative."})
                continue

            # Execute
            if self.ui:
                self.ui.tool_use(tool_name, args)
            else:
                self.print_fn(
                    f"\n[tool] {tool_name}(" +
                    ", ".join(f"{k}={str(v)[:40]!r}" for k, v in args.items()) +
                    ")\n", dim=True)
            result = self._exec(tool_name, args)

            if not result.ok and result.mistake_hint:
                memory.record_mistake(
                    result.mistake_hint, f"{tool_name} error",
                    f"Check args for {tool_name}")

            if self.ui:
                if tool_name in {"edit_file", "multi_edit"}:
                    self.ui.show_edit_result(result)
                else:
                    self.ui.tool_result(result.ok, result.output)
            else:
                display = result.output[:120] + ("…" if len(result.output) > 120 else "")
                self.print_fn(f"[{'ok' if result.ok else 'error'}] {display}\n", dim=True)

            self.history.append({"role": "user",
                                 "content": self.summarize_tool(tool_name, result.output, result.ok)})

            # Fast path: one mechanical tool ask, done — skip the model
            # round trip that would only restate the tool output.
            if (fast_path and result.ok
                    and tool_name in {"list_tree", "read_file", "search"}
                    and self._is_single_tool_ask(user_input)):
                final_response = self._fast_summary(tool_name, args, result)
                self.print_fn(final_response + "\n")
                self.history.append({"role": "assistant",
                                     "content": final_response})
                break

            if turn == config.MAX_TURNS - 1:
                final_response = full
                if self.ui:
                    self.ui.error_box(
                        f"Reached {config.MAX_TURNS} turns — stopping.")
                else:
                    self.print_fn(
                        f"\n[nexus] Reached {config.MAX_TURNS} turns — stopping.\n",
                        error=True)
                break

        # ── Persist ─────────────────────────────────────────────────────────
        if final_response:
            dataset.save_pair(
                user_input, final_response.strip(),
                reasoning=reasoning,
                context_sources=sources,
            )
            if self.store:
                try:
                    from rag import embeddings
                    from rag.chunker import Chunk
                    qa = Chunk(
                        chunk_id=f"qa:{user_input[:30]}",
                        text=f"Q: {user_input}\nA: {final_response[:2000]}",
                        source="conversation",
                        metadata={},
                    )
                    self.store.upsert(
                        [qa],
                        [embeddings.embed_one(user_input).tolist()],
                    )
                except Exception:
                    pass

    def reset(self):
        self.history = []

    def set_thinking(self, mode: str):
        mode = mode.lower()
        self.thinking_mode = {"on": True, "off": False}.get(mode, "auto")
