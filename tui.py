"""
tui.py — opencode-style full-screen TUI for Nexus (Textual).

Layout: header · chat log (blocks: user, tool panes, diffs,
assistant text) · spinner + input box · footer with keybindings.

The agent loop runs in a worker thread; every render goes through
call_from_thread so the UI stays responsive. Confirmations use a
blocking modal question.
"""

import itertools
import sys
import threading

from rich import box
from rich.panel import Panel
from rich.text import Text

from textual.app import App
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, Static

import config

ACCENT = "#c084fc"

TOOL_COLORS = {
    "run_shell":  "green",
    "read_file":  "cyan",
    "write_file": "yellow",
    "edit_file":  "magenta",
    "multi_edit": "magenta",
    "search":     "blue",
    "list_tree":  "dim white",
}


class _Spinner(Static):
    """Animated status line shown while the model is busy."""

    def start(self, message: str):
        self._frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self._msg = message
        self._timer = self.set_interval(1 / 12, self._tick)
        self._tick()

    def _tick(self):
        ch = next(self._frames)
        self.update(Text(f" {ch} {self._msg}", style="italic " + ACCENT))

    def stop(self):
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
            self._timer = None
        self.update(Text(""))


class _ConfirmScreen(ModalScreen):
    """Modal question for tool confirmations (y/N in the input box)."""

    BINDINGS = [
        Binding("escape", "no", "Maybe not", show=False),
        Binding("ctrl+c", "no", "No", show=False),
    ]

    def __init__(self, prompt: str, evt, box: dict):
        super().__init__()
        self._prompt = prompt
        self._evt = evt
        self._box = box
        self._input = Input(placeholder="y / n")

    def compose(self):
        from textual.containers import Center, Vertical

        with Vertical():
            yield Static(Text(self._prompt, style="bold yellow"),
                         markup=False, id=None)
            yield Center(self._input)

    def on_mount(self):
        self._input.focus()

    def _answer(self, ok: bool):
        self._box["ok"] = ok
        self._evt.set()
        self.dismiss()

    def on_input_submitted(self, event: Input.Submitted):
        event.stop()
        v = event.value.strip().lower()
        self._answer(v in ("y", "yes"))

    def action_no(self):
        self._answer(False)


class NexusApp(App):
    """opencode-shaped Nexus: chat log + input line + status + footer."""

    TITLE = "NEXUS"
    SUB_TITLE = f"{config.CHAT_DISPLAY} · 100% offline"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("tab", "cycle_complete", "Complete"),
        Binding("escape", "cancel_modal", "Cancel", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.chat: VerticalScroll = None
        self.input: Input = None
        self.spinner: _Spinner = None
        self._agent = None
        self._stream = Text()
        self._stream_static: Static = None
        self._history: list[str] = []
        self._hpos = -1
        self._complete = []
        self._complete_idx = -1
        self._busy = False
        self._pending: list[str] = []

    # ── Layout ──────────────────────────────────────────────────────────

    def compose(self):
        self.chat = VerticalScroll(id="chat")
        yield self.chat
        self.spinner = _Spinner(expand=True, markup=False)
        yield self.spinner
        self.input = Input(placeholder="Ask me anything — code, files, questions.  (/help)",
                           id="ask")
        yield self.input
        yield Footer()

    def on_mount(self):
        self._ui_thread = threading.get_ident()
        self.input.focus()
        threading.Thread(target=self._boot, daemon=True).start()

    # ── Rendering (main thread only) ────────────────────────────────────

    def go(self, fn):
        """Run fn on the UI thread from any thread."""
        self._run_on_ui(fn)

    def _run_on_ui(self, fn):
        if threading.get_ident() == self._ui_thread:
            fn()
        else:
            self.call_from_thread(fn)

    def _block(self, renderable, classes: str = ""):
        static = Static(renderable, markup=False, classes=classes)
        self.chat.mount(static)
        self.call_after_refresh(self.chat.scroll_end, animate=False, speed=50)

    def _line(self, text: str):
        self._block(Text(text, style="dim"))

    def _user_bubble(self, text: str):
        panel = Panel(Text(f"›  {text}", style="bold " + ACCENT),
                      box=box.ROUNDED, border_style=ACCENT, padding=(0, 1))
        self._block(panel, classes="user")

    def _assistant_chunk(self, text: str):
        """Stream assistant text into the live block, flushing on newlines."""
        if self._stream_static is None or not self._stream_static.is_attached:
            self._stream = Text()
            self._stream_static = Static(self._stream, markup=False)
            self.chat.mount(self._stream_static)
        self._stream.append(text)
        if text.endswith("\n") or "\n" in text:
            self._stream_static.update(self._stream)
            self.call_after_refresh(self.chat.scroll_end, animate=False, speed=50)

    def _finish_assistant(self):
        if self._stream_static is not None and self._stream_static.is_attached:
            self._stream_static.update(self._stream)
        self._stream_static = None
        self._stream = Text()

    # ── Boot sequence (worker thread) ───────────────────────────────────

    def _show_banner(self):
        """NEXUS gradient banner card, shown first on launch."""
        import cli as cli_mod
        lines = cli_mod.BANNER.strip("\n").splitlines()
        art = Text()
        for i, line in enumerate(lines):
            art.append(line, style=cli_mod.GRADIENT[i % len(cli_mod.GRADIENT)])
            art.append("\n")
        self._block(Panel(art, box=box.ROUNDED, border_style=ACCENT,
                          padding=(0, 2)), classes="banner")
        self._block(Text("Nexus 2 · 100% offline · Tab completes /commands · /help",
                         style="dim italic"), classes="banner")

    def _boot(self):
        import cli
        import memory
        import model
        from agent import Agent
        from rag.vector_store import get_store

        self.go(self._show_banner)

        if not model.is_running():
            self.go(lambda: self._block(
                Panel(Text(" ✘  Ollama is not running — start it with: ollama serve",
                           style="bold red"),
                      box=box.ROUNDED, border_style="red")))
            self.go(lambda: self.exit(1))
            return
        if not model.is_model_available():
            self.go(lambda: self._block(
                Panel(Text(f" ✘  Model '{config.CHAT_MODEL}' not pulled — "
                           f"run: ollama pull {config.CHAT_MODEL}\n"
                           f"     ollama pull {config.EMBED_MODEL}",
                           style="bold red"),
                      box=box.ROUNDED, border_style="red")))
            self.go(lambda: self.exit(1))
            return

        store = None
        rag = None
        try:
            store = get_store()
            stats = store.stats()
            rag = (f"{stats['backend']} · {stats['total_chunks']} chunks"
                   if stats["total_chunks"] else stats["backend"])
        except Exception as e:
            self.go(lambda: self._line(f"RAG disabled: {e}"))

        self.go(lambda: self.spinner.start("Warming up model"))
        t0 = __import__("time").time()
        model.warmup()
        elapsed = __import__("time").time() - t0
        self.go(self.spinner.stop)


        from pathlib import Path
        mistakes = memory.list_mistakes()

        ui = TuiUi(self)
        self._agent = Agent(ask_fn=ui.ask_confirm, print_fn=ui.print_chunk,
                            store=store, ui=ui)

        def card():
            rows = [("model     ", model.get_short_name()),
                    ("workspace ", str(Path.cwd()))]
            if rag:
                rows.append(("rag       ", rag))
            rows.append(("ready     ", f"{elapsed:.1f}s · 100% offline"))
            body = Text()
            for label, value in rows:
                body.append(f"  {label}", style="bold " + ACCENT)
                body.append(value)
                body.append("\n")
            body.rstrip()
            self._block(Panel(body, box=box.ROUNDED, border_style=ACCENT,
                              padding=(0, 1)), classes="session")
            if mistakes:
                self._block(Text(
                    f"{len(mistakes)} mistake(s) in memory — injected into prompts.",
                    style="dim yellow"), classes="session")
                self.go(card)
        self.go(self.spinner.stop)

        if self._pending:
            for q in self._pending:
                self.go(lambda q=q: self._run_turn(q))
            self._pending.clear()

    # ── Input ───────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted):
        event.stop()
        q = event.value.strip()
        if not q or self._busy:
            return
        self._history.append(q)
        self._hpos = len(self._history)
        if not q.startswith("/"):
            self.go(lambda: self._user_bubble(q))
        if self._agent is None:
            self._pending.append(q)
            self.go(lambda: self._line("(queued — Nexus is still starting up)"))
            return
        self._busy = True
        self._handle(q)

    def _handle(self, q: str):
        if q in ("/exit", "/quit"):
            self._busy = False
            self.exit()
            return
        if q in ("/clear", "/reset"):
            self.chat.remove_children()
            self._agent.reset()
            self._busy = False
            return
        if q == "/help":
            self.go(self._show_help)
            self._busy = False
            return
        if q == "/status":
            self.go(self._show_status)
            self._busy = False
            return
        if q.startswith("/model "):
            name = q[7:].strip()
            self._busy = False
            self.go(lambda: self._line(f"Switch model with: /model {name} "
                                         "(restart to apply)"))
            return
        if q == "/undo":
            import tools as toolz
            r = toolz.undo_last()
            self.go(lambda: self._line(f"undo: {r.output}" if r.ok
                                         else f"undo failed: {r.output}"))
            self._busy = False
            return
        self.input.clear()
        threading.Thread(target=self._run_turn, args=(q,), daemon=True).start()

    def _run_turn(self, q: str):
        self.go(lambda: self.spinner.start("Thinking"))
        try:
            self._agent.run(q)
        except Exception as e:
            self.go(lambda: self._block(
                Panel(Text(f" ✘  {e}", style="bold red"),
                      box=box.ROUNDED, border_style="red")))
        self.go(self._finish_assistant)
        self.go(self.spinner.stop)
        self._busy = False
        self.go(self.input.focus)

    def _show_help(self):
        import cli as cli_mod
        body = Text()
        for cmd, desc in cli_mod.COMMANDS.items():
            body.append(f"  {cmd:<12}", style="bold cyan")
            body.append(desc)
            body.append("\n")
        body.rstrip()
        self._block(Panel(body, title="[bold " + ACCENT + "]Commands[/bold "
                       + ACCENT + "]", box=box.ROUNDED, border_style=ACCENT,
                       padding=(0, 1)))

    def _show_status(self):
        import model
        body = Text()
        body.append(f"  model     ", style="bold " + ACCENT)
        body.append(model.get_short_name())
        body.append("\n")
        body.append(f"  workspace ", style="bold " + ACCENT)
        body.append(config.PROJECT_DIR if hasattr(config, "PROJECT_DIR")
                    else str(__import__("pathlib").Path.cwd()))
        self.go(lambda: self._block(Panel(body.rstrip(), box=box.ROUNDED,
                                            border_style=ACCENT,
                                            padding=(0, 1))))

    def _cycle_complete(self):
        if self._busy:
            return
        import cli as cli_mod
        v = self.input.value.lstrip()
        if not v.startswith("/"):
            return
        if self._complete_idx < 0 or not self._complete:
            self._complete = [c for c in cli_mod.SLASH_COMMANDS
                              if c.startswith(v)]
            self._complete_idx = -1
        if not self._complete:
            return
        self._complete_idx = (self._complete_idx + 1) % len(self._complete)
        self.input.value = self._complete[self._complete_idx]
        self.input.cursor_position = len(self.input.value)

    def action_cycle_complete(self):
        self._cycle_complete()

    def action_clear_log(self):
        if not self._busy:
            self.chat.remove_children()
            self._stream_static = None

    def action_cancel_modal(self):
        if isinstance(self.screen, _ConfirmScreen):
            self.screen.action_no()

    def on_key(self, event):
        if event.key in ("up", "down") and not self._busy \
                and not self.input.value:
            event.stop()
            self._history_nav(event.key)

    def _history_nav(self, key: str):
        if not self._history:
            return
        idx = self._hpos + (-1 if key == "up" else 1)
        idx = max(0, min(len(self._history), idx))
        self._hpos = idx
        self.input.value = self._history[idx] if idx < len(self._history) else ""


class TuiUi:
    """UI adapter used by the Agent loop (worker thread side)."""

    def __init__(self, app: NexusApp):
        self.app = app

    def go(self, fn):
        self.app.go(fn)

    # confirmations (blocking)
    def ask_confirm(self, prompt: str) -> bool:
        evt = threading.Event()
        holder = {"ok": False}
        self.app.go(lambda: self.app.push_screen(
            _ConfirmScreen(prompt, evt, holder)))
        evt.wait()
        return holder["ok"]

    # streaming reply text
    def print_chunk(self, text: str, dim=False, error=False, italic=False):
        if error:
            self.go(lambda: self.app._line(text))
        elif dim or italic:
            self.go(lambda: self.app._line(text))
        else:
            self.go(lambda: self.app._assistant_chunk(text))

    # status lines
    def print_status(self, msg: str, style: str = "dim"):
        self.go(lambda: self.app._line(f"[{style}] {msg}"))

    # thinking
    def thinking_header(self):
        self.go(lambda: self.app._block(
            Text("◤ Thinking ── reasoning pass", style="italic cyan")))

    # spinner (no-op — the TUI has its own)
    def start_spinner(self, message: str = "Thinking"):
        return {}

    def stop_spinner(self, spinner: dict):
        pass

    # tool panes
    def tool_use(self, tool_name: str, args: dict):
        import cli as cli_mod
        label = cli_mod.TOOL_LABELS.get(tool_name, tool_name)
        color = TOOL_COLORS.get(tool_name, "cyan")
        arg = ""
        for key in ("path", "root", "command"):
            if args.get(key):
                arg = str(args[key])
                break
        body = Text()
        body.append(f" {label}", style=f"bold {color}")
        if arg:
            body.append(f"  ", style="dim")
            body.append(arg, style="white")
        body.append(" ")
        self.go(lambda: self.app._block(
            Panel(body, box=box.ROUNDED, border_style=color, padding=(0, 1)),
            classes="tool"))

    def tool_result(self, ok: bool, output: str):
        lines = [l for l in output.splitlines() if l.strip()]
        if lines and lines[0] == "```":
            lines = lines[1:]
        while lines and lines[-1] == "```":
            lines = lines[:-1]
        if ok:
            mark = Text(" ✓", style="bold dim")
            body = Text()
            for line in lines[:12]:
                body.append(line[:200], style="dim")
                body.append("\n")
            if len(lines) > 12:
                body.append(f"… {len(lines) - 12} more lines", style="dim italic")
                body.append("\n")
            body.rstrip()
            body.append(mark)
            self.go(lambda: self.app._block(
                Panel(body or Text("(no output)", style="dim"),
                      box=box.ROUNDED, border_style="dim", padding=(0, 1))))
        else:
            err = " | ".join(lines[:2])[:150] if lines else "unknown error"
            self.go(lambda: self.app._block(
                Panel(Text(f" ✘  {err}", style="bold red"),
                      box=box.ROUNDED, border_style="red", padding=(0, 1))))

    def show_edit_result(self, result) -> None:
        if not result.ok:
            first = result.output.splitlines()[0][:150] \
                if result.output.splitlines() else "error"
            self.go(lambda: self.app._block(
                Panel(Text(f" ✘  {first}", style="bold red"),
                      box=box.ROUNDED, border_style="red", padding=(0, 1))))
            return
        first, _, diff = result.output.partition("\n")
        diff_lines = diff.splitlines()
        body = Text()
        body.append(f" {first[:130]}", style="bold white")
        body.append("\n")
        for i, line in enumerate(diff_lines):
            if i >= 40 and len(diff_lines) > 42:
                body.append(f" … {len(diff_lines) - 40} more lines",
                            style="dim italic")
                break
            if line.startswith(("+++", "---", "@@")):
                body.append(line, style="dim")
            elif line.startswith("+"):
                body.append(line, style="green")
            elif line.startswith("-"):
                body.append(line, style="red")
            else:
                body.append(line, style="dim")
            body.append("\n")
        body.rstrip()
        self.go(lambda: self.app._block(
            Panel(body, box=box.ROUNDED, border_style="magenta", padding=(0, 1)),
            classes="edit"))

    def error_box(self, msg: str):
        self.go(lambda: self.app._block(
            Panel(Text(f" ✘  {msg}", style="bold red"),
                  box=box.ROUNDED, border_style="red", padding=(0, 1))))

    def usage_footer(self, prompt_tokens: int, gen_tokens: int):
        self.go(lambda: self.app._line(
            f"· tokens: {prompt_tokens:,} in · {gen_tokens:,} out"))

    def show_files(self, result, path_label: str):
        rows = result.output.splitlines()
        body = Text()
        body.append(f" {rows[0]}", style="bold cyan")
        body.append("\n")
        for row in rows[1:]:
            body.append(row, style="dim white")
            body.append("\n")
        body.rstrip()
        self.go(lambda: self.app._block(
            Panel(body, box=box.ROUNDED, border_style="cyan", padding=(0, 1))))

    def show_mistakes(self, mistakes: list):
        if not mistakes:
            self.go(lambda: self.app._line("No mistakes recorded yet."))
            return
        body = Text()
        for i, m in enumerate(mistakes, 1):
            body.append(f" {i}. ", style="bold red")
            body.append(m["pattern"], style="yellow")
            body.append("\n")
            body.append(f"    cause  ", style="bold dim")
            body.append(m["cause"], style="dim")
            body.append("\n")
            body.append(f"    fix    ", style="bold green")
            body.append(m["fix"], style="green")
            body.append("\n\n")
        body.rstrip()
        self.go(lambda: self.app._block(
            Panel(body, box=box.ROUNDED, border_style="dim", padding=(0, 1))))

    def show_rag_hits(self, hits):
        if not hits:
            self.go(lambda: self.app._line("No relevant chunks found."))
            return
        body = Text()
        for h in hits:
            body.append(f" {h.source}", style="bold cyan")
            body.append(f"  ·  score {h.score}", style="dim")
            body.append("\n")
        body.rstrip()
        self.go(lambda: self.app._block(
            Panel(body, box=box.ROUNDED, border_style=ACCENT, padding=(0, 1))))


def run_tui() -> int:
    """Entry point: boots the full-screen TUI. Returns exit code."""
    app = NexusApp()
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(run_tui())