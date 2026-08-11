"""
cli.py — Claude Code-premium terminal UI. Streaming REPL, no full-screen TUI.

  gradient NEXUS banner + session card on startup
  > prompt with /-command tab completion
  ⠋ spinner while the model loads its first token
  ╭─ boxed tool-call panes (per-tool border colors) ─╮
  ◤ Thinking —, colored streaming reply, diff panes, token footer
"""

import itertools
import sys
import threading
import time

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config

console = Console(highlight=False)

BANNER = """
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██╔════╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    █████╗
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██╔══╝
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ███████╗
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝
"""

GRADIENT = ["bold #e879f9", "bold #c084fc", "bold #a78bfa",
            "bold #818cf8", "bold #60a5fa", "bold #22d3ee"]

SLASH_COMMANDS = [
    "/help", "/run", "/ingest", "/rag", "/think", "/memory",
    "/dataset", "/mistakes", "/clear", "/reset", "/plan",
    "/permissions", "/undo", "/init", "/model", "/status",
    "/cost", "/doctor", "/files", "/exit",
]

COMMANDS = {
    "/help":       "Show all commands",
    "/run":        "Autonomous mode: /run <goal>",
    "/ingest":     "RAG: /ingest <file|folder>",
    "/rag":        "Ask the knowledge base: /rag <question>",
    "/think":      "Thinking: /think on|off|auto",
    "/memory":     "Vector store stats",
    "/dataset":    "Training dataset stats",
    "/mistakes":   "Show recorded mistakes",
    "/clear":      "Clear conversation history",
    "/reset":      "Clear conversation history",
    "/plan":       "Plan mode (no changes without approval): /plan on|off",
    "/permissions": "Auto-approve tools: /permissions on|off",
    "/undo":       "Restore the last edited/written file from backup",
    "/init":       "Detect project and write CLAUDE.md",
    "/model":      "Switch model: /model <name>",
    "/status":     "Show model, RAG, project status",
    "/cost":       "Session token usage",
    "/doctor":     "Health checks for Ollama, RAG, tokenizer",
    "/files":      "List project files (git-aware): /files [path] [--all]",
    "/exit":       "Exit",
}

TOOL_LABELS = {
    "run_shell":  "Bash",
    "read_file":  "Read",
    "write_file": "Write",
    "edit_file":  "Edit",
    "multi_edit": "MultiEdit",
    "search":     "Grep",
    "list_tree":  "LS",
}

TOOL_COLORS = {
    "run_shell":  "green",
    "read_file":  "cyan",
    "write_file": "yellow",
    "edit_file":  "magenta",
    "multi_edit": "magenta",
    "search":     "blue",
    "list_tree":  "dim white",
}

ACCENT = "#c084fc"


def _arg_text(args: dict) -> str:
    for key in ("path", "root", "command"):
        if args.get(key):
            return str(args[key])
    return ""


# ─── Startup: banner + session card ────────────────────────────────────────

def print_banner():
    lines = BANNER.strip("\n").splitlines()
    art = Text()
    for i, line in enumerate(lines):
        art.append(line, style=GRADIENT[i % len(GRADIENT)])
        art.append("\n")
    console.print(Panel(art, box=box.ROUNDED, border_style=ACCENT,
                        padding=(0, 2), expand=False))
    console.print(Text(
        f"{config.CHAT_DISPLAY} · 100% offline · {config.store_backend()} RAG",
        style="dim italic", justify="center"))
    console.print(Text("Tab completes /commands · /help for all commands",
                       style="dim", justify="center"))
    console.print()


def session_start(info: dict):
    """Premium session card shown once the model is warm."""
    rows = [(f"model     ", info.get("model", "")),
            (f"workspace ", info.get("workspace", ""))]
    if info.get("rag"):
        rows.append((f"rag       ", info["rag"]))
    rows.append((f"ready     ", info.get("ready", "")))
    body = Text()
    for label, value in rows:
        body.append(f"  {label}", style="bold " + ACCENT)
        body.append(value, style="white" if value else "dim")
        body.append("\n")
    body.rstrip()
    console.print(Panel(body, box=box.ROUNDED,
                        border_style=ACCENT, padding=(0, 1), expand=False))
    if info.get("mistakes"):
        console.print(Text(f"{info['mistakes']} mistake(s) in memory — "
                           "injected into prompts", style="dim yellow"))
    console.print(Text("Ask me anything — code, files, questions.",
                       style="dim italic"))
    console.print()


def print_help():
    table = Table(box=box.SIMPLE_HEAVY, border_style="dim #7c3aed",
                  padding=(0, 1), show_header=False, title="Commands",
                  title_style="bold " + ACCENT, expand=False)
    for cmd, desc in COMMANDS.items():
        table.add_row(Text(f"{cmd:<12}", style="bold cyan"), Text(desc))
    console.print()
    console.print(table)
    console.print()


# ─── Input ─────────────────────────────────────────────────────────────────

try:
    import readline

    def _completer(text: str, state: int):
        if text.startswith("/"):
            matches = [c for c in SLASH_COMMANDS if c.startswith(text)]
        else:
            matches = []
        return matches[state] if state < len(matches) else None

    readline.set_completer(_completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
except Exception:
    pass


def get_input() -> str:
    try:
        console.print()
        return console.input(f"[bold {ACCENT}]>[/bold {ACCENT}] ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


def ask_confirm(prompt: str) -> bool:
    try:
        console.print()
        answer = console.input(
            f"[bold yellow]?[/bold yellow] [white]{prompt}[/white]"
        ).strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ─── Streaming output ──────────────────────────────────────────────────────

def _style_line(line: str) -> Text:
    """Light markdown-lite coloring for complete reply lines."""
    s = line.strip()
    if not s:
        return Text("")
    if s.startswith(("#", "---", "===")):
        return Text(line, style="bold cyan")
    if s.startswith("```"):
        return Text(line, style="dim")
    if s.startswith((">", "»")):
        return Text(line, style="italic green")
    return Text(line)


def print_output(text: str, dim: bool = False, error: bool = False,
                 italic: bool = False):
    if error:
        console.print(text, style="bold red", end="")
    elif dim or italic:
        console.print(text, style=("dim italic" if italic else "dim"), end="")
    elif text.endswith("\n"):
        # Complete lines only — streamed mid-line chunks stay plain.
        body = text[:-1].split("\n")
        for i, line in enumerate(body):
            console.print(_style_line(line), end="\n" if i < len(body) - 1 else "")
    else:
        print(text, end="", flush=True)


def print_status(msg: str, style: str = "dim"):
    console.print(f"[{style}]{msg}[/{style}]")


# ─── Spinner ───────────────────────────────────────────────────────────────

def start_spinner(message: str = "Thinking"):
    evt = threading.Event()
    frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    stop = {"evt": evt}

    def run():
        while not evt.is_set():
            sys.stdout.write(f"\r\033[38;5;141m{next(frames)} {message}\033[0m")
            sys.stdout.flush()
            time.sleep(0.08)

    stop["thread"] = threading.Thread(target=run, daemon=True)
    stop["thread"].start()
    return stop


def stop_spinner(spinner: dict):
    if not spinner:
        return
    spinner["evt"].set()
    spinner["thread"].join(timeout=0.3)
    sys.stdout.write("\r" + " " * 48 + "\r")
    sys.stdout.flush()


# ─── Claude Code-style tool panes ──────────────────────────────────────────

def tool_use(tool_name: str, args: dict):
    label = TOOL_LABELS.get(tool_name, tool_name)
    color = TOOL_COLORS.get(tool_name, "cyan")
    arg = _arg_text(args)
    body = Text()
    body.append(f" {label}", style=f"bold {color}")
    if arg:
        body.append(f"  ", style="dim")
        body.append(arg, style="white")
    body.append(" ")
    console.print()
    console.print(Panel(body, box=box.ROUNDED, border_style=color,
                        padding=(0, 1)))


def tool_result(ok: bool, output: str):
    lines = [l for l in output.splitlines() if l.strip()]
    if lines and lines[0] == "```":   # read_file fenced output — skip fence
        lines = lines[1:]
    while lines and lines[-1] == "```":
        lines = lines[:-1]
    if ok:
        body = Text()
        for line in lines[:12]:
            body.append(line[:200], style="dim")
            body.append("\n")
        if len(lines) > 12:
            body.append(f"… {len(lines) - 12} more lines", style="dim italic")
            body.append("\n")
        body.rstrip()
        console.print(Panel(body or Text("(no output)", style="dim"),
                            box=box.ROUNDED, border_style="dim",
                            padding=(0, 1)))
    else:
        err = " | ".join(lines[:2])[:150] if lines else "unknown error"
        console.print()
        console.print(Panel(Text(f" ✘  {err}"),
                            box=box.ROUNDED, border_style="red",
                            padding=(0, 1)))


def show_edit_result(result) -> None:
    """Claude Code-style edit pane: summary + color-coded diff."""
    ok, output = result.ok, result.output
    if not ok:
        first = output.splitlines()[0][:150] if output.splitlines() else "error"
        console.print()
        console.print(Panel(Text(f" ✘  {first}"),
                            box=box.ROUNDED, border_style="red", padding=(0, 1)))
        return
    first, _, diff = output.partition("\n")
    diff_lines = diff.splitlines()
    body = Text()
    body.append(f" {first[:130]}", style="bold white")
    body.append("\n")
    n = len(diff_lines)
    for i, line in enumerate(diff_lines):
        if i >= 40 and n > 42:
            body.append(f" … {n - 40} more lines", style="dim italic")
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
    console.print()
    console.print(Panel(body, box=box.ROUNDED, border_style="magenta",
                        padding=(0, 1)))


def thinking_header():
    t = Text()
    t.append("\n", style="")
    t.append("◤ ", style="bold cyan")
    t.append("Thinking", style="italic cyan")
    t.append(" ── reasoning pass", style="dim italic")
    console.print(t)


def error_box(msg: str):
    console.print()
    console.print(Panel(Text(f" ✘  {msg}", style="bold red"),
                        box=box.ROUNDED, border_style="red", padding=(0, 1)))


def usage_footer(prompt_tokens: int, gen_tokens: int):
    console.print(Text(
        f"· tokens: {prompt_tokens:,} in · {gen_tokens:,} out",
        style="dim"))


def show_files(result, path_label: str):
    """Render the /files listing like Claude Code's file view."""
    rows = result.output.splitlines()
    body = Text()
    body.append(f" {rows[0]}", style="bold cyan")
    body.append("\n")
    for row in rows[1:]:
        if row.startswith("  …"):
            body.append(row, style="dim")
        else:
            body.append(row, style="dim white")
        body.append("\n")
    body.rstrip()
    console.print()
    console.print(Panel(body, box=box.ROUNDED, border_style="cyan",
                        padding=(0, 1)))


def show_mistakes(mistakes: list[dict]):
    if not mistakes:
        console.print(Text("\nNo mistakes recorded yet.", style="dim italic"))
        return
    console.print()
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
        body.append("\n")
        body.append(f"    seen   ", style="bold dim")
        body.append(f"{m.get('count', 1)} time(s)", style="dim")
        body.append("\n\n")
    body.rstrip()
    console.print(Panel(body, title=f"[bold red]Recorded mistakes ({len(mistakes)})[/bold red]",
                        box=box.ROUNDED,
                        border_style="dim", padding=(0, 1)))


def show_rag_hits(hits):
    if not hits:
        console.print(Text("No relevant chunks found.", style="dim italic"))
        return
    console.print()
    parts: list = []
    for h in hits:
        head = Text()
        head.append(f" {h.source}", style="bold cyan")
        head.append(f"  ·  score {h.score}", style="dim")
        parts.append(head)
        parts.append(Markdown(h.text[:300]))
        parts.append(Text("\n\n"))
    console.print(Panel(Group(*parts[:-1]),
                        title=f"[bold {ACCENT}]Context ({len(hits)} chunks)[/bold {ACCENT}]",
                        box=box.ROUNDED,
                        border_style=ACCENT, padding=(0, 1)))