"""
cli.py — Claude Code-style terminal UI.

  > prompt with /-command tab completion
  ⠋ spinner while the model loads its first token
  ── Tool Use ── headers with ⎿ result rows (Claude Code look)
  dim italic thinking pane, token usage footer
"""

import itertools
import sys
import threading
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

import config

console = Console(highlight=False)

BANNER = """
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗██╗   ██╗███████╗    ██╗    ██╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██║    ██║
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ██║ █╗ ██║
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██║███╗██║
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ╚███╔███╔╝
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝     ╚══╝╚══╝
"""

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


def print_banner():
    console.print(Text(BANNER, style="bold purple"))
    console.print(
        f"[dim]Nexus 2 · {config.CHAT_DISPLAY} · offline · "
        f"{config.store_backend()} RAG[/dim]\n"
        "[dim]type /help for commands · Tab completes /commands[/dim]",
        style="none",
    )
    console.print()


def print_help():
    console.print("\n[bold]Commands:[/bold]")
    for cmd, desc in COMMANDS.items():
        console.print(f"  [cyan]{cmd:<12}[/cyan] {desc}")
    console.print()


# ─── Input with /-completion ───────────────────────────────────────────────

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
        return console.input("[bold purple]>[/bold purple] ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


def ask_confirm(prompt: str) -> bool:
    try:
        console.print()
        answer = console.input(f"[yellow]{prompt}[/yellow]").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ─── Streaming output ──────────────────────────────────────────────────────

def print_output(text: str, dim: bool = False, error: bool = False,
                 italic: bool = False):
    if error:
        console.print(text, style="bold red", end="")
    elif dim or italic:
        console.print(text, style=("dim italic" if italic else "dim"), end="")
    else:
        print(text, end="", flush=True)


def print_status(msg: str, style: str = "dim"):
    console.print(f"[{style}]{msg}[/{style}]")


# ─── Spinner (before the first token) ──────────────────────────────────────

def start_spinner(message: str = "Thinking"):
    evt = threading.Event()
    frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    stop = {"evt": evt}

    def run():
        while not evt.is_set():
            sys.stdout.write(f"\r\033[90m{next(frames)} {message}\033[0m")
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
    arg = args.get("path") or args.get("root") or args.get("command") or ""
    summary = f"{label}  {arg}"
    console.print(f"\n[cyan]── {summary} [/cyan]"
                  f"[dim]{'─' * max(4, 48 - len(summary))}[/dim]")
    console.print(f"[cyan]⎿  {summary}[/cyan]")


def tool_result(ok: bool, output: str):
    lines = [l for l in output.splitlines() if l.strip()]
    if lines and lines[0] == "```":   # read_file fenced output — skip fence
        lines = lines[1:]
    if ok:
        first = lines[0][:130] if lines else "(no output)"
        console.print(f"[dim]⎿  {first}[/dim]")
    else:
        err = " | ".join(lines[:2])[:150] if lines else "unknown error"
        console.print(f"[bold red]✘ {err}[/bold red]")


def show_edit_result(result) -> None:
    """Claude Code-style edit preview: header + color-coded diff."""
    ok, output = result.ok, result.output
    if not ok:
        console.print(f"[bold red]✘ {output.splitlines()[0][:150]}[/bold red]")
        return
    first, _, diff = output.partition("\n")
    console.print(f"[cyan]⎿  {first[:130]}[/cyan]")
    n = 0
    for line in diff.splitlines():
        if n >= 40 and len(diff.splitlines()) > 42:
            console.print(f"[dim]… {len(diff.splitlines()) - 40} more lines[/dim]")
            break
        if line.startswith(("+++", "---")):
            console.print(f"[dim]{line}[/dim]")
        elif line.startswith("+"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"[red]{line}[/red]")
        elif line.startswith("@@"):
            console.print(f"[dim yellow]{line}[/dim yellow]")
        else:
            console.print(f"[dim]{line}[/dim]")
        n += 1


def thinking_header():
    console.print("\n[dim italic]── Thinking ──[/dim italic]")


def error_box(msg: str):
    console.print(f"[bold red]── Error ──[/bold red]\n[red]{msg}[/red]")


def usage_footer(prompt_tokens: int, gen_tokens: int):
    console.print(f"[dim]tokens: {prompt_tokens:,} in · {gen_tokens:,} out[/dim]")


def show_files(result, path_label: str):
    """Render the /files listing like Claude Code's file view."""
    rows = result.output.splitlines()
    console.print(f"\n[bold]Files[/bold] [dim]({rows[0]})[/dim]")
    for row in rows[1:]:
        if row.startswith("  …"):
            console.print(row)
        else:
            console.print(f"[dim]{row}[/dim]")
    if not rows[1:]:
        console.print("[dim]  (empty)[/dim]")


def show_mistakes(mistakes: list[dict]):
    if not mistakes:
        console.print("[dim]No mistakes recorded yet.[/dim]")
        return
    console.print(f"\n[bold]Recorded mistakes ({len(mistakes)}):[/bold]\n")
    for i, m in enumerate(mistakes, 1):
        console.print(f"[bold red]{i}.[/bold red] [yellow]{m['pattern']}[/yellow]")
        console.print(f"   Cause: [dim]{m['cause']}[/dim]")
        console.print(f"   Fix:   [green]{m['fix']}[/green]")
        console.print(f"   Seen:  {m.get('count', 1)} time(s)\n")


def show_rag_hits(hits):
    if not hits:
        console.print("[dim]No relevant chunks found.[/dim]")
        return
    console.print(f"\n[bold]{len(hits)} chunks found:[/bold]\n")
    for h in hits:
        console.print(f"[cyan]{h.source}[/cyan]  [dim]score {h.score}[/dim]")
        console.print(Markdown(h.text[:300]))
        console.print()