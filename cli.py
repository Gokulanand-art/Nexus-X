"""
cli.py — Terminal UI using Rich.

Handles all input/output. Keeps agent.py and model.py clean.
"""

import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.text import Text
from rich import print as rprint

console = Console()

BANNER = """
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
"""

COMMANDS = {
    "/exit":    "Exit nexus",
    "/reset":   "Clear conversation history",
    "/mistakes":"Show recorded mistakes",
    "/clear":   "Clear mistake memory",
    "/files":   "List files in current directory",
    "/run":     "Autonomous mode — give a goal, agent works alone",
    "/ingest":  "Ingest a file or folder into memory: /ingest <path>",
    "/memory":  "Show vector memory stats",
    "/dataset": "Show training dataset stats",
    "/help":    "Show this help",
}


def print_banner():
    console.print(Text(BANNER, style="bold purple"))
    console.print(
        Panel(
            "[dim]Offline AI coding assistant · Phi-3 Mini · 100% local[/dim]\n"
            "[dim]Type [bold white]/help[/bold white] for commands[/dim]",
            border_style="purple",
            padding=(0, 2),
        )
    )
    console.print()


def print_help():
    console.print("\n[bold]Commands:[/bold]")
    for cmd, desc in COMMANDS.items():
        console.print(f"  [cyan]{cmd:<14}[/cyan] {desc}")
    console.print()


def print_output(text: str, dim: bool = False, error: bool = False):
    """Called by agent to print text (streaming + status messages)."""
    if error:
        console.print(text, style="red", end="")
    elif dim:
        console.print(text, style="dim", end="")
    else:
        # Print raw — agent streams tokens one by one
        print(text, end="", flush=True)


def ask_confirm(prompt: str) -> bool:
    """Called by agent for confirm gate and mistake guard."""
    try:
        console.print()
        answer = console.input(f"[yellow]{prompt}[/yellow]").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def get_input() -> str:
    """Get user input with a styled prompt."""
    try:
        console.print()
        return console.input("[bold purple]>[/bold purple] ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


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


def print_status(msg: str, style: str = "dim"):
    console.print(f"[{style}]{msg}[/{style}]")
