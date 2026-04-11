"""
main.py — Entry point for nexus.

Startup sequence:
  1. Print banner
  2. Load Phi-3 mini (llama-cpp-python)
  3. Start REPL loop
  4. Route /commands or user messages to agent
"""

import sys
import argparse
from pathlib import Path

import cli
import memory
import model
import tools
from agent import Agent
from worker import Worker


def parse_args():
    p = argparse.ArgumentParser(
        description="Nexus — offline AI coding assistant powered by Phi-3 mini"
    )
    p.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to .gguf model file (default: auto-detect in ./models/)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show llama.cpp loading logs",
    )
    p.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts for writes/shell (dangerous — dev use only)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    cli.print_banner()

    # ── Load model ────────────────────────────────────────────────────────────
    cli.print_status("Loading Phi-3 mini... (first load may take 10–30s)", style="dim")

    ok = model.load_model(model_path=args.model, verbose=args.verbose)
    if not ok:
        cli.print_status("Model failed to load. Exiting.", style="bold red")
        sys.exit(1)

    cli.print_status("Model ready. Running on CPU — 100% offline.", style="green")

    # ── Mistake memory: show count on startup ─────────────────────────────────
    mistakes = memory.list_mistakes()
    if mistakes:
        cli.print_status(
            f"Loaded {len(mistakes)} mistake(s) from memory — injected into every prompt.",
            style="dim yellow",
        )

    # ── Build agent + worker ──────────────────────────────────────────────────
    def ask_fn(prompt: str) -> bool:
        if args.no_confirm:
            return True
        return cli.ask_confirm(prompt)

    agent  = Agent(ask_fn=ask_fn, print_fn=cli.print_output)
    worker = Worker(ask_fn=ask_fn, print_fn=cli.print_output)

    cli.console.print()
    cli.console.print("[dim]Ready. Ask me anything about your code.[/dim]")
    cli.console.print()

    # ── REPL loop ─────────────────────────────────────────────────────────────
    while True:
        user_input = cli.get_input()

        if not user_input:
            continue

        # ── Built-in commands ────────────────────────────────────────────────
        if user_input == "/exit" or user_input == "/quit":
            cli.console.print("\n[dim]Bye.[/dim]\n")
            break

        elif user_input == "/help":
            cli.print_help()

        elif user_input == "/reset":
            agent.reset()
            cli.print_status("Conversation cleared.", style="green")

        elif user_input == "/mistakes":
            cli.show_mistakes(memory.list_mistakes())

        elif user_input == "/clear":
            confirmed = cli.ask_confirm("Clear all recorded mistakes? [y/N] ")
            if confirmed:
                memory.clear_mistakes()
                cli.print_status("Mistake memory cleared.", style="green")

        elif user_input == "/files":
            result = tools.list_tree(".")
            cli.console.print(result.output)

        elif user_input.startswith("/run "):
            goal = user_input[5:].strip()
            if not goal:
                cli.print_status("Usage: /run <your goal>  e.g. /run build a flask app", style="yellow")
            else:
                try:
                    worker.run(goal)
                except KeyboardInterrupt:
                    cli.console.print("\n[dim]Worker interrupted.[/dim]")

        elif user_input == "/run":
            cli.print_status("Usage: /run <your goal>  e.g. /run build a flask app", style="yellow")

        elif user_input.startswith("/"):
            cli.print_status(f"Unknown command: {user_input}. Type /help.", style="yellow")

        # ── Agent loop ───────────────────────────────────────────────────────
        else:
            try:
                agent.run(user_input)
            except KeyboardInterrupt:
                cli.console.print("\n[dim]Interrupted.[/dim]")
                agent.reset()


if __name__ == "__main__":
    main()
