"""
main.py — Entry point for nexus.

Startup sequence:
  1. Print banner
  2. Load Nexus Coder 1.0
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
import ingestor


def parse_args():
    p = argparse.ArgumentParser(
        description="Nexus — offline AI coding assistant powered by Nexus Coder 1.0"
    )
    p.add_argument(
        "--model", "-m",
        nargs="+",
        default=None,
        help="Model to use: phi3 or Nexus Coder 1.0 (default: Nexus Coder 1.0)",
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
    cli.print_status("Loading model... (first load may take 10–30s)", style="dim")

    chosen = " ".join(args.model) if args.model else model.DEFAULT_MODEL
    ok = model.load_model(model_path=chosen, verbose=args.verbose)
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

        elif user_input.startswith("/ingest "):
            path = user_input[8:].strip()
            if not path:
                cli.print_status("Usage: /ingest <file or folder>", style="yellow")
            else:
                import os
                if os.path.isdir(path):
                    result = ingestor.ingest_folder(path)
                    cli.print_status(f"Ingested {result['files_processed']} files, {result['total_chunks']} chunks stored.", style="green")
                    if result["errors"]:
                        for e in result["errors"]:
                            cli.print_status(f"  error: {e}", style="yellow")
                else:
                    result = ingestor.ingest_file(path)
                    if result["ok"]:
                        cli.print_status(f"Ingested: {result['source']} → {result['chunks_stored']} chunks stored.", style="green")
                    else:
                        cli.print_status(f"Failed: {result.get('error')}", style="red")

        elif user_input == "/memory":
            import learning
            s = learning.stats()
            cli.print_status(f"Vector memory: {s['total_chunks']} chunks stored at {s['db_path']}", style="cyan")

        elif user_input == "/dataset":
            import dataset
            s = dataset.stats()
            cli.print_status(f"Dataset: {s['total_pairs']} training pairs at {s['dataset_path']}", style="cyan")

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
