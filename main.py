"""
main.py — Nexus v2 entry point + REPL.

Startup: banner → health checks (Ollama, model) → RAG store init →
warm-up → REPL.
"""

import sys
import time

import cli
import config
import memory
import model

from agent import Agent
from rag.vector_store import get_store


def _preflight() -> bool:
    if not model.is_running():
        cli.print_status("Ollama is not running. Start it with: ollama serve",
                         style="bold red")
        return False
    if not model.is_model_available():
        cli.print_status(
            f"Model '{config.CHAT_MODEL}' not pulled. Run:\n"
            f"  ollama pull {config.CHAT_MODEL}",
            style="bold yellow",
        )
        return False
    if not model.is_model_available(config.EMBED_MODEL):
        cli.print_status(
            f"Embedding model '{config.EMBED_MODEL}' not pulled. Run:\n"
            f"  ollama pull {config.EMBED_MODEL}",
            style="bold yellow",
        )
    return True


def _run_autonomous(goal: str, agent: Agent, print_fn):
    """Minimal autonomous worker: plan → execute steps (same pattern as v1)."""
    from worker import Worker
    worker = Worker(ask_fn=agent.ask_fn, print_fn=print_fn, agent=agent)
    worker.run(goal)


def main():
    cli.print_banner()

    if not _preflight():
        sys.exit(1)

    # ── RAG store ──────────────────────────────────────────────────────────
    store = None
    try:
        store = get_store()
        stats = store.stats()
        cli.print_status(
            f"RAG: {stats['backend']} — {stats['total_chunks']} chunks",
            style="green" if stats["total_chunks"] else "dim",
        )
    except Exception as e:
        cli.print_status(f"RAG disabled: {e}", style="yellow")

    # ── Warm up the 2B model so the first prompt is fast ───────────────────
    cli.print_status("Warming up model (first load 5–20s)...", style="dim")
    t = model.warmup()
    cli.print_status(f"Model ready in {t:.1f}s — 100% offline.", style="green")

    mistakes = memory.list_mistakes()
    if mistakes:
        cli.print_status(
            f"{len(mistakes)} mistake(s) in memory — injected into prompts.",
            style="dim yellow",
        )

    agent = Agent(ask_fn=cli.ask_confirm, print_fn=cli.print_output,
                  store=store, ui=cli)

    cli.console.print()
    cli.console.print("[dim]Welcome. Ask me anything — code, files, questions.[/dim]")
    cli.console.print("[dim]Type /help for commands · Ctrl+C to quit[/dim]")
    cli.console.print()

    while True:
        user_input = cli.get_input()
        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            cli.console.print("\n[dim]Bye.[/dim]\n")
            break

        elif user_input == "/help":
            cli.print_help()

        elif user_input == "/reset" or user_input == "/clear":
            agent.reset()
            cli.print_status("Conversation cleared.", style="green")

        elif user_input == "/mistakes":
            cli.show_mistakes(memory.list_mistakes())

        elif user_input.startswith("/permissions"):
            flag = user_input[13:].strip()
            if flag == "on":
                agent.auto_approve = True
                cli.print_status("Auto-approve ON (like --dangerously-skip-permissions)",
                                 style="green")
            elif flag == "off":
                agent.auto_approve = False
                cli.print_status("Auto-approve OFF — confirmations on.",
                                 style="green")
            else:
                cli.print_status(
                    f"Auto-approve: {'ON' if agent.auto_approve else 'OFF'} — "
                    "use /permissions on|off", style="cyan")

        elif user_input.startswith("/plan"):
            flag = user_input[5:].strip()
            if flag in ("on", "off"):
                agent.plan_mode = flag == "on"
                cli.print_status(
                    f"Plan mode: {'ON — no changes without approval' if agent.plan_mode else 'OFF'}",
                    style="green")
            else:
                cli.print_status(
                    f"Plan mode: {'ON' if agent.plan_mode else 'OFF'} — use /plan on|off",
                    style="cyan")

        elif user_input == "/undo":
            import tools as toolz
            r = toolz.undo_last()
            cli.print_status(r.output,
                             style="green" if r.ok else "yellow")

        elif user_input.startswith("/model"):
            _model_switch(user_input[6:].strip())

        elif user_input == "/status":
            _status(agent, store)

        elif user_input == "/cost":
            t_in, t_out = agent.total_tokens_in, agent.total_tokens_out
            cli.print_status(
                f"Session tokens: {t_in:,} in · {t_out:,} out "
                f"({t_in + t_out:,} total)", style="cyan")

        elif user_input == "/doctor":
            _doctor()

        elif user_input == "/init":
            _init_project()

        elif user_input == "/memory":
            if store:
                s = store.stats()
                cli.print_status(
                    f"Vector store [{s['backend']}]: {s['total_chunks']} chunks",
                    style="cyan")
            else:
                cli.print_status("No vector store configured.", style="yellow")

        elif user_input == "/dataset":
            import dataset
            s = dataset.stats()
            cli.print_status(
                f"Dataset: {s['total_pairs']} pairs at {s['dataset_path']}",
                style="cyan")

        elif user_input.startswith("/files"):
            path, all_flag = ".", False
            if user_input.strip() != "/files":
                toks = user_input[6:].strip().split()
                if toks and toks[0] == "--all":
                    all_flag = True
                elif toks:
                    path = toks[0]
                    all_flag = "--all" in toks[1:]
            import tools as toolz
            if all_flag:
                r = toolz.list_tree(path)
            else:
                r = toolz.project_files(path)
            if r.ok:
                cli.show_files(r, path)
            else:
                cli.print_status(r.output, style="red")

        elif user_input.startswith("/think"):
            mode = user_input[6:].strip() or "auto"
            if mode not in ("on", "off", "auto"):
                cli.print_status("Usage: /think on|off|auto", style="yellow")
            else:
                agent.set_thinking(mode)
                cli.print_status(f"Thinking: {mode}", style="green")

        elif user_input.startswith("/ingest "):
            path = user_input[8:].strip()
            _ingest(path, store)

        elif user_input == "/ingest":
            cli.print_status("Usage: /ingest <file or folder>", style="yellow")

        elif user_input.startswith("/rag "):
            query = user_input[5:].strip()
            _rag_query(query, store)

        elif user_input.startswith("/run "):
            _run_autonomous(user_input[5:].strip(), agent, cli.print_output)

        elif user_input == "/run":
            cli.print_status("Usage: /run <goal>  e.g. /run build a flask app",
                             style="yellow")

        elif user_input.startswith("/"):
            cli.print_status(f"Unknown command: {user_input}. Type /help.",
                             style="yellow")

        else:
            try:
                t0 = time.time()
                agent.run(user_input)
                cli.print_status(f"Wall time: {time.time() - t0:.1f}s",
                                 style="dim")
            except KeyboardInterrupt:
                cli.console.print("\n[dim]Interrupted.[/dim]")
                agent.reset()
            except Exception as e:
                cli.print_status(f"Agent error: {e}", style="bold red")


def _ingest(path: str, store):
    if not store:
        cli.print_status("No vector store configured — can't ingest.",
                         style="yellow")
        return
    import os
    from rag import indexer

    if os.path.isdir(path):
        r = indexer.ingest_folder(path, store=store)
        status = "green" if r["ok"] else "red"
        msg = (f"Ingested {r['files_processed']} files → "
               f"{r['total_chunks']} chunks.")
        cli.print_status(msg, style=status)
        for e in r["errors"][:5]:
            cli.print_status(f"  error: {e}", style="yellow")
    else:
        r = indexer.ingest_file(path, store=store)
        if r["ok"]:
            cli.print_status(
                f"Ingested {r['source']} → {r['chunks_stored']} chunks.",
                style="green")
        else:
            cli.print_status(f"Failed: {r.get('error')}", style="red")


def _model_switch(arg: str):
    import model
    if not arg:
        cli.print_status(f"Model: {model.get_short_name()} — "
                         f"use /model <name>", style="cyan")
        return
    try:
        ok, msg = model.set_model(arg)
        cli.print_status(msg, style="green" if ok else "yellow")
    except Exception as e:
        cli.print_status(f"Model switch failed: {e}", style="red")


def _status(agent, store):
    import config, model, tokenizer, os
    lines = [
        f"model        : {model.get_short_name()}",
        f"workspace    : {os.getcwd()}",
        f"thinking     : {agent.thinking_mode}",
        f"plan mode    : {'ON' if agent.plan_mode else 'OFF'}",
        f"auto-approve : {'ON' if agent.auto_approve else 'OFF'}",
        f"context      : {config.CONTEXT_BUDGET:,} tokens budget",
        f"threads      : {model.get_threads()}",
        f"project      : {config.PROJECT_DIR}",
    ]
    if store:
        s = store.stats()
        lines.append(f"rag          : {s['backend']} · {s['total_chunks']} chunks")
    try:
        import subprocess
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        if branch:
            lines.append(f"git branch   : {branch}")
    except Exception:
        pass
    cli.console.print()
    for l in lines:
        cli.console.print(f"[bold cyan]{l.split(':')[0]:<12}[/bold cyan]"
                          f"[dim]{l.split(':', 1)[1].strip()}[/dim]")
    cli.console.print()


def _doctor():
    import model
    cli.console.print("\n[bold]Doctor report:[/bold]\n")
    running = model.is_running()
    cli.console.print(f"[green]✓[/green] Ollama running"
                      if running else f"[red]✘[/red] Ollama NOT running")
    if running:
        from model import get_short_name
        cli.console.print(f"[green]✓[/green] Model loaded: {get_short_name()}")
    try:
        from rag.vector_store import get_store
        s = get_store().stats()
        cli.console.print(f"[green]✓[/green] Vector store "
                          f"[{s['backend']}] — {s['total_chunks']} chunks")
    except Exception as e:
        cli.console.print(f"[yellow]✘[/yellow] Vector store: {e}")
    try:
        import tokenizer
        cli.console.print(f"[green]✓[/green] Tokenizer ({tokenizer.usage_report(0, 0)})")
    except Exception as e:
        cli.console.print(f"[yellow]✘[/yellow] Tokenizer: {e}")
    cli.console.print()


def _init_project():
    """Claude Code /init: detect the project and write CLAUDE.md."""
    import os
    from pathlib import Path
    p = Path.cwd()
    lang, build, test = "unknown", "", ""
    if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
        lang, build = "python", "pip install -r requirements.txt"
        test = "python -m pytest"
        if (p / "pyproject.toml").exists():
            build = "pip install -e ."
    elif (p / "package.json").exists():
        lang = "node"
        build = "npm install"
        test = "npm test"
    elif (p / "Cargo.toml").exists():
        lang = "rust"
        build = "cargo build"
        test = "cargo test"
    elif (p / "go.mod").exists():
        lang = "go"
        build = "go build ./..."
        test = "go test ./..."
    elif (p / "Makefile").exists():
        lang = "make"
        build = "make"
        test = "make test"

    target = p / "CLAUDE.md"
    if target.exists():
        cli.print_status("CLAUDE.md already exists — not overwriting.",
                         style="yellow")
        return
    body = (f"# {p.name}\n\n"
            f"Language: {lang}\n"
            + (f"Build: {build}\n" if build else "")
            + (f"Test: {test}\n" if test else "")
            + "Use /run to delegate multi-step builds.\n")
    target.write_text(body)
    cli.print_status(f"Wrote {target} — loaded into every prompt.",
                     style="green")


def _rag_query(query: str, store):
    if not store:
        cli.print_status("No vector store configured.", style="yellow")
        return
    from rag import embeddings
    cli.print_status("Searching knowledge base...", style="dim")
    try:
        qv = embeddings.embed_one(query).tolist()
        hits = store.hybrid_search(query, qv)
        cli.show_rag_hits(hits)
    except Exception as e:
        cli.print_status(f"Search error: {e}", style="red")


if __name__ == "__main__":
    main()
