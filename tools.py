"""
tools.py — Agent tools. Local only, safety-gated.

Every tool returns ToolResult(ok, output, mistake_hint).
The agent executes these based on model decisions, always behind a
user confirmation gate for writes and shell commands.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config


@dataclass
class ToolResult:
    ok: bool
    output: str
    mistake_hint: Optional[str] = None


def _safe_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def _backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


# ─── read_file ──────────────────────────────────────────────────────────────

def read_file(path: str, max_lines: int = 300) -> ToolResult:
    try:
        p = _safe_path(path)
        if not p.exists():
            return ToolResult(False, f"File not found: {path}",
                              mistake_hint=f"read_file on missing file: {path}")
        if not p.is_file():
            return ToolResult(False, f"Not a file: {path}")
        lines = p.read_text(errors="replace").splitlines()
        shown = lines[:max_lines]
        suffix = f"\n[… {len(lines) - max_lines} more lines…]" if len(lines) > max_lines else ""
        return ToolResult(True, f"```\n{chr(10).join(shown)}{suffix}\n```")
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, f"Error reading {path}: {e}")


# ─── write_file (with .bak backup) ──────────────────────────────────────────

def write_file(path: str, content: str, backup: bool = True) -> ToolResult:
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bak = _backup(p) if backup else None
        p.write_text(content)
        msg = f"Written: {path}"
        if bak:
            msg += f" (backup: {bak.name})"
        return ToolResult(True, msg)
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, f"Error writing {path}: {e}")


# ─── list_tree ──────────────────────────────────────────────────────────────

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             ".env", "dist", "build", ".mypy_cache", ".pytest_cache", ".tox"}
SKIP_EXTS = {".pyc", ".pyo", ".class", ".o", ".so", ".dylib"}


def list_tree(root: str = ".", max_files: int = 120) -> ToolResult:
    try:
        rp = _safe_path(root)
        if not rp.exists():
            return ToolResult(False, f"Directory not found: {root}")
        lines, count = [str(rp)], 0
        for dirpath, dirnames, filenames in os.walk(rp):
            dirnames[:] = [d for d in sorted(dirnames)
                           if d not in SKIP_DIRS and not d.startswith(".")]
            depth = len(Path(dirpath).relative_to(rp).parts)
            indent = "  " * depth
            for f in sorted(filenames):
                if Path(f).suffix in SKIP_EXTS:
                    continue
                lines.append(f"{indent}{f}")
                count += 1
                if count >= max_files:
                    lines.append(f"{indent}… truncated at {max_files} files")
                    return ToolResult(True, "\n".join(lines))
            for d in dirnames:
                lines.append(f"{indent}{d}/")
        return ToolResult(True, "\n".join(lines))
    except Exception as e:
        return ToolResult(False, f"Error listing {root}: {e}")


# ─── run_shell (blocklist + timeout) ────────────────────────────────────────

BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bdd\b.*of=/dev/",
    r":\(\)\{.*\};:",
    r"\bmkfs\b",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r">\s*/dev/sd", r"\bformat\b.*\bdrive\b",
]


def run_shell(command: str, cwd: str = None, timeout: int = 30) -> ToolResult:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return ToolResult(
                False,
                f"Blocked: matches dangerous pattern ({pattern})",
                mistake_hint=f"tried dangerous command: {command}",
            )
    try:
        work = _safe_path(cwd) if cwd else Path.cwd()
        result = subprocess.run(
            command, shell=True, cwd=work,
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout
        if result.stderr.strip():
            out += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            return ToolResult(
                False,
                f"Exit {result.returncode}:\n{out or '(no output)'}",
                mistake_hint=f"command failed (exit {result.returncode}): {command}",
            )
        return ToolResult(True, out or "(command succeeded, no output)")
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Timed out after {timeout}s: {command}",
                          mistake_hint=f"command timed out: {command}")
    except Exception as e:
        return ToolResult(False, f"Shell error: {e}")


# ─── search (grep) ──────────────────────────────────────────────────────────

def search(query: str, root: str = ".", file_pattern: str = "*") -> ToolResult:
    try:
        rp = _safe_path(root)
        results, count = [], 0
        for path in sorted(rp.rglob(file_pattern)):
            if not path.is_file():
                continue
            if any(p in SKIP_DIRS for p in path.parts) or path.suffix in SKIP_EXTS:
                continue
            try:
                for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if query.lower() in line.lower():
                        results.append(f"{path.relative_to(rp)}:{i}: {line.strip()}")
                        count += 1
                        if count >= 50:
                            results.append("… truncated at 50 matches")
                            return ToolResult(True, "\n".join(results))
            except Exception:
                continue
        if not results:
            return ToolResult(True, f"No matches for '{query}'")
        return ToolResult(True, "\n".join(results))
    except Exception as e:
        return ToolResult(False, f"Search error: {e}")


# ─── Edit tools (Claude Code-style, with backup + diff preview) ─────────────

import difflib

BACKUP_LOG: list[tuple[str, str]] = []   # (path, backup_path) — last first


def _unified_diff(before: str, after: str, path: str) -> str:
    b = before.splitlines()
    a = after.splitlines()
    return "\n".join(difflib.unified_diff(
        b, a, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
    ))


def _log_backup(path: Path, backup: Optional[Path]):
    if backup:
        BACKUP_LOG.append((str(path), str(backup)))


def undo_last() -> ToolResult:
    """Restore the most recent backup (Claude Code /undo)."""
    if not BACKUP_LOG:
        return undo_scan()
    path, backup = BACKUP_LOG.pop()
    try:
        Path(path).write_text(Path(backup).read_text())
        return ToolResult(True, f"Restored {path} from {backup}")
    except Exception as e:
        return ToolResult(False, f"Undo failed for {path}: {e}")


def undo_scan() -> ToolResult:
    """Fallback: find the newest *.bak on disk and restore it."""
    import glob
    import time
    try:
        baks = [p for p in Path(".").glob("**/*.bak")
                if p.is_file() and not p.name.startswith(".")]
        if not baks:
            return ToolResult(False, "No backups found on disk.")
        newest = max(baks, key=lambda p: p.stat().st_mtime)
        target = newest.with_suffix("")
        if not target.exists():
            return ToolResult(False,
                              f"Backup {newest} has no target file — skipped.")
        target.write_text(newest.read_text())
        return ToolResult(True, f"Restored {target} from {newest} "
                                f"({time.strftime('%H:%M', time.localtime(newest.stat().st_mtime))})")
    except Exception as e:
        return ToolResult(False, f"Undo scan failed: {e}")


def edit_file(path: str, old_string: str, new_string: str) -> ToolResult:
    """Replace the first exact occurrence of old_string in a file.

    Shows a unified diff in the result, keeps a .bak backup.
    Very small — like Claude Code's Edit tool: the model must Read
    before Edit so old_string matches EXACTLY.
    """
    try:
        p = _safe_path(path)
        if not p.exists():
            return ToolResult(False, f"File not found: {path}",
                              mistake_hint=f"edit_file on missing file: {path}")
        before = p.read_text()
        if old_string not in before:
            return ToolResult(
                False,
                f"old_string NOT found in {path}.\n"
                f"Given: {old_string[:80]!r}\nRead the file first, then retry.",
                mistake_hint=f"edit_file: old_string missing in {path}",
            )
        if old_string == new_string:
            return ToolResult(False, "old_string == new_string — nothing to change.")
        after = before.replace(old_string, new_string, 1)
        bak = _backup(p)
        p.write_text(after)
        _log_backup(p, bak)
        return ToolResult(
            True,
            f"Edited: {path} (backup: {bak.name})\n{_unified_diff(before, after, path)}",
        )
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, f"Edit failed on {path}: {e}")


def multi_edit(path: str, edits: list[dict]) -> ToolResult:
    """Apply several old→new replacements in one file (Claude Code MultiEdit).

    edits: [{"old_string": ..., "new_string": ...}, ...]
    Applies atomically: if ANY old_string is missing, nothing is written.
    """
    try:
        p = _safe_path(path)
        if not p.exists():
            return ToolResult(False, f"File not found: {path}")
        before = p.read_text()
        working = before
        for i, e in enumerate(edits, 1):
            old, new = e.get("old_string", ""), e.get("new_string", "")
            if old not in working:
                return ToolResult(
                    False,
                    f"MultiEdit block {i}: old_string not found in {path}\n"
                    f"Given: {old[:80]!r}\nNothing was changed.",
                    mistake_hint=f"multi_edit: block {i} missing in {path}",
                )
            working = working.replace(old, new, 1)
        if working == before:
            return ToolResult(False, "MultiEdit made no changes.")
        bak = _backup(p)
        p.write_text(working)
        _log_backup(p, bak)
        return ToolResult(
            True,
            f"MultiEdit: {path} — {len(edits)} change(s) (backup: {bak.name})\n"
            + _unified_diff(before, working, path),
        )
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, f"MultiEdit failed on {path}: {e}")


# ─── project_files (Claude Code-style listing) ─────────────────────────────

def project_files(root: str = ".", max_files: int = 300) -> ToolResult:
    """
    Flat, junk-free file listing like Claude Code's file view.

    Inside a git repo: uses `git ls-files` (respects .gitignore, shows
    tracked + untracked non-ignored files — no build artifacts, no
    virtualenvs, no hidden caches). Outside a repo: walks the tree,
    skipping hidden dirs and known junk.
    """
    try:
        rp = _safe_path(root)
        if not rp.exists():
            return ToolResult(False, f"Path not found: {root}")
        if rp.is_file():
            return ToolResult(True, f"1 file\n{rp}")

        # ── git-aware ────────────────────────────────────────────────────
        try:
            r = subprocess.run(
                ["git", "-C", str(rp), "ls-files",
                 "--cached", "--others", "--exclude-standard"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                rows = r.stdout.splitlines()
                return _format_rows(rows, rp, max_files, tracked=True)
        except Exception:
            pass

        # ── plain walk fallback ──────────────────────────────────────────
        rows: list[str] = []
        for dirpath, dirnames, filenames in os.walk(rp):
            dirnames[:] = [d for d in sorted(dirnames)
                           if d not in SKIP_DIRS and not d.startswith(".")]
            for f in sorted(filenames):
                if Path(f).suffix in SKIP_EXTS:
                    continue
                rel = Path(dirpath).relative_to(rp)
                rows.append(str(rel / f) if str(rel) != "." else f)
        return _format_rows(sorted(rows), rp, max_files, tracked=False)

    except Exception as e:
        return ToolResult(False, f"Error listing {root}: {e}")


def _format_rows(rows: list[str], root: Path, max_files: int,
                 tracked: bool) -> ToolResult:
    # Safety net on top of gitignore: never list caches or nexus internals
    rows = [r for r in rows
            if not r.startswith((".nexus_", "mistakes.json", ".env"))
            and "__pycache__" not in r
            and not r.endswith((".pyc", ".pyo"))]
    total = len(rows)
    shown = rows[:max_files]
    lines = [f"{total} file(s)" + (" · git-tracked" if tracked else "")]
    lines += [f"  {r}" for r in shown]
    if total > max_files:
        lines.append(f"  … {total - max_files} more")
    return ToolResult(True, "\n".join(lines))


# ─── Registry ───────────────────────────────────────────────────────────────

TOOLS = {
    "read_file":  read_file,
    "write_file": write_file,
    "edit_file":  edit_file,
    "multi_edit": multi_edit,
    "list_tree":  list_tree,
    "run_shell":  run_shell,
    "search":     search,
}

TOOL_DOC = """
Tools (XML tags, one per turn):

<read_file>
  <path>relative/or/absolute/path</path>
</read_file>
  Use when the user asks to read, show, or view a file.

<write_file>
  <path>filename.py</path>
  <content>
complete file content here — never placeholders
  </content>
</write_file>
  Use when the user asks to write, create, or rewrite an entire file.

<edit_file>
  <path>filename.py</path>
  <old_string>exact text currently in the file</old_string>
  <new_string>replacement text</new_string>
</edit_file>
  Use for a precise change to one spot of an existing file.
  old_string must match the file EXACTLY (Read the file first).

<multi_edit>
  <path>filename.py</path>
  <edit>
    <old_string>exact text in the file</old_string>
    <new_string>replacement text</new_string>
  </edit>
  <edit>
    <old_string>exact text in the file</old_string>
    <new_string>replacement text</new_string>
  </edit>
</multi_edit>
  Use to apply several changes to ONE file in a single call.

<list_tree>
  <root>.</root>
</list_tree>
  Use when the user asks to list, show, or inspect files/directories.

<run_shell>
  <command>exact shell command</command>
</run_shell>
  Use when the user asks to run, execute, install, or test something.

<search>
  <query>text to find</query>
  <root>.</root>
  <file_pattern>*.py</file_pattern>
</search>
  Use when the user asks to find, grep, or search the code.

When no tool applies, answer directly in plain text.
"""
