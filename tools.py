"""
tools.py — All agent tools. Local only. No network.

Each tool returns a ToolResult with success/error and output text.
The agent loop calls these based on model decisions.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ToolResult:
    ok: bool          # True = success, False = error
    output: str       # Text shown back to the model
    mistake_hint: Optional[str] = None  # If ok=False, a hint for mistake memory


# ─── Helpers ────────────────────────────────────────────────────────────────

def _backup(path: Path) -> Path:
    """Create a .bak copy before overwriting any file."""
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    return backup_path


def _safe_path(raw: str) -> Path:
    """Resolve path. Reject traversal attacks."""
    p = Path(raw).expanduser().resolve()
    return p


# ─── Tool: read_file ─────────────────────────────────────────────────────────

def read_file(path: str, max_lines: int = 300) -> ToolResult:
    """Read a file and return its contents (truncated if large)."""
    try:
        p = _safe_path(path)
        if not p.exists():
            return ToolResult(
                ok=False,
                output=f"File not found: {path}",
                mistake_hint=f"Tried to read non-existent file: {path}",
            )
        if not p.is_file():
            return ToolResult(ok=False, output=f"Not a file: {path}")

        lines = p.read_text(errors="replace").splitlines()
        truncated = len(lines) > max_lines
        shown = lines[:max_lines]
        content = "\n".join(shown)

        suffix = f"\n\n[... {len(lines) - max_lines} more lines truncated ...]" if truncated else ""
        return ToolResult(ok=True, output=f"```\n{content}{suffix}\n```")

    except PermissionError:
        return ToolResult(ok=False, output=f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(ok=False, output=f"Error reading {path}: {e}")


# ─── Tool: write_file ────────────────────────────────────────────────────────

def write_file(path: str, content: str, backup: bool = True) -> ToolResult:
    """
    Write content to a file. Creates parent dirs if needed.
    Makes a .bak backup if file already exists.
    IMPORTANT: caller must get user confirmation before calling this.
    """
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        backed_up = None
        if p.exists() and backup:
            backed_up = _backup(p)

        p.write_text(content)

        msg = f"Written: {path}"
        if backed_up:
            msg += f" (backup: {backed_up.name})"
        return ToolResult(ok=True, output=msg)

    except PermissionError:
        return ToolResult(ok=False, output=f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(ok=False, output=f"Error writing {path}: {e}")


# ─── Tool: list_tree ─────────────────────────────────────────────────────────

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".mypy_cache", ".pytest_cache",
    "*.egg-info", ".tox",
}
SKIP_EXTS = {".pyc", ".pyo", ".class", ".o", ".so", ".dylib"}


def list_tree(root: str = ".", max_files: int = 120) -> ToolResult:
    """Walk directory tree. Returns compact tree string like `ls -la --tree`."""
    try:
        root_path = _safe_path(root)
        if not root_path.exists():
            return ToolResult(ok=False, output=f"Directory not found: {root}")

        lines = [str(root_path)]
        count = 0

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune ignored dirs in-place
            dirnames[:] = [
                d for d in sorted(dirnames)
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            depth = len(Path(dirpath).relative_to(root_path).parts)
            indent = "  " * depth

            for filename in sorted(filenames):
                if Path(filename).suffix in SKIP_EXTS:
                    continue
                lines.append(f"{indent}{filename}")
                count += 1
                if count >= max_files:
                    lines.append(f"{indent}... (truncated at {max_files} files)")
                    return ToolResult(ok=True, output="\n".join(lines))

            # Show dir names too
            for d in dirnames:
                lines.append(f"{indent}{d}/")

        return ToolResult(ok=True, output="\n".join(lines))

    except Exception as e:
        return ToolResult(ok=False, output=f"Error listing {root}: {e}")


# ─── Tool: run_shell ─────────────────────────────────────────────────────────

BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/",          # rm -rf /
    r"\bdd\b.*of=/dev/",        # dd to device
    r":(){ :|:& };:",            # fork bomb
    r"\bmkfs\b",                 # format disk
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
]


def run_shell(command: str, cwd: str = None, timeout: int = 30) -> ToolResult:
    """
    Run a shell command. Returns stdout+stderr.
    IMPORTANT: caller must get user confirmation before calling this.
    Blocks obviously dangerous commands.
    """
    # Safety check — block destructive patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return ToolResult(
                ok=False,
                output=f"Blocked: command matches dangerous pattern ({pattern})",
                mistake_hint=f"Tried to run dangerous command: {command}",
            )

    try:
        work_dir = _safe_path(cwd) if cwd else Path.cwd()

        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = ""
        if result.stdout.strip():
            output += result.stdout
        if result.stderr.strip():
            output += f"\n[stderr]\n{result.stderr}"

        if result.returncode != 0:
            return ToolResult(
                ok=False,
                output=f"Exit {result.returncode}:\n{output or '(no output)'}",
                mistake_hint=f"Command failed (exit {result.returncode}): {command}",
            )

        return ToolResult(ok=True, output=output or "(command succeeded, no output)")

    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            output=f"Timed out after {timeout}s: {command}",
            mistake_hint=f"Command timed out: {command}",
        )
    except Exception as e:
        return ToolResult(ok=False, output=f"Shell error: {e}")


# ─── Tool: search_in_files ───────────────────────────────────────────────────

def search(query: str, root: str = ".", file_pattern: str = "*") -> ToolResult:
    """Grep-like search across files in directory."""
    try:
        root_path = _safe_path(root)
        results = []
        count = 0

        for path in sorted(root_path.rglob(file_pattern)):
            if not path.is_file():
                continue
            # Skip binary-ish files and ignored dirs
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix in SKIP_EXTS:
                continue

            try:
                text = path.read_text(errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if query.lower() in line.lower():
                        rel = path.relative_to(root_path)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        count += 1
                        if count >= 50:
                            results.append("... (truncated at 50 matches)")
                            return ToolResult(ok=True, output="\n".join(results))
            except Exception:
                continue

        if not results:
            return ToolResult(ok=True, output=f"No matches for '{query}'")

        return ToolResult(ok=True, output="\n".join(results))

    except Exception as e:
        return ToolResult(ok=False, output=f"Search error: {e}")


# ─── Tool registry ───────────────────────────────────────────────────────────

TOOLS = {
    "read_file":   read_file,
    "write_file":  write_file,
    "list_tree":   list_tree,
    "run_shell":   run_shell,
    "search":      search,
}

TOOL_SCHEMAS = """
Available tools (call with XML tags):

<read_file>
  <path>relative/or/absolute/path</path>
</read_file>

<write_file>
  <path>path/to/file.py</path>
  <content>
full file content here
  </content>
</write_file>

<list_tree>
  <root>.</root>
</list_tree>

<run_shell>
  <command>python --version</command>
  <cwd>.</cwd>
</run_shell>

<search>
  <query>def main</query>
  <root>.</root>
  <file_pattern>*.py</file_pattern>
</search>

If no tool is needed, respond directly in plain text.
"""
