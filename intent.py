"""
intent.py — Deterministic tool-intent classification for the 1.5B agent.

A 1.5B model will happily *explain* how to do something instead of doing
it. Claude Code never explains — it acts. So we classify the user's
request in Python and prefill the matching tool call, which the model
then completes. Prefill == the Claude Code "tool forcing" mechanism.

classify(query) → tool name or None (None = plain chat answer)
prefill(tool, query) → the exact partial XML to inject after the user turn
"""

import re

FILE_RE = re.compile(r"[\w][\w.\-]*\.[a-zA-Z0-9]{1,6}\b")

WRITE_VERBS = (
    "write", "create", "make", "generate", "add", "edit", "update",
    "fix", "implement", "refactor", "change", "modify", "rewrite",
    "build", "develop",
)
CHANGE_VERBS = (
    "change", "update", "modify", "refactor", "improve", "rewrite",
    "edit ", "fix", "replace", "remove ", "delete",
)
FILE_HINTS = (
    r"\.[a-zA-Z0-9]{1,6}\b",        # filename.ext
    r"\bfile\b", r"\bthe files\b", r"\bthis file\b", r"\bscript\b",
)
RUN_MARKERS = (
    "run ", "execute", "start ", "launch", "install", "compile",
    " npm ", " pip ", " pip2", "cargo ", " go ", " git ", "deploy",
)
TEST_MARKERS = ("test", "pytest", "unittest")
LIST_MARKERS = (
    "list", "ls", "show the files", "show files", "files in",
    "what's in", "what is in", "directory", "folder", "tree",
    "contents", " ls", "ls -",
)
READ_MARKERS = (
    "read ", "show me", "print the file", "display", "cat ",
    "open the file", "show the content", "look at", "view ",
)
SEARCH_MARKERS = (
    "search", "find ", "grep", "locate", "where is", "where's",
    "which file", "usage of",
)


def extract_filename(query: str) -> str | None:
    """Pull the first filename-like token from a query.

    Prefer full paths (absolute or ./ relative) so writes/reads land in
    the right place even when the cwd differs from the user's intent.
    """
    for tok in query.split():
        if tok.startswith(("/", "./", "../")) and FILE_RE.search(tok):
            return tok
    m = FILE_RE.search(query)
    return m.group(0) if m else None


def _has_write_hint(query: str) -> bool:
    q = query.lower()
    if extract_filename(query):
        return True
    return any(re.search(h, q) for h in FILE_HINTS)


def _test_command(query: str) -> str | None:
    import config
    py = config.PYTHON_BIN
    q = query.lower()
    if re.search(r"npm|yarn|pnpm", q):
        return "npm test"
    if re.search(r"cargo", q):
        return "cargo test"
    if re.search(r"^go |\bgo test", q):
        return "go test ./..."
    if "pytest" in q:
        return f"{py} -m pytest"
    if re.search(r"\btests?\b", q):
        return f"{py} -m pytest"
    if "install" in q and ("requirements" in q or "pip" in q):
        return f"{py} -m pip install -r requirements.txt --break-system-packages"
    return None


def classify(query: str) -> str | None:
    """Return the tool that should run for this query, else None (chat)."""
    q = query.lower().strip()

    # Change intent → Read the file first, then the model edits it
    # (Claude Code: read before edit — and never blind-write a new file).
    if any(v in q for v in CHANGE_VERBS) and extract_filename(query):
        return "read_file"

    # Write intent — needs a file-ish target
    if any(v in q for v in WRITE_VERBS) and _has_write_hint(query):
        return "write_file"

    # Run / test intent
    if any(t in q for t in TEST_MARKERS) and any(
        r in q for r in ("run", "execute", "test", "pytest", "unittest")
    ):
        return "run_shell"
    if any(m in q for m in RUN_MARKERS):
        return "run_shell"

    # List intent
    if any(m in q for m in LIST_MARKERS):
        return "list_tree"

    # Read intent (needs a target file)
    if any(m in q for m in READ_MARKERS):
        if extract_filename(query) or r"\bfile" in q:
            return "read_file"

    # Search intent
    if any(m in q for m in SEARCH_MARKERS):
        return "search"

    return None


def prefill(tool: str, query: str) -> str:
    """
    Partial XML that forces the model to complete this tool call.

    Tools with fixed/known args return a CLOSED call (the agent can then
    execute directly without waiting for model output — pure Claude Code
    behavior). Tools needing model content (file content, search query)
    return an OPEN call that the model completes.
    """
    if tool == "write_file":
        fname = extract_filename(query)
        if fname:
            return f"<write_file>\n  <path>{fname}</path>\n  <content>\n"
        return "<write_file>\n  <path>"
    if tool == "read_file":
        fname = extract_filename(query)
        if fname:
            return f"<read_file>\n  <path>{fname}</path>\n</read_file>"
        return "<read_file>\n  <path>"
    if tool == "list_tree":
        return "<list_tree>\n  <root>.</root>\n</list_tree>"
    if tool == "search":
        return "<search>\n  <query>"
    if tool == "run_shell":
        cmd = _test_command(query)
        if cmd:
            return (f"<run_shell>\n  <command>{cmd}</command>\n"
                    f"</run_shell>")
        return "<run_shell>\n  <command>"
    return ""