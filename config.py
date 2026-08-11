"""
config.py — All runtime configuration for Nexus v2.

Settings come from environment variables (.env file) with sane defaults.
Everything is offline-first: only Supabase credentials are optional,
and only used when explicitly provided.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_DIR  = Path(__file__).resolve().parent
CACHE_DIR    = PROJECT_DIR / ".nexus_cache"

# Absolute interpreter path — immune to PATH weirdness in spawned shells.
# Some pyenv installs symlink bin/python to a shim script that breaks when
# invoked from a subprocess, so we resolve + verify the real binary here.
def _resolve_python() -> str:
    import subprocess
    import sysconfig
    base = Path(sys.executable)
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        base,
        Path(sysconfig.get_config_var("BINDIR") or base.parent) / ver,
        base.parent / ver,
    ]
    seen = set()
    for c in candidates:
        if str(c) in seen or not c.exists():
            continue
        seen.add(str(c))
        if not os.access(c, os.X_OK):
            continue
        try:
            r = subprocess.run([str(c), "-c", "pass"],
                               capture_output=True, timeout=10)
            if r.returncode == 0:
                return str(c)
        except Exception:
            continue
    return sys.executable


PYTHON_BIN = _resolve_python()

# ─── Models ────────────────────────────────────────────────────────────────
CHAT_MODEL       = os.getenv("NEXUS_CHAT_MODEL", "qwen2.5-coder:1.5b")
EMBED_MODEL      = os.getenv("NEXUS_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM        = int(os.getenv("NEXUS_EMBED_DIM", "768"))
CHAT_DISPLAY     = os.getenv("NEXUS_CHAT_DISPLAY", "Nexus 2 · Qwen2.5-Coder 1.5B")

# ─── Ollama ────────────────────────────────────────────────────────────────
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT   = int(os.getenv("OLLAMA_TIMEOUT", "180"))
KEEP_ALIVE       = os.getenv("OLLAMA_KEEPALIVE", "24h")

# ─── Context budget (tokens) ───────────────────────────────────────────────
# qwen2.5-coder supports 32k; we budget conservatively so generation
# never truncates and latency stays low on CPU.
NUM_CTX          = int(os.getenv("NEXUS_NUM_CTX", "8192"))
CONTEXT_BUDGET   = int(os.getenv("NEXUS_CONTEXT_BUDGET", str(NUM_CTX // 2)))
MAX_GENERATION   = int(os.getenv("NEXUS_MAX_GENERATION", "1536"))
THINK_BUDGET     = int(os.getenv("NEXUS_THINK_BUDGET", "256"))
TEMPERATURE      = float(os.getenv("NEXUS_TEMPERATURE", "0.2"))
NUM_THREADS      = int(os.getenv("NEXUS_THREADS", str(os.cpu_count() or 4)))

# ─── RAG ───────────────────────────────────────────────────────────────────
RAG_TOP_K        = int(os.getenv("NEXUS_RAG_TOP_K", "8"))
RAG_MIN_SCORE    = float(os.getenv("NEXUS_RAG_MIN_SCORE", "0.35"))
RAG_FUSION_K     = int(os.getenv("NEXUS_RAG_FUSION_K", "60"))   # RRF constant
CHUNK_TOKENS     = int(os.getenv("NEXUS_CHUNK_TOKENS", "400"))
CHUNK_OVERLAP    = int(os.getenv("NEXUS_CHUNK_OVERLAP", "60"))

# ─── Storage ───────────────────────────────────────────────────────────────
# Supabase (pgvector) is used only when URL+KEY are set — otherwise Nexus
# falls back to the zero-config local SQLite store (FTS5 + numpy vectors).
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY     = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY", ""))

MEMORY_DB        = PROJECT_DIR / ".nexus_vectors" / "local.db"
DATASET_DIR      = PROJECT_DIR / ".nexus_dataset"
DATASET_FILE     = DATASET_DIR / "conversations.jsonl"
MISTAKES_FILE    = PROJECT_DIR / "mistakes.json"

# ─── Agent ─────────────────────────────────────────────────────────────────
MAX_TURNS        = int(os.getenv("NEXUS_MAX_TURNS", "10"))
MAX_TOOL_OUTPUT  = int(os.getenv("NEXUS_MAX_TOOL_OUTPUT", "800"))
THINKING_AUTO    = os.getenv("NEXUS_THINKING", "off").lower()  # on | off | auto — off = fast

def store_backend() -> str:
    """'supabase' if configured, else 'local'."""
    return "supabase" if SUPABASE_URL and SUPABASE_KEY else "local"
