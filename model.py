"""
model.py — Ollama brain connector for Nexus.

How it works:
  - Ollama runs as a background service on your machine (localhost:11434)
  - We talk to it over a simple HTTP API — no C++ compilation, no .gguf files
  - Phi-3 Mini runs inside Ollama — fully offline after first pull
  - Streams tokens back one by one, exactly like Claude Code does

Setup (one time only):
  1. Install Ollama:   https://ollama.com/download
  2. Pull the model:   ollama pull phi3
  3. Run nexus:        python main.py

That's it. Nothing else to install.
"""

import json
import urllib.request
import urllib.error
from typing import Iterator

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_HOST   = "http://localhost:11434"
DEFAULT_MODEL = "phi3"       # Microsoft Phi-3 Mini — best small coding model
TIMEOUT       = 45          # seconds before giving up on a slow response
TEMPERATURE   = 0.2          # low = focused, deterministic code output
MAX_TOKENS    = 1024

# ─── Connection check ─────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Check if Ollama daemon is up. Fast — just hits the root endpoint."""
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def is_model_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if the model has been pulled and is ready to use."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return model.split(":")[0] in models
    except Exception:
        return False


# ─── Startup check ────────────────────────────────────────────────────────────

def load_model(model_path: str = None, verbose: bool = False) -> bool:
    """
    Verify Ollama is running and the model is available.
    model_path is accepted for API compatibility but ignored —
    Ollama manages models itself via ollama pull.
    Returns True if ready, False if setup needed.
    """
    model = model_path or DEFAULT_MODEL

    if not is_ollama_running():
        print()
        print("[nexus] Ollama is not running.")
        print()
        print("  Fix:")
        print("  1. Install Ollama:  https://ollama.com/download")
        print("  2. Start it:        ollama serve")
        print("     (Mac/Windows: starts automatically after install)")
        print()
        return False

    if not is_model_available(model):
        print()
        print(f"[nexus] Model '{model}' not found in Ollama.")
        print()
        print("  Fix — run this once in your terminal:")
        print(f"  ollama pull {model}")
        print()
        print("  (~2.3GB download — then works offline forever)")
        print()
        return False

    if verbose:
        print(f"[nexus] Ollama ready — using model: {model}")

    return True


def is_loaded() -> bool:
    """True if Ollama is up and default model is available."""
    return is_ollama_running() and is_model_available(DEFAULT_MODEL)


# ─── Streaming response ───────────────────────────────────────────────────────

def stream_response(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    stop: list[str] = None,
    model: str = DEFAULT_MODEL,
) -> Iterator[str]:
    """
    Stream tokens from Ollama one by one.
    messages: [{"role": "system"|"user"|"assistant", "content": str}]

    Yields text chunks as they arrive — agent.py needs zero changes.
    Uses only Python stdlib — no extra pip installs needed.
    """
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop":        stop or [],
        },
    }).encode()

    req = urllib.request.Request(
        url     = f"{OLLAMA_HOST}/api/chat",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            # Ollama sends one JSON object per line while streaming
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract the text token from this chunk
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text

                # Ollama signals end of stream with done: true
                if chunk.get("done"):
                    break

    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
            f"Make sure it is running: ollama serve\n"
            f"Error: {e}"
        )
    except TimeoutError:
        raise RuntimeError(
            f"Ollama timed out after {TIMEOUT}s.\n"
            f"The model may still be loading — try again in a moment."
        )


# ─── Non-streaming convenience ────────────────────────────────────────────────

def complete(
    messages: list[dict],
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    model: str = DEFAULT_MODEL,
) -> str:
    """Returns the full response as one string. For short internal tasks."""
    return "".join(stream_response(messages, max_tokens, temperature, model=model))
