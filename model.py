"""
model.py — Ollama brain connector for Nexus X.

Models:
  phi3          Microsoft Phi-3 Mini    — fast, 2.5GB RAM
  deepseek-coder:6.7b DeepSeek Coder   — best code, 5.5GB RAM
  mistral       Mistral 7B              — best chat, 4.5GB RAM
  tinyllama     TinyLlama 1.1B          — lightest, 800MB RAM
"""

import json
import urllib.request
import urllib.error
from typing import Iterator

OLLAMA_HOST   = "http://localhost:11434"
TIMEOUT       = 45
TEMPERATURE   = 0.2
MAX_TOKENS    = 1024

# Stop tokens per model — prevents repetition loops like ²³¹²³¹²³¹
MODEL_STOP_TOKENS = {
    "phi3":                 ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"],
    "deepseek-coder:6.7b":  ["<|EOT|>", "User:", "Assistant:"],
    "mistral":              ["[INST]", "[/INST]", "</s>"],
    "tinyllama":            ["</s>", "<|user|>", "<|assistant|>"],
}
DEFAULT_STOP = ["<|end|>", "</s>", "User:", "Human:"]

MODEL_ALIASES = {
    "phi3":      "phi3",
    "deepseek":  "deepseek-coder:6.7b",
    "deepseek-coder": "deepseek-coder:6.7b",
    "mistral":   "mistral",
    "tinyllama": "tinyllama",
}

_active_model = "phi3"

# ─── Compatibility for CLI ─────────────────────────────────────────────
AVAILABLE_MODELS = MODEL_ALIASES.copy()
DEFAULT_MODEL = "phi3"


def set_model(alias: str) -> bool:
    global _active_model
    _active_model = MODEL_ALIASES.get(alias.lower(), alias)
    return True


def get_model() -> str:
    return _active_model


def _get_stop_tokens(model_name: str) -> list[str]:
    base = model_name.split(":")[0]
    for key, tokens in MODEL_STOP_TOKENS.items():
        if key.split(":")[0] == base:
            return tokens
    return DEFAULT_STOP


def is_ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def is_model_available(model_name: str = None) -> bool:
    name = model_name or _active_model
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            data   = json.loads(resp.read())
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return name.split(":")[0] in models
    except Exception:
        return False


def list_available_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def load_model(model_path: str = None, verbose: bool = False) -> bool:
    global _active_model
    if model_path:
        _active_model = MODEL_ALIASES.get(model_path.lower(), model_path)

    if not is_ollama_running():
        print()
        print("[nexus] Ollama is not running.")
        print("  1. Install: https://ollama.com/download")
        print("  2. Start:   ollama serve")
        print()
        return False

    if not is_model_available(_active_model):
        print()
        print(f"[nexus] Model '{_active_model}' not pulled.")
        print(f"  Run: ollama pull {_active_model}")
        print()
        return False

    if verbose:
        print(f"[nexus] Ready — model: {_active_model}")
    return True


def is_loaded() -> bool:
    return is_ollama_running() and is_model_available(_active_model)


def stream_response(
    messages:    list[dict],
    max_tokens:  int   = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    stop:        list  = None,
    model:       str   = None,
) -> Iterator[str]:
    """Stream tokens. Automatically uses correct stop tokens to prevent repetition."""
    active      = model or _active_model
    stop_tokens = stop or _get_stop_tokens(active)

    payload = json.dumps({
        "model":    active,
        "messages": messages,
        "stream":   True,
        "options": {
            "temperature":   temperature,
            "num_predict":   max_tokens,
            "stop":          stop_tokens,
            "repeat_penalty": 1.1,
            "top_k":         40,
            "top_p":         0.9,
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
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
                if chunk.get("done"):
                    break

    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama: {e}\nRun: ollama serve")
    except TimeoutError:
        raise RuntimeError(f"Ollama timed out after {TIMEOUT}s — try /reset")


def complete(
    messages:    list[dict],
    max_tokens:  int   = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    model:       str   = None,
) -> str:
    return "".join(stream_response(messages, max_tokens, temperature, model=model))
