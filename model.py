"""
model.py — Ollama connector for Nexus v2 (Qwen2.5-Coder 1.5B).

Everything streams. Stop tokens are set per-model to kill repetition loops.
Exposes:
  - chat()      one-shot completion
  - stream()    generator of (text, done, usage) chunks
  - embed()     batch embeddings for RAG (nomic-embed-text)
  - health()    ollama running? model pulled?
"""

import json
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse
from typing import Iterator, Optional

import config


class ModelError(RuntimeError):
    pass


def _post(path: str, payload: dict) -> HTTPResponse:
    req = urllib.request.Request(
        url=f"{config.OLLAMA_HOST}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT)


def _get(path: str) -> dict:
    with urllib.request.urlopen(
        f"{config.OLLAMA_HOST}{path}", timeout=10
    ) as resp:
        return json.loads(resp.read())


# ─── Health ─────────────────────────────────────────────────────────────────

def is_running() -> bool:
    try:
        _get("/api/tags")
        return True
    except Exception:
        return False


def is_model_available(model: Optional[str] = None) -> bool:
    name = model or config.CHAT_MODEL
    try:
        tags = _get("/api/tags")
        return any(m["name"].split(":")[0] == name.split(":")[0]
                   for m in tags.get("models", []))
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        tags = _get("/api/tags")
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return []


def get_short_name() -> str:
    return config.CHAT_MODEL.split(":")[0]


def get_threads() -> int:
    return config.NUM_THREADS


def set_model(name: str) -> tuple[bool, str]:
    """Switch CHAT_MODEL to any pulled model (Claude Code /model)."""
    available = list_models()
    match = next((m for m in available if m.split(":")[0] == name.split(":")[0]),
                 None)
    if not match:
        return False, (f"Model '{name}' not pulled. Available: "
                       + (", ".join(available) or "none"))
    config.CHAT_MODEL = match
    return True, f"Switched to {match} (offline)"


# ─── Generation ─────────────────────────────────────────────────────────────

def _stop_tokens(model: Optional[str] = None) -> list[str]:
    name = model or config.CHAT_MODEL
    base = name.split(":")[0].lower()
    if "qwen" in base:
        return ["<|im_end|>", "<|im_start|>", "User:", "Assistant:"]
    if "gemma" in base or "phi" in base or "granite" in base:
        return ["<end_of_turn>", "<|end|>", "<|user|>", "<|assistant|>"]
    return ["<|end|>", "</s>", "User:", "Human:"]


def _sampling(temperature: float) -> dict:
    """Balanced sampling — fast and stable on small models."""
    return {
        "temperature":   max(0.0, min(1.5, temperature)),
        "top_k":         40,
        "top_p":         0.9,
        "repeat_penalty": 1.1,
    }


def stream(
    messages: list[dict],
    max_tokens: int = config.MAX_GENERATION,
    temperature: float = config.TEMPERATURE,
    model: Optional[str] = None,
    num_ctx: int = config.NUM_CTX,
    stop: Optional[list[str]] = None,
) -> Iterator[tuple[str, bool, dict]]:
    """
    Yield (text, done, usage) tuples while the model generates.
    usage = {"prompt_tokens": int, "gen_tokens": int} on the done chunk.
    """
    payload = {
        "model":    model or config.CHAT_MODEL,
        "messages": messages,
        "stream":   True,
        "keep_alive": config.KEEP_ALIVE,
        "options": {
            "num_predict": max_tokens,
            "num_ctx":     num_ctx,
            "num_thread":  config.NUM_THREADS,
            "stop":        stop if stop is not None else _stop_tokens(model),
            **_sampling(temperature),
        },
    }
    try:
        with _post("/api/chat", payload) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message", {})
                text = msg.get("content", "")
                done = bool(chunk.get("done"))
                usage = {
                    "prompt_tokens": chunk.get("prompt_eval_count", 0),
                    "gen_tokens":    chunk.get("eval_count", 0),
                }
                yield text, done, usage
                if done:
                    return
    except urllib.error.URLError as e:
        raise ModelError(f"Cannot reach Ollama at {config.OLLAMA_HOST}: {e}")
    except TimeoutError:
        raise ModelError(
            f"Ollama timed out after {config.OLLAMA_TIMEOUT}s — "
            "the model may still be loading. Try again."
        )


def complete(
    messages: list[dict],
    max_tokens: int = config.MAX_GENERATION,
    temperature: float = config.TEMPERATURE,
    model: Optional[str] = None,
    num_ctx: int = config.NUM_CTX,
) -> str:
    """One-shot completion (used for thinking, planning, extraction)."""
    out = []
    for text, done, _ in stream(
        messages, max_tokens=max_tokens, temperature=temperature,
        model=model, num_ctx=num_ctx,
    ):
        out.append(text)
    return "".join(out)


# ─── Embeddings (RAG) ───────────────────────────────────────────────────────

def embed(texts: list[str], model: Optional[str] = None) -> list[list[float]]:
    """
    Batch embeddings via Ollama. Returns list of vectors.
    Empty/whitespace texts yield zero vectors (dropped by caller).
    """
    name = model or config.EMBED_MODEL
    vectors: list[list[float]] = []
    batch = 16
    for i in range(0, len(texts), batch):
        group = texts[i:i + batch]
        payload = {
            "model": name,
            "input": group,
            "keep_alive": config.KEEP_ALIVE,
        }
        try:
            with _post("/api/embed", payload) as resp:
                data = json.loads(resp.read())
                vectors.extend(data.get("embeddings", []))
        except Exception:
            # Fall back to one-by-one for the batch on failure
            for t in group:
                try:
                    with _post("/api/embed", {
                        "model": name, "input": t,
                        "keep_alive": config.KEEP_ALIVE,
                    }) as r:
                        d = json.loads(r.read())
                        vectors.append(d["embeddings"][0])
                except Exception:
                    vectors.append([0.0] * config.EMBED_DIM)
    return vectors


def embed_one(text: str, model: Optional[str] = None) -> list[float]:
    vecs = embed([text], model=model)
    return vecs[0] if vecs else [0.0] * config.EMBED_DIM


# ─── Warm-up ────────────────────────────────────────────────────────────────

def warmup(model: Optional[str] = None) -> float:
    """Load the model into RAM so the first real prompt is fast."""
    t0 = time.time()
    try:
        complete(
            [{"role": "user", "content": "ping"}],
            max_tokens=1, temperature=0.0,
            model=model or config.CHAT_MODEL,
        )
    except Exception:
        pass
    return time.time() - t0
