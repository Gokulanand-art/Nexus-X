# Nexus 2 — Offline AI Coding Assistant (2B-class)

Reconstruction of **Nexus X** rebuilt for a **1.5B model** (`qwen2.5-coder:1.5b`)
with a **massive hybrid RAG** (Supabase pgvector + local SQLite fallback),
**Claude-style extended thinking**, tokenizer-exact context management, and
fast single-pass streaming. 100% offline after setup.

## Setup

### One-command install

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Gokulanand-art/Nexus-X/main/install.sh | bash
```

**Windows (PowerShell as Admin):**
```powershell
irm https://raw.githubusercontent.com/Gokulanand-art/Nexus-X/main/install.ps1 | iex
```

Then just type:

```bash
nexus
```

Run `nexus` from any folder, like `kilo` — the session works on the folder
you're in (it finds CLAUDE.md project instructions, ingests that folder into
RAG, and `/init` writes project files there), while your code, models, and
data stay in `~/.nexus`.

The installer handles Python 3.10+, Ollama, the `qwen2.5-coder:1.5b` and
`nomic-embed-text` models, a virtualenv, and the `nexus` launcher. Re-running
it updates in place without touching your data.

### Manual setup

```bash
pip install -r requirements.txt

ollama pull qwen2.5-coder:1.5b     # chat model (~1 GB)
ollama pull nomic-embed-text       # embeddings for RAG (~300 MB)
ollama serve

nexus  # or: python3 main.py
```

First launch downloads the exact Qwen tokenizer once (then fully offline).
On first prompt the model loads and warms up (5–20s); after that it stays
resident.

## How it differs from Nexus X v1

| | v1 | v2 |
|---|---|---|
| Model | deepseek-coder:6.7b | qwen2.5-coder:1.5b |
| RAG | ChromaDB, simple similarity | Hybrid (semantic + FTS5/pgvector), RRF fusion, heading-aware chunking, disk-cached embeddings |
| Reasoning | critic loop after answer | Claude-style thinking pass *before* the answer |
| Context | char-based trim | exact Qwen tokenizer, token-budget trimming |
| Tool calls | model decides freely | intent detection + prefill steering (reliable on small models) |
| Speed | critic re-generations | single streaming pass, self-heal only on failure |

## Commands

| Command | Description |
|---|---|
| `/run <goal>` | Autonomous mode — planner + executor + retries |
| `/ingest <path>` | Ingest file or folder into the vector store |
| `/rag <question>` | Query the knowledge base directly |
| `/think on\|off\|auto` | Extended thinking toggle |
| `/memory`, `/dataset`, `/mistakes` | Store stats |
| `/clear`, `/reset`, `/files`, `/help`, `/exit` | as before |

## RAG backends

Default is **local**: SQLite FTS5 (full-text BM25) + numpy batch cosine,
RRF fusion — zero config, zero cloud.

For the **Supabase free tier** (pgvector, scales to 100k+ chunks):

1. Create a project at supabase.com (free)
2. Run `scripts/supabase_schema.sql` in the SQL editor
3. Install client + set credentials:

```bash
pip install supabase
cp .env.example .env    # fill SUPABASE_URL + SUPABASE_SERVICE_KEY
```

Nexus auto-detects the backend: credentials present → supabase, else local.

## Architecture

```
main.py        REPL + preflight (Ollama, models, store)
agent.py       Claude-style loop: RAG → thinking → stream → tools → persist
thinking.py    extended thinking (budget-based, tag-stripped streaming)
tokenizer.py   exact Qwen BPE counts + token-budget trimming
model.py       Ollama chat/stream/embed client (urllib, no SDK)
tools.py       read/write/list/run/search — safety-gated
memory.py      mistakes.json — injected into prompts, guard rails
worker.py      /run autonomous planner+executor
rag/
  chunker.py       recursive, heading-aware, overlapping chunks
  embeddings.py    Ollama nomic-embed-text, LRU + disk cache
  vector_store.py  store interface (backend auto-factory)
  local_store.py   SQLite FTS5 hybrid (default)
  supabase_store.py pgvector hybrid via PostgREST (optional)
  indexer.py       files/pdf/OCR → chunks → embeddings → upsert
scripts/supabase_schema.sql   one-time cloud RAG setup
```

## Notes

- Tweaks live in `config.py` / `.env` (context budget, temperature,
  thinking budget, top-k...).
- Every exchange is saved to `.nexus_dataset/conversations.jsonl` with the
  reasoning + RAG sources attached — ready for future fine-tuning.
- Minimal RAM: ~1.5GB for the model + embeddings. Runs on 8GB machines.