# Nexus — Offline AI Coding Assistant

A terminal-based AI coding agent powered by **Phi-3 Mini (3.8B)** running fully offline via `llama.cpp`.  
No API key. No internet. No cloud. Runs on any laptop CPU.

---

## What it does

- Reads, writes, and creates code files
- Runs shell commands (with your approval)
- Understands your repo structure
- **Learns from its own mistakes** — records every failure and never repeats it
- Single clean agent loop — no sub-agents, no mess

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

> On some systems, install llama-cpp-python separately for CPU-only build:
> ```bash
> CMAKE_ARGS="-DLLAMA_BLAS=OFF" pip install llama-cpp-python --break-system-packages
> ```

### 2. Download the model (one time, ~2.3GB)

```bash
mkdir -p models
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
    Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models
```

### 3. Run

```bash
python main.py
```

---

## Usage

```
> Create a Flask app with a /health endpoint
> Read utils.py and fix the bug on line 42
> List all files in this directory
> Run the tests and tell me what failed
```

### Commands

| Command      | Description                        |
|--------------|------------------------------------|
| `/help`      | Show all commands                  |
| `/reset`     | Clear conversation history         |
| `/mistakes`  | View recorded mistake memory       |
| `/clear`     | Erase all recorded mistakes        |
| `/files`     | List files in current directory    |
| `/exit`      | Quit nexus                         |

---

## How mistake learning works

Every time a tool fails or a command errors out, nexus records:
- What the failing pattern was
- Why it failed
- What the fix is

These are saved to `mistakes.json` and **injected into every prompt** at the top.  
The model reads its own past failures before every action and avoids repeating them.

Example entry in `mistakes.json`:
```json
{
  "pattern": "running pip without --break-system-packages",
  "cause": "modern Linux blocks global pip installs",
  "fix": "always use pip install X --break-system-packages",
  "count": 2
}
```

---

## Project structure

```
nexus/
├── main.py         Entry point, REPL loop
├── agent.py        Single agent loop (think → decide → act → observe)
├── model.py        Phi-3 mini wrapper (llama-cpp-python)
├── tools.py        read_file, write_file, run_shell, list_tree, search
├── memory.py       Mistake learning system
├── cli.py          Rich terminal UI
├── mistakes.json   Persistent mistake store (auto-updated)
└── models/         Put your .gguf file here
```

---

## System requirements

| Requirement | Minimum          |
|-------------|------------------|
| RAM         | 4GB (8GB recommended) |
| CPU         | Any x86-64 or ARM |
| Python      | 3.10+            |
| Disk        | ~2.5GB for model |
| GPU         | Not required     |

---

## CLI flags

```bash
python main.py --model /path/to/model.gguf   # custom model path
python main.py --verbose                      # show llama.cpp logs
python main.py --no-confirm                   # skip confirmation prompts (dev only)
```

---

## Offline, always

Nexus makes zero network calls at runtime.  
The model runs entirely in-process via `llama-cpp-python`.  
Your code never leaves your machine.
