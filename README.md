# 🚀 Nexus X — Offline AI Coding Assistant
## ⚡ Built by Gokulanand

A fully offline AI coding assistant that runs **100% on your machine.**  
No API keys. No subscriptions. No internet after setup.

---

# ⚙️ One-Command Install

## 🐧 Linux / macOS
```bash
curl -fsSL https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.sh | bash
```

## 🪟 Windows (PowerShell as Admin)
```powershell
irm https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.ps1 | iex
```

Then just type:

```bash
nexus
```

---

# 💾 Disk & RAM Requirements

| Setup | Disk Needed | RAM Needed |
|-------|-------------|-------------|
| Nexus Coder 1.0 default | 10GB free | 5.5GB RAM |

---

# 🧠 Running Nexus Coder 1.0 on 8GB RAM

Enable swap first:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Make permanent across reboots:

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

# 🔄 Model

```bash
nexus --model deepseek    # Nexus Coder 1.0 — default and only supported model
```

---

# 🛠️ Commands

| Command | Description |
|---------|-------------|
| /help | Show all commands |
| /run goal | Autonomous mode — agent works alone |
| /ingest path | Ingest a file or folder into vector memory |
| /memory | Show vector memory stats |
| /dataset | Show training dataset stats |
| /mistakes | Show recorded mistakes |
| /clear | Clear mistake memory |
| /reset | Clear conversation history |
| /files | List files in current directory |
| /exit | Exit Nexus |

---

# ✨ Features

- 🤖 Agent loop — thinks, acts, observes, repeats until task is done
- 🧩 Chain-of-thought reasoning — thinks step by step before answering
- 🚀 Autonomous mode `/run` — give it a goal, it plans and executes alone
- 🧠 Vector memory — remembers every conversation via ChromaDB RAG
- 📂 File ingestion — learns from your code, PDFs, and images
- ❌ Mistake learning — records failures and never repeats them
- 📊 Dataset builder — saves every conversation as JSONL training data
- 🎯 Critic system — scores and improves its own outputs
- 🤖 Powered by Nexus Coder 1.0 (deepseek-coder:6.7b)
- 🔒 100% offline after setup

---

# 🏗️ Architecture

| Module | Job |
|--------|-----|
| `model.py` | Ollama wrapper and Nexus Coder 1.0 runtime |
| `agent.py` | Think act observe loop + RAG + critic + dataset |
| `tools.py` | read_file, write_file, run_shell, search, list_tree |
| `memory.py` | Mistake learning — never repeats errors |
| `worker.py` | Autonomous planner + executor with retry logic |
| `cli.py` | Rich terminal UI |
| `main.py` | Entry point + REPL |
| `learning.py` | ChromaDB vector memory — RAG retrieval |
| `ingestor.py` | Ingests files, PDFs, images into memory |
| `dataset.py` | Saves conversations as JSONL training data |
| `critic.py` | Scores outputs, triggers improvement loop |
| `install.sh` | One command Linux/Mac installer |
| `install.ps1` | One command Windows installer |

---

# 🔍 How It Works

```text
You type a task
      ↓
learning.py searches past memory and injects relevant context
      ↓
agent THINKS via chain-of-thought reasoning
      ↓
agent ACTS using tools — reads/writes files, runs commands
      ↓
agent OBSERVES result and loops if needed
      ↓
critic.py scores output and improves if score below 6/10
      ↓
dataset.py saves conversation as training pair
      ↓
memory.py stores mistakes to avoid repeating them
      ↓
You see the final answer
```

---

# 💻 Platform Support

| Platform | Status |
|----------|--------|
| Arch Linux | Primary |
| Ubuntu / Debian | Supported |
| Fedora | Supported |
| macOS | Supported |
| Windows 10/11 | Supported |
| Android proot | deepseek-coder:6.7b only |

---

# 🌐 GitHub

https://github.com/Gokulanand-art/nexus-x

---

# 📜 License

MIT License — free to use, modify, and distribute.
