Nexus X — Offline AI Coding Assistant

«Built by Gokulanand»

A fully offline AI coding assistant that runs 100% on your machine.
No API keys. No subscriptions. No internet after setup.

---

One-Command Install

Linux / macOS

curl -fsSL https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.sh | bash

Windows (PowerShell as Admin)

irm https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.ps1 | iex

Then just type:

nexus

---

Disk & RAM Requirements

Setup| Disk Needed| RAM Needed
phi3 lightweight| 5GB free| 2.5GB RAM
deepseek-coder:6.7b default| 10GB free| 5.5GB RAM

---

Running DeepSeek on 8GB RAM

Enable swap first:

sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

Make permanent across reboots:

echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

---

Switch Models

nexus --model deepseek    # deepseek-coder:6.7b — best for code, needs 5.5GB RAM
nexus --model phi3        # phi3 mini — fast, needs 2.5GB RAM
nexus --model mistral     # mistral 7b — best for chat
nexus --model tinyllama   # lightest — works on phone and low RAM devices

---

Commands

Command| Description
/help| Show all commands
/run goal| Autonomous mode — agent works alone
/ingest path| Ingest a file or folder into vector memory
/memory| Show vector memory stats
/dataset| Show training dataset stats
/mistakes| Show recorded mistakes
/clear| Clear mistake memory
/reset| Clear conversation history
/files| List files in current directory
/exit| Exit Nexus

---

Features

- Agent loop — thinks, acts, observes, repeats until task is done
- Chain-of-thought reasoning — thinks step by step before answering
- Autonomous mode "/run" — give it a goal, it plans and executes alone
- Vector memory — remembers every conversation via ChromaDB RAG
- File ingestion — learns from your code, PDFs, and images
- Mistake learning — records failures and never repeats them
- Dataset builder — saves every conversation as JSONL training data
- Critic system — scores and improves its own outputs
- Model switchable — deepseek, phi3, mistral, tinyllama
- 100% offline after setup

---

Architecture

Module| Job
"model.py"| Ollama wrapper, streaming inference, model switching
"agent.py"| Think act observe loop + RAG + critic + dataset
"tools.py"| read_file, write_file, run_shell, search, list_tree
"memory.py"| Mistake learning — never repeats errors
"worker.py"| Autonomous planner + executor with retry logic
"cli.py"| Rich terminal UI
"main.py"| Entry point + REPL
"learning.py"| ChromaDB vector memory — RAG retrieval
"ingestor.py"| Ingests files, PDFs, images into memory
"dataset.py"| Saves conversations as JSONL training data
"critic.py"| Scores outputs, triggers improvement loop
"install.sh"| One command Linux/Mac installer
"install.ps1"| One command Windows installer

---

How It Works

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

---

Platform Support

Platform| Status
Arch Linux| Primary
Ubuntu / Debian| Supported
Fedora| Supported
macOS| Supported
Windows 10/11| Supported
Android proot| phi3 or tinyllama only

---

GitHub

https://github.com/Gokulanand-art/nexus-x

---

License

MIT License — free to use, modify, and distribute.
