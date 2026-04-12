🚀 Nexus X — Offline AI Coding Assistant
Built by Gokulanand

Nexus X is a fully offline AI coding assistant that runs 100% on your machine.
No API keys. No subscriptions. No internet after setup.
✨ Features

    💀 Fully offline after initial install
    🧠 Autonomous coding agent (/run goal)
    🔁 Think → Act → Observe loop
    📚 Vector memory using ChromaDB RAG
    📂 File ingestion (code, PDFs, images)
    🛠 Built-in tools:
        Read files
        Write files
        Run shell commands
        Search directories
    ❌ Mistake memory system (learns from failures)
    🧪 Dataset builder for future fine-tuning
    🧑‍⚖️ Critic scoring system improves outputs
    🔄 Switchable local models:
        DeepSeek Coder
        Phi-3
        Mistral
        TinyLlama

⚡ One-Command Install
Linux / macOS

curl -fsSL https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.sh | bash

Windows (PowerShell as Admin)

irm https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.ps1 | iex

Run Nexus

nexus

---

##💾 System Requirements

Model| Disk Needed| RAM Needed
phi3 lightweight| 5 GB| 2.5 GB
deepseek-coder:6.7b| 10 GB| 5.5 GB
mistral 7b| 8–10 GB| 6 GB
tinyllama| 2 GB| 1.5 GB

---

##🧠 Recommended for 8GB RAM Users

Enable swap before running DeepSeek:

sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

Make permanent:

echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

---

##🔄 Switch Models

nexus --model deepseek
nexus --model phi3
nexus --model mistral
nexus --model tinyllama

---

##🖥 Commands

Command| Description
"/help"| Show all commands
"/run goal"| Autonomous mode
"/ingest path"| Ingest file/folder into memory
"/memory"| Show vector memory stats
"/dataset"| Show dataset stats
"/mistakes"| Show mistake log
"/clear"| Clear mistake memory
"/reset"| Reset conversation
"/files"| List current directory
"/exit"| Exit Nexus

---

##🏗 Architecture

Module| Purpose
model.py| Ollama wrapper + inference
agent.py| Main reasoning loop
tools.py| Shell/file tools
memory.py| Mistake learning
worker.py| Autonomous executor
cli.py| Terminal UI
main.py| Entry point
learning.py| ChromaDB memory
ingestor.py| File ingestion
dataset.py| JSONL dataset builder
critic.py| Output evaluator
install.sh| Linux/mac installer
install.ps1| Windows installer

---

##🔁 How Nexus X Works

User Task
   ↓
Memory Retrieval (RAG)
   ↓
Agent THINKS step-by-step
   ↓
Agent ACTS using tools
   ↓
Observes result
   ↓
Critic scores output
   ↓
Improves if needed
   ↓
Stores memory + mistakes
   ↓
Returns final answer

---

##🌍 Platform Support

Platform| Status
Arch Linux| ✅ Primary
Ubuntu / Debian| ✅ Supported
Fedora| ✅ Supported
macOS| ✅ Supported
Windows 10/11| ✅ Supported
Android Proot| ⚠ Tiny models only

---

##🔗 GitHub Repository

https://github.com/Gokulanand-art/nexus-x

---

##📜 License

MIT License — free to use, modify, and distribute.

---

##💬 About Nexus X

Nexus X is designed for developers who want:

- Privacy-first AI coding
- No cloud dependency
- Full control over local models
- Autonomous coding workflows

##Built with passion for offline AI systems 💀
