#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Nexus X v2 — One-command installer for Linux / macOS
#
#  Usage:
#    curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/install.sh | bash
#
#  Sets up: Python venv, Ollama, qwen2.5-coder:1.5b + nomic-embed-text,
#  the `nexus` launcher. Re-runs update in place and NEVER delete user data
#  (mistakes.json, .nexus_dataset/, .nexus_vectors/, .nexus_cache/, venv).
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ─── Config — point REPO at your hosted copy ─────────────────────────────────
REPO="https://github.com/Gokulanand-art/Nexus-X"
INSTALL_DIR="$HOME/.nexus"
BIN_PATH="$HOME/.local/bin/nexus"

CHAT_MODEL="qwen2.5-coder:1.5b"
EMBED_MODEL="nomic-embed-text"
PY_MAJOR_MIN=3
PY_MINOR_MIN=10

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

log()     { echo -e "${CYAN}[nexus]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ─── Banner ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
cat << 'BANNER'
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
BANNER
echo -e "${NC}"
echo -e "${CYAN}Nexus 2 — Offline AI Coding Assistant — Installer${NC}"
echo "──────────────────────────────────────────"
echo ""

# ─── 1. Detect OS + package manager ─────────────────────────────────────────
log "Detecting system..."

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Linux" ]]; then
    if command -v pacman &>/dev/null; then
        PKG_MANAGER="pacman"
    elif command -v apt-get &>/dev/null; then
        PKG_MANAGER="apt"
    elif command -v dnf &>/dev/null; then
        PKG_MANAGER="dnf"
    else
        PKG_MANAGER="unknown"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    command -v brew &>/dev/null || error "Homebrew not found. Install from https://brew.sh"
    PKG_MANAGER="brew"
else
    error "Unsupported OS: $OS"
fi

success "Detected: $OS ($ARCH) — package manager: $PKG_MANAGER"

# ─── 2. System dependencies (failures tolerated — Python is checked below) ──
log "Installing system dependencies..."

case "$PKG_MANAGER" in
    pacman)
        sudo pacman -S --needed --noconfirm python git curl base-devel cmake gcc 2>/dev/null || true
        ;;
    apt)
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv git curl build-essential cmake 2>/dev/null || true
        ;;
    dnf)
        sudo dnf install -y python3 python3-pip git curl gcc cmake make 2>/dev/null || true
        ;;
    brew)
        brew install python git curl cmake 2>/dev/null || true
        ;;
esac

success "System dependencies ready"

# ─── 3. Check Python 3.10+ ───────────────────────────────────────────────────
log "Checking Python..."

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null)
        if [[ "$VER" == "True" ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

[[ -z "$PYTHON" ]] && error "Python ${PY_MAJOR_MIN}.${PY_MINOR_MIN}+ not found. Install it first."
success "Python found: $($PYTHON --version)"

# ─── 4. Install Ollama ───────────────────────────────────────────────────────
log "Checking Ollama..."

if ! command -v ollama &>/dev/null; then
    log "Installing Ollama..."
    if [[ "$OS" == "Darwin" ]]; then
        brew install ollama 2>/dev/null || brew install --cask ollama || error "Could not install Ollama (brew install ollama)"
    else
        curl -fsSL https://ollama.com/install.sh | sh || error "Could not install Ollama"
    fi
    success "Ollama installed"
else
    success "Ollama already installed: $(ollama --version 2>/dev/null || echo 'found')"
fi

# ─── 5. Start Ollama and wait until the API answers ──────────────────────────
log "Starting Ollama service..."

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    ollama serve >/dev/null 2>&1 &
    for _ in $(seq 1 15); do
        curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 \
    || warn "Ollama API not reachable yet — model pulls will be retried by 'ollama pull' on demand."

# ─── 6. Pull models (skipped when already present) ───────────────────────────
already_pulled() { ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; }

log "Pulling chat model: $CHAT_MODEL (~1GB — only needed once)..."
if ! already_pulled "$CHAT_MODEL"; then
    ollama pull "$CHAT_MODEL"
fi
success "Chat model ready"

log "Pulling embedding model: $EMBED_MODEL (~300MB — only needed once)..."
if ! already_pulled "$EMBED_MODEL"; then
    ollama pull "$EMBED_MODEL"
fi
success "Embedding model ready"

# ─── 7. Clone / update repo (user data is never deleted) ─────────────────────
log "Setting up Nexus 2..."

mkdir -p "$INSTALL_DIR"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Updating existing install..."
    git -C "$INSTALL_DIR" pull --ff-only || warn "Update failed — continuing with existing files."
elif [[ -d "$INSTALL_DIR" ]]; then
    # Dir exists but isn't a repo (previous file-based install): overlay the
    # repo over it, preserving mistakes.json / .nexus_* / venv.
    log "Existing install dir found — merging repo files (keeping your data)..."
    TMP="$(mktemp -d)"
    git clone --depth 1 "$REPO" "$TMP" || { rm -rf "$TMP"; error "Could not clone $REPO"; }
    ( cd "$TMP" && tar cf - . ) | ( cd "$INSTALL_DIR" && tar xf - )
    rm -rf "$TMP"
else
    log "Cloning repo to $INSTALL_DIR..."
    git clone --depth 1 "$REPO" "$INSTALL_DIR"
fi

# Remove stale v1-only modules if this dir was previously a Nexus X v1 install
rm -f "$INSTALL_DIR/learning.py" "$INSTALL_DIR/critic.py" "$INSTALL_DIR/ingestor.py"

success "Nexus 2 files ready at $INSTALL_DIR"

# ─── 8. Create virtual environment (reused if present) ───────────────────────
log "Creating Python virtual environment..."

cd "$INSTALL_DIR"

if [[ ! -d "venv" ]]; then
    $PYTHON -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
success "Virtual environment ready"

# ─── 9. Install Python dependencies ──────────────────────────────────────────
log "Installing Python dependencies..."

pip install --upgrade pip --quiet || true
pip install --quiet -r requirements.txt \
    || error "pip install failed — check network/disk and re-run the installer."

success "Python dependencies installed"

# ─── 10. Default .env (only on first install) ────────────────────────────────
if [[ -f .env.example ]] && [[ ! -f .env ]]; then
    cp .env.example .env
    log "Created default .env (tweak config there if needed)"
fi

# ─── 11. Create the `nexus` launcher + PATH ──────────────────────────────────
log "Creating 'nexus' command..."

# Clear any stale v1 launcher that could shadow ours
rm -f /usr/local/bin/nexus 2>/dev/null || true

mkdir -p "$HOME/.local/bin"

cat > "$BIN_PATH" << LAUNCHER
#!/usr/bin/env bash
# Nexus 2 launcher — auto-generated by installer

NEXUS_DIR="$INSTALL_DIR"
cd "\$NEXUS_DIR"

# Start Ollama if not running
if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    ollama serve >/dev/null 2>&1 &
    for _ in \$(seq 1 10); do
        curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

source "\$NEXUS_DIR/venv/bin/activate"
exec python "\$NEXUS_DIR/main.py" "\$@"
LAUNCHER

chmod +x "$BIN_PATH"
success "Installed: nexus → $BIN_PATH"

if ! echo ":$PATH:" | grep -q ":$HOME/.local/bin:"; then
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [[ -f "$rc" ]] && ! grep -q '\.local/bin' "$rc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
            warn "Added ~/.local/bin to PATH in $rc"
        fi
    done
    warn "PATH updated — open a new terminal (or run: source ~/.bashrc) before typing 'nexus'"
fi

# ─── 12. Done ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}──────────────────────────────────────────${NC}"
echo -e "${GREEN}  Nexus 2 installed successfully!${NC}"
echo -e "${GREEN}──────────────────────────────────────────${NC}"
echo ""
echo -e "  Run it anytime with:  ${BOLD}nexus${NC}"
echo -e "  First launch downloads the Qwen tokenizer once (then fully offline)."
echo ""
