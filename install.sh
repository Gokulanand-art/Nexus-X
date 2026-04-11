#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Nexus X — One-command installer for Linux / macOS
#  Usage:
#    curl -fsSL https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.sh | bash
# ─────────────────────────────────────────────────────────────────────────────

set -e

REPO="https://github.com/Gokulanand-art/nexus-x"
INSTALL_DIR="$HOME/.nexus"

# Clean up any previous failed install
rm -rf "$INSTALL_DIR" 2>/dev/null || true
BIN_PATH="/usr/local/bin/nexus"

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
echo -e "${CYAN}Offline AI Coding Assistant — Installer${NC}"
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
    PKG_MANAGER="brew"
else
    error "Unsupported OS: $OS"
fi

success "Detected: $OS ($ARCH) — package manager: $PKG_MANAGER"

# ─── 2. Install system dependencies ─────────────────────────────────────────
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
        command -v brew &>/dev/null || error "Homebrew not found. Install from https://brew.sh"
        brew install python git curl cmake 2>/dev/null || true
        ;;
esac

success "System dependencies ready"

# ─── 3. Check Python ─────────────────────────────────────────────────────────
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

[[ -z "$PYTHON" ]] && error "Python 3.10+ not found. Install it first."
success "Python found: $($PYTHON --version)"

# ─── 4. Install Ollama ───────────────────────────────────────────────────────
log "Checking Ollama..."

if ! command -v ollama &>/dev/null; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed"
else
    success "Ollama already installed: $(ollama --version 2>/dev/null || echo 'found')"
fi

# ─── 5. Start Ollama + pull model ────────────────────────────────────────────
log "Starting Ollama service..."

if ! pgrep -x "ollama" &>/dev/null; then
    ollama serve &>/dev/null &
    sleep 3

# Remove stale nexus launcher if exists
rm -f /usr/local/bin/nexus 2>/dev/null || true
fi

log "Pulling Phi-3 Mini model (~2.3GB — only needed once)..."
ollama pull phi3
success "Model ready"

# ─── 6. Clone / update repo ──────────────────────────────────────────────────
log "Setting up Nexus X..."

if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Updating existing install..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    log "Cloning repo to $INSTALL_DIR..."
    git clone "$REPO" "$INSTALL_DIR"
fi

success "Nexus X files ready at $INSTALL_DIR"

# ─── 7. Create virtual environment ───────────────────────────────────────────
log "Creating Python virtual environment..."

cd "$INSTALL_DIR"

if [[ ! -d "venv" ]]; then
    $PYTHON -m venv venv
fi

source venv/bin/activate
success "Virtual environment ready"

# ─── 8. Install Python dependencies ──────────────────────────────────────────
log "Installing Python dependencies..."

pip install --upgrade pip --quiet
pip install --quiet chromadb rich

success "Python dependencies installed"

# ─── 9. Create the `nexus` launcher ──────────────────────────────────────────
log "Creating 'nexus' command..."

cat > /tmp/nexus_launcher << LAUNCHER
#!/usr/bin/env bash
# Nexus X launcher — auto-generated by installer

NEXUS_DIR="$INSTALL_DIR"
cd "\$NEXUS_DIR"

# Start Ollama if not running
if ! pgrep -x "ollama" &>/dev/null; then
    ollama serve &>/dev/null &
    sleep 2
fi

source "\$NEXUS_DIR/venv/bin/activate"
exec python "\$NEXUS_DIR/main.py" "\$@"
LAUNCHER

chmod +x /tmp/nexus_launcher

# Try to install to /usr/local/bin, fallback to ~/.local/bin
if sudo mv /tmp/nexus_launcher "$BIN_PATH" 2>/dev/null; then
    success "Installed: nexus → $BIN_PATH"
else
    mkdir -p "$HOME/.local/bin"
    mv /tmp/nexus_launcher "$HOME/.local/bin/nexus"
    BIN_PATH="$HOME/.local/bin/nexus"
    success "Installed: nexus → $BIN_PATH"

    # Add to PATH if not already there
    SHELL_RC=""
    if [[ -f "$HOME/.zshrc" ]]; then SHELL_RC="$HOME/.zshrc"
    elif [[ -f "$HOME/.bashrc" ]]; then SHELL_RC="$HOME/.bashrc"
    fi

    if [[ -n "$SHELL_RC" ]] && ! grep -q '\.local/bin' "$SHELL_RC"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        warn "Added ~/.local/bin to PATH in $SHELL_RC"
        warn "Run: source $SHELL_RC  (or open a new terminal)"
    fi
fi

# ─── 10. Done ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}──────────────────────────────────────────${NC}"
echo -e "${GREEN}  Nexus X installed successfully!${NC}"
echo -e "${GREEN}──────────────────────────────────────────${NC}"
echo ""
echo -e "  Run it anytime with:  ${BOLD}nexus${NC}"
echo ""
