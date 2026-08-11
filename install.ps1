# Nexus 2 — Windows Installer
# Run in PowerShell as Administrator:
#   irm https://raw.githubusercontent.com/<owner>/<repo>/main/install.ps1 | iex
#
# Sets up: Python venv, Ollama, qwen2.5-coder:1.5b + nomic-embed-text,
# a `nexus.bat` launcher on your PATH. Re-runs overwrite only source files —
# user data (mistakes.json, .nexus_dataset, .nexus_vectors) is preserved.

$ErrorActionPreference = "Stop"

# ── Config — point these at your hosted copy ─────────────────────────────────
$OWNER        = "Gokulanand-art"
$REPO         = "Nexus-X"
$RAW          = "https://raw.githubusercontent.com/$OWNER/$REPO/main"
$INSTALL_DIR  = "$env:USERPROFILE\.nexus"
$VENV_DIR     = "$INSTALL_DIR\venv"
$CHAT_MODEL   = "qwen2.5-coder:1.5b"
$EMBED_MODEL  = "nomic-embed-text"
$OLLAMA_APP   = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

$TOP_FILES = @(
    "main.py", "config.py", "model.py", "tokenizer.py", "thinking.py",
    "agent.py", "tools.py", "worker.py", "memory.py", "dataset.py",
    "cli.py", "requirements.txt", "README.md", ".env.example"
)
$RAG_FILES = @(
    "__init__.py", "chunker.py", "embeddings.py", "indexer.py",
    "local_store.py", "supabase_store.py", "vector_store.py"
)

function Log  ($msg) { Write-Host "[nexus] $msg" -ForegroundColor Cyan }
function Ok   ($msg) { Write-Host "  [ok] $msg"  -ForegroundColor Green }
function Warn ($msg) { Write-Host "  [!]  $msg"  -ForegroundColor Yellow }
function Die  ($msg) { Write-Host "  [X]  $msg"  -ForegroundColor Red; exit 1 }

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if ($machine -or $user) {
        $env:PATH = (@($machine, $user) | Where-Object { $_ }) -join ";"
    }
}

function Test-OllamaUp {
    try {
        Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

Clear-Host
Write-Host @"
  _   _                       _  __
 | \ | | _____  ___   _ ___  | |/ /
 |  \| |/ _ \ \/ / | | / __| | ' /
 | |\  |  __/>  <| |_| \__ \ | . \
 |_| \_|\___/_/\_\\__,_|___/ |_|\_\

"@ -ForegroundColor Magenta
Write-Host "  Nexus 2 — Offline AI Coding Assistant" -ForegroundColor White
Write-Host "  Windows Installer`n" -ForegroundColor Gray

# ── 1. Check PowerShell version ──────────────────────────────────────────────
Log "Checking PowerShell..."
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Die "PowerShell 5+ required. Please update Windows."
}
Ok "PowerShell $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor)"

# ── 2. Check Python 3.10+ ────────────────────────────────────────────────────
Log "Checking Python 3.10+..."
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>$null
        $m = [regex]::Match(($ver -join " "), "(\d+)\s+(\d+)")
        if ($m.Success) {
            $major = [int]$m.Groups[1].Value
            $minor = [int]$m.Groups[2].Value
            if (($major -gt 3) -or ($major -eq 3 -and $minor -ge 10)) {
                $python = $cmd
                Ok "Python $major.$minor via '$cmd'"
                break
            }
        }
    } catch { continue }
}
if (-not $python) {
    Log "Python 3.10+ not found. Opening download page..."
    Start-Process "https://www.python.org/downloads/"
    Die "Install Python 3.10+ then re-run this installer."
}

# ── 3. Install Ollama ────────────────────────────────────────────────────────
Log "Checking Ollama..."
Refresh-Path
$ollamaCmd = $null
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $ollamaCmd = "ollama"
} elseif (Test-Path $OLLAMA_APP) {
    $ollamaCmd = $OLLAMA_APP
    Set-Alias ollama $OLLAMA_APP -Scope Global
    Ok "Ollama found at $OLLAMA_APP"
}

if (-not $ollamaCmd) {
    Log "Downloading Ollama installer (~100MB)..."
    $ollamaInstaller = "$env:TEMP\OllamaSetup.exe"
    try {
        $progressPreference = 'silentlyContinue'
        Invoke-WebRequest "https://ollama.com/download/OllamaSetup.exe" `
            -OutFile $ollamaInstaller -UseBasicParsing
        $progressPreference = 'Continue'
    } catch {
        Die "Could not download Ollama. Check internet and try again."
    }

    Log "Installing Ollama (silent mode — a window may flash)..."
    Start-Process $ollamaInstaller -ArgumentList "/S" -Wait
    Start-Sleep -Seconds 3

    if (-not (Test-Path $OLLAMA_APP)) {
        Warn "Silent install did not finish — starting the interactive installer."
        Warn "Click 'Install' in the window, then come back here."
        Start-Process $ollamaInstaller -Wait
    }
    Refresh-Path

    if (Test-Path $OLLAMA_APP) {
        $ollamaCmd = $OLLAMA_APP
        Set-Alias ollama $OLLAMA_APP -Scope Global
        $env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
        $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$env:LOCALAPPDATA\Programs\Ollama*") {
            [System.Environment]::SetEnvironmentVariable(
                "Path", "$userPath;$env:LOCALAPPDATA\Programs\Ollama", "User"
            )
        }
        Ok "Ollama installed and added to PATH"
    } else {
        Die "Ollama not found after install. Restart PowerShell as Admin and re-run."
    }
} else {
    Ok "Ollama ready: $(& $ollamaCmd --version 2>&1)"
}

# ── 4. Start Ollama service and wait for the API ─────────────────────────────
Log "Starting Ollama service..."
if (-not (Test-OllamaUp)) {
    Start-Process -FilePath $ollamaCmd -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 10; $i++) {
        if (Test-OllamaUp) { break }
        Start-Sleep -Seconds 2
    }
}
if (Test-OllamaUp) {
    Ok "Ollama service running"
} else {
    Warn "Ollama API not reachable yet — pulls may still work, otherwise run 'ollama serve'."
}

# ── 5. Pull models (skipped when already present) ────────────────────────────
function Test-ModelPulled($model) {
    $list = & $ollamaCmd list 2>$null
    return [bool]($list | Select-String -SimpleMatch $model)
}

Log "Pulling chat model: $CHAT_MODEL (~1GB — only needed once)..."
if (-not (Test-ModelPulled $CHAT_MODEL)) {
    try {
        & $ollamaCmd pull $CHAT_MODEL
        if ($LASTEXITCODE -ne 0) { throw "ollama pull exited $LASTEXITCODE" }
        Ok "Chat model ready"
    }
    catch { Warn "Model pull failed ($($_.Exception.Message)) — run 'ollama pull $CHAT_MODEL' manually later" }
} else {
    Ok "Chat model already present"
}

Log "Pulling embedding model: $EMBED_MODEL (~300MB — only needed once)..."
if (-not (Test-ModelPulled $EMBED_MODEL)) {
    try {
        & $ollamaCmd pull $EMBED_MODEL
        if ($LASTEXITCODE -ne 0) { throw "ollama pull exited $LASTEXITCODE" }
        Ok "Embedding model ready"
    }
    catch { Warn "Model pull failed ($($_.Exception.Message)) — run 'ollama pull $EMBED_MODEL' manually later" }
} else {
    Ok "Embedding model already present"
}

# ── 6. Download Nexus 2 source files ─────────────────────────────────────────
Log "Creating install directory: $INSTALL_DIR"
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\rag"   | Out-Null
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\scripts" | Out-Null

Log "Downloading Nexus 2 source files..."

$missing = @()
foreach ($f in $TOP_FILES) {
    try {
        Invoke-WebRequest "$RAW/$f" -OutFile "$INSTALL_DIR\$f" -UseBasicParsing
    } catch {
        $missing += $f
    }
}
foreach ($f in $RAG_FILES) {
    try {
        Invoke-WebRequest "$RAW/rag/$f" -OutFile "$INSTALL_DIR\rag\$f" -UseBasicParsing
    } catch {
        $missing += "rag/$f"
    }
}

# Optional schema — warn only (cloud backend needs it, local store does not)
try {
    Invoke-WebRequest "$RAW/scripts/supabase_schema.sql" `
        -OutFile "$INSTALL_DIR\scripts\supabase_schema.sql" -UseBasicParsing
} catch {
    Warn "supabase_schema.sql not downloaded (optional — only needed for the cloud RAG backend)"
}

foreach ($f in $missing) { Warn "Could not download $f" }
foreach ($critical in @("main.py", "config.py", "requirements.txt")) {
    if (-not (Test-Path "$INSTALL_DIR\$critical")) {
        Die "Critical file missing: $critical — aborting. Check the repo URL in this script."
    }
}
Ok "Source files ready"

# Default .env on first install
if (-not (Test-Path "$INSTALL_DIR\.env")) {
    try {
        Copy-Item "$INSTALL_DIR\.env.example" "$INSTALL_DIR\.env" -Force
        Log "Created default .env"
    } catch { Warn "Could not create .env — defaults will be used" }
}

# ── 7. Create Python virtual environment ─────────────────────────────────────
Log "Creating Python virtual environment..."
$PY = $python
try {
    & $python -m venv $VENV_DIR
    if (Test-Path "$VENV_DIR\Scripts\python.exe") {
        $PY = "$VENV_DIR\Scripts\python.exe"
        Ok "Virtual environment created: $VENV_DIR"
    } else {
        Warn "venv missing python.exe — falling back to system Python"
    }
} catch {
    Warn "venv creation failed — using system Python"
}

# ── 8. Install Python dependencies ───────────────────────────────────────────
Log "Upgrading pip..."
try { & $PY -m pip install --upgrade pip --quiet } catch { Warn "pip upgrade failed — continuing" }

Log "Installing Python dependencies (rich, numpy, tokenizers, dotenv)..."
try {
    & $PY -m pip install -r "$INSTALL_DIR\requirements.txt" --quiet
    if ($LASTEXITCODE -ne 0) { throw "pip exited $LASTEXITCODE" }
    Ok "Python dependencies installed"
} catch {
    Die "Could not install dependencies: $($_.Exception.Message). Check Python/pip and re-run."
}

# ── 9. Create nexus.bat launcher + PATH ──────────────────────────────────────
Log "Creating 'nexus' command..."

$batContent = @"
@echo off
rem Nexus 2 launcher - auto-generated by installer
curl -s -o NUL http://localhost:11434/api/tags
if errorlevel 1 start "" ollama
"$PY" "$INSTALL_DIR\main.py" %*
"@

$batPath = "$INSTALL_DIR\nexus.bat"
$batContent | Out-File $batPath -Encoding ascii
Ok "Launcher created: $batPath"

$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$INSTALL_DIR*") {
    [System.Environment]::SetEnvironmentVariable("Path", "$userPath;$INSTALL_DIR", "User")
    Ok "Added $INSTALL_DIR to user PATH"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Nexus 2 installed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  How to run:" -ForegroundColor White
Write-Host "    Option 1 — restart PowerShell, then type:   " -NoNewline
Write-Host "nexus" -ForegroundColor Cyan
Write-Host "    Option 2 — run directly now:                " -NoNewline
Write-Host "& `"$batPath`"" -ForegroundColor Cyan
Write-Host ""
Write-Host "  First launch downloads the Qwen tokenizer once (then fully offline)." -ForegroundColor Gray
Write-Host ""
