# Nexus X — Windows Installer
# Run in PowerShell as Administrator:
# irm https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$INSTALL_DIR = "$env:USERPROFILE\.nexus-x"
$RAW         = "https://raw.githubusercontent.com/Gokulanand-art/nexus-x/main"
$MODEL       = "phi3"
$VENV_DIR    = "$INSTALL_DIR\venv"

function Log  ($msg) { Write-Host "[nexus] $msg" -ForegroundColor Cyan }
function Ok   ($msg) { Write-Host "  [ok] $msg"  -ForegroundColor Green }
function Warn ($msg) { Write-Host "  [!]  $msg"  -ForegroundColor Yellow }
function Die  ($msg) { Write-Host "  [X]  $msg"  -ForegroundColor Red; exit 1 }

function Refresh-Path {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

Clear-Host
Write-Host @"
  _   _                       _  __
 | \ | | _____  ___   _ ___  | |/ /
 |  \| |/ _ \ \/ / | | / __| | ' /
 | |\  |  __/>  <| |_| \__ \ | . \
 |_| \_|\___/_/\_\\__,_|___/ |_|\_\

"@ -ForegroundColor Magenta
Write-Host "  Nexus X — Offline AI Coding Assistant" -ForegroundColor White
Write-Host "  Windows Installer`n" -ForegroundColor Gray

# ── 1. Check PowerShell version ───────────────────────────────────────────────
Log "Checking PowerShell..."
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Die "PowerShell 5+ required. Please update Windows."
}
Ok "PowerShell $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor)"

# ── 2. Check Python ───────────────────────────────────────────────────────────
Log "Checking Python..."
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3") {
            $python = $cmd
            Ok "$ver"
            break
        }
    } catch { continue }
}
if (-not $python) {
    Log "Python not found. Opening download page..."
    Start-Process "https://www.python.org/downloads/"
    Die "Install Python 3.10+ then re-run this installer."
}

# ── 3. Install Ollama ─────────────────────────────────────────────────────────
Log "Checking Ollama..."
Refresh-Path
$ollamaExists = Get-Command ollama -ErrorAction SilentlyContinue

if (-not $ollamaExists) {
    Log "Downloading Ollama installer (~100MB)..."
    $ollamaInstaller = "$env:TEMP\OllamaSetup.exe"

    try {
        $progressPreference = 'silentlyContinue'
        Invoke-WebRequest "https://ollama.com/download/OllamaSetup.exe" `
            -OutFile $ollamaInstaller `
            -UseBasicParsing
        $progressPreference = 'Continue'
    } catch {
        Die "Could not download Ollama. Check internet and try again."
    }

    Log "Installing Ollama (this opens an installer window — click Install)..."
    $proc = Start-Process $ollamaInstaller -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        Warn "Installer exited with code $($proc.ExitCode) — trying to continue anyway"
    }

    # Refresh PATH so ollama command is found
    Refresh-Path
    Start-Sleep -Seconds 3

    # Ollama may be in AppData after install
    $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $ollamaPath) {
        $env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
        [System.Environment]::SetEnvironmentVariable(
            "Path",
            [System.Environment]::GetEnvironmentVariable("Path","User") + ";$env:LOCALAPPDATA\Programs\Ollama",
            "User"
        )
        Ok "Ollama installed and added to PATH"
    } else {
        Warn "Could not find ollama.exe — it may still be installing. Waiting 10s..."
        Start-Sleep -Seconds 10
        Refresh-Path
    }
} else {
    Ok "Ollama already installed: $(ollama --version 2>&1)"
}

# Verify ollama is callable after all the above
Refresh-Path
try {
    $ollamaVer = & ollama --version 2>&1
    Ok "Ollama ready: $ollamaVer"
} catch {
    # Try direct path as last resort
    $ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $ollamaExe) {
        Set-Alias -Name ollama -Value $ollamaExe -Scope Global
        Ok "Ollama found at $ollamaExe"
    } else {
        Die "Ollama not found after install. Please restart PowerShell as Admin and re-run."
    }
}

# ── 4. Start Ollama service ───────────────────────────────────────────────────
Log "Starting Ollama service..."
try {
    $running = Invoke-RestMethod "http://localhost:11434/api/tags" -ErrorAction SilentlyContinue
    Ok "Ollama service already running"
} catch {
    Log "Launching ollama serve..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5

    # Verify it started
    try {
        Invoke-RestMethod "http://localhost:11434/api/tags" | Out-Null
        Ok "Ollama service started"
    } catch {
        Warn "Ollama service may not have started — will try pulling anyway"
    }
}

# ── 5. Pull model ─────────────────────────────────────────────────────────────
Log "Pulling model: $MODEL (~2.3GB — downloaded once, works offline forever)..."
Log "This will take a few minutes on first run..."
try {
    & ollama pull $MODEL
    Ok "Model ready: $MODEL"
} catch {
    Warn "Model pull failed — you can run 'ollama pull $MODEL' manually later"
}

# ── 6. Create install directory ───────────────────────────────────────────────
Log "Creating install directory: $INSTALL_DIR"
New-Item -ItemType Directory -Force -Path $INSTALL_DIR | Out-Null

# ── 7. Download Nexus X source files ─────────────────────────────────────────
Log "Downloading Nexus X source files..."

$FILES = @(
    "main.py", "agent.py", "model.py", "tools.py", "memory.py",
    "cli.py",  "worker.py", "learning.py", "ingestor.py",
    "dataset.py", "critic.py", "requirements.txt"
)

$downloaded = 0
foreach ($f in $FILES) {
    try {
        Invoke-WebRequest "$RAW/$f" -OutFile "$INSTALL_DIR\$f" -UseBasicParsing
        $downloaded++
    } catch {
        Warn "Could not download $f — will skip"
    }
}

# Init data files
"[]" | Out-File "$INSTALL_DIR\mistakes.json"  -Encoding utf8
""   | Out-File "$INSTALL_DIR\nexus_dataset.jsonl" -Encoding utf8

Ok "Downloaded $downloaded/$($FILES.Count) source files"

# ── 8. Create Python virtual environment ──────────────────────────────────────
Log "Creating Python virtual environment..."
try {
    & $python -m venv $VENV_DIR
    Ok "Virtual environment created: $VENV_DIR"
} catch {
    Warn "venv creation failed — will use system Python"
    $VENV_DIR = $null
}

# Determine pip and python paths
if ($VENV_DIR -and (Test-Path "$VENV_DIR\Scripts\python.exe")) {
    $PY  = "$VENV_DIR\Scripts\python.exe"
    $PIP = "$VENV_DIR\Scripts\pip.exe"
} else {
    $PY  = $python
    $PIP = "$python -m pip"
}

# ── 9. Install Python dependencies ────────────────────────────────────────────
Log "Upgrading pip..."
try {
    & $PY -m pip install --upgrade pip --quiet
    Ok "pip upgraded"
} catch {
    Warn "pip upgrade failed — continuing"
}

Log "Installing rich (terminal UI)..."
try {
    & $PY -m pip install rich --quiet
    Ok "rich installed"
} catch {
    Die "Could not install rich. Check Python/pip is working."
}

Log "Installing chromadb (vector memory — optional, ~200MB)..."
try {
    & $PY -m pip install chromadb --quiet
    Ok "chromadb installed (vector memory enabled)"
} catch {
    Warn "chromadb install failed — vector memory disabled (Nexus still works fine)"
}

# ── 10. Create nexus.bat launcher ─────────────────────────────────────────────
Log "Creating nexus launcher..."

$batContent = "@echo off
`"$PY`" `"$INSTALL_DIR\main.py`" %*"

$batPaths = @(
    "$env:USERPROFILE\AppData\Local\Microsoft\WindowsApps\nexus.bat",
    "$env:USERPROFILE\.local\bin\nexus.bat",
    "$INSTALL_DIR\nexus.bat"
)

$launcherInstalled = $false
foreach ($batPath in $batPaths) {
    try {
        $dir = Split-Path $batPath
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        $batContent | Out-File $batPath -Encoding ascii
        $launcherInstalled = $true
        Ok "Launcher created: $batPath"
        break
    } catch { continue }
}

if (-not $launcherInstalled) {
    $batContent | Out-File "$INSTALL_DIR\nexus.bat" -Encoding ascii
    Warn "Launcher at: $INSTALL_DIR\nexus.bat"
}

# Add INSTALL_DIR to user PATH if needed
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$INSTALL_DIR*") {
    [System.Environment]::SetEnvironmentVariable(
        "Path", "$userPath;$INSTALL_DIR", "User"
    )
    Ok "Added $INSTALL_DIR to user PATH"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Nexus X installed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  How to run:" -ForegroundColor White
Write-Host "    Option 1 — restart PowerShell then type:  " -NoNewline
Write-Host "nexus" -ForegroundColor Cyan
Write-Host "    Option 2 — run directly now:              " -NoNewline
Write-Host "& `"$INSTALL_DIR\nexus.bat`"" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Switch model:" -ForegroundColor White
Write-Host "    nexus --model deepseek   " -ForegroundColor Cyan -NoNewline
Write-Host "(best for code, needs 5.5GB RAM)" -ForegroundColor Gray
Write-Host "    nexus --model phi3       " -ForegroundColor Cyan -NoNewline
Write-Host "(fast, needs 2.5GB RAM)" -ForegroundColor Gray
Write-Host "    nexus --model tinyllama  " -ForegroundColor Cyan -NoNewline
Write-Host "(lightest, 800MB RAM)" -ForegroundColor Gray
Write-Host ""
Write-Host "  Inside Nexus type /help for all commands" -ForegroundColor Gray
Write-Host ""
