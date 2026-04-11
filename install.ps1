# ─────────────────────────────────────────────────────────────────────────────
#  Nexus X — One-command installer for Windows (PowerShell)
#  Usage (run in PowerShell as Administrator):
#    irm https://raw.githubusercontent.com/Gokulanand-art/nexus-code/main/install.ps1 | iex
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

$REPO       = "https://github.com/Gokulanand-art/nexus-code"
$INSTALL_DIR = "$env:USERPROFILE\.nexus"
$NEXUS_BAT  = "$env:USERPROFILE\AppData\Local\Microsoft\WindowsApps\nexus.bat"

# ─── Colors ──────────────────────────────────────────────────────────────────
function Log    { param($m) Write-Host "[nexus] $m" -ForegroundColor Cyan }
function Ok     { param($m) Write-Host "[OK] $m"    -ForegroundColor Green }
function Warn   { param($m) Write-Host "[!]  $m"    -ForegroundColor Yellow }
function Err    { param($m) Write-Host "[X]  $m"    -ForegroundColor Red; exit 1 }

# ─── Banner ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host " ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗" -ForegroundColor Magenta
Write-Host " ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝" -ForegroundColor Magenta
Write-Host " ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗" -ForegroundColor Magenta
Write-Host " ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║" -ForegroundColor Magenta
Write-Host " ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║" -ForegroundColor Magenta
Write-Host " ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝" -ForegroundColor Magenta
Write-Host ""
Write-Host " Offline AI Coding Assistant — Windows Installer" -ForegroundColor Cyan
Write-Host "──────────────────────────────────────────────────"
Write-Host ""

# ─── 1. Check PowerShell version ────────────────────────────────────────────
Log "Checking PowerShell version..."
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Err "PowerShell 5+ required. Please update Windows."
}
Ok "PowerShell $($PSVersionTable.PSVersion)"

# ─── 2. Check / install winget ──────────────────────────────────────────────
Log "Checking winget..."
$hasWinget = Get-Command winget -ErrorAction SilentlyContinue
if (-not $hasWinget) {
    Warn "winget not found. Install App Installer from the Microsoft Store."
    Warn "Then re-run this installer."
    exit 1
}
Ok "winget found"

# ─── 3. Install Git ──────────────────────────────────────────────────────────
Log "Checking Git..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Log "Installing Git..."
    winget install --id Git.Git -e --source winget --silent --accept-package-agreements --accept-source-agreements
    $env:PATH += ";C:\Program Files\Git\cmd"
    Ok "Git installed"
} else {
    Ok "Git already installed"
}

# ─── 4. Install Python ───────────────────────────────────────────────────────
Log "Checking Python..."
$pyCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(sys.version_info >= (3,10))" 2>$null
        if ($ver -eq "True") { $pyCmd = $cmd; break }
    } catch {}
}

if (-not $pyCmd) {
    Log "Installing Python 3.11..."
    winget install --id Python.Python.3.11 -e --source winget --silent --accept-package-agreements --accept-source-agreements
    $env:PATH += ";$env:LOCALAPPDATA\Programs\Python\Python311;$env:LOCALAPPDATA\Programs\Python\Python311\Scripts"
    $pyCmd = "python"
    Ok "Python installed"
} else {
    Ok "Python found: $(& $pyCmd --version)"
}

# ─── 5. Install Ollama ───────────────────────────────────────────────────────
Log "Checking Ollama..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Log "Installing Ollama..."
    winget install --id Ollama.Ollama -e --source winget --silent --accept-package-agreements --accept-source-agreements
    # Give it a moment to register
    Start-Sleep -Seconds 3
    Ok "Ollama installed"
} else {
    Ok "Ollama already installed"
}

# ─── 6. Start Ollama + pull model ────────────────────────────────────────────
Log "Starting Ollama..."
$ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

Log "Pulling Phi-3 Mini model (~2.3GB — only needed once)..."
& ollama pull phi3
Ok "Model ready"

# ─── 7. Clone / update repo ──────────────────────────────────────────────────
Log "Setting up Nexus X..."

if (Test-Path "$INSTALL_DIR\.git") {
    Log "Updating existing install..."
    git -C $INSTALL_DIR pull --ff-only
} else {
    Log "Cloning to $INSTALL_DIR..."
    git clone $REPO $INSTALL_DIR
}
Ok "Nexus X files ready at $INSTALL_DIR"

# ─── 8. Create virtual environment ───────────────────────────────────────────
Log "Creating Python virtual environment..."
Set-Location $INSTALL_DIR

if (-not (Test-Path "venv")) {
    & $pyCmd -m venv venv
}

& "$INSTALL_DIR\venv\Scripts\Activate.ps1"
Ok "Virtual environment ready"

# ─── 9. Install Python dependencies ──────────────────────────────────────────
Log "Installing Python dependencies..."
& "$INSTALL_DIR\venv\Scripts\pip.exe" install --upgrade pip --quiet
& "$INSTALL_DIR\venv\Scripts\pip.exe" install chromadb rich --quiet
Ok "Python dependencies installed"

# ─── 10. Create nexus.bat launcher ───────────────────────────────────────────
Log "Creating 'nexus' command..."

$launcherDir = "$env:USERPROFILE\AppData\Local\Microsoft\WindowsApps"
$launcherPath = "$launcherDir\nexus.bat"

$batContent = @"
@echo off
:: Nexus X launcher — auto-generated by installer

set NEXUS_DIR=$INSTALL_DIR

:: Start Ollama if not running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    start /B "" ollama serve >NUL 2>&1
    timeout /t 3 /nobreak >NUL
)

call "%NEXUS_DIR%\venv\Scripts\activate.bat"
python "%NEXUS_DIR%\main.py" %*
"@

# Ensure launcher dir exists
if (-not (Test-Path $launcherDir)) {
    New-Item -ItemType Directory -Path $launcherDir -Force | Out-Null
}

Set-Content -Path $launcherPath -Value $batContent -Encoding ASCII
Ok "Launcher created: $launcherPath"

# ─── 11. Verify PATH ─────────────────────────────────────────────────────────
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*WindowsApps*") {
    Warn "WindowsApps may not be in your PATH."
    Warn "If 'nexus' doesn't work, add this to your PATH manually:"
    Warn "  $launcherDir"
}

# ─── 12. Done ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "──────────────────────────────────────────" -ForegroundColor Green
Write-Host "  Nexus X installed successfully!" -ForegroundColor Green
Write-Host "──────────────────────────────────────────" -ForegroundColor Green
Write-Host ""
Write-Host "  Run it anytime with:  " -NoNewline
Write-Host "nexus" -ForegroundColor Yellow
Write-Host ""
Write-Host "  (Open a new terminal for PATH changes to take effect)" -ForegroundColor DarkGray
Write-Host ""
