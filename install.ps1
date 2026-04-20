# install.ps1
# -----------------------------------------------------------------------------
# BehaveAI installer for Windows (PowerShell 5.1+).
#
# What it does:
#   1. Installs `uv` (Python package manager) if it isn't already on PATH.
#   2. Uses uv to install Python 3.13 (the version pinned in .python-version).
#   3. Runs `uv sync` to create a .venv and install all project dependencies
#      from pyproject.toml + uv.lock.
#   4. Checks for ffmpeg and prints install guidance if missing (optional).
#
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File .\install.ps1
#
# Flags:
#     -SkipFfmpegCheck   don't check/warn about ffmpeg
#     -Reinstall         force re-install of uv even if present
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [switch]$SkipFfmpegCheck,
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"

function Write-Step   ($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok     ($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2  ($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Write-Err2   ($msg) { Write-Host "    $msg" -ForegroundColor Red }

# Move to the directory containing this script so relative paths work regardless
# of where the user ran it from.
Set-Location -Path $PSScriptRoot

# Sanity: we should be at the repo root.
if (-not (Test-Path -Path ".\pyproject.toml")) {
    Write-Err2 "pyproject.toml not found in $PSScriptRoot."
    Write-Err2 "Run this script from the BehaveAI repository root."
    exit 1
}

# ---------------------------------------------------------------------------
# 1. uv
# ---------------------------------------------------------------------------
Write-Step "Checking for uv"

$uvPresent = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)

if ($uvPresent -and -not $Reinstall) {
    $uvVersion = (uv --version) 2>$null
    Write-Ok "uv already installed ($uvVersion)"
}
else {
    if ($Reinstall) { Write-Warn2 "Reinstalling uv (per -Reinstall)" }
    else            { Write-Warn2 "uv not found — installing" }

    # Official Astral install script. It drops uv.exe into
    # %USERPROFILE%\.local\bin and updates PATH for future sessions.
    try {
        Invoke-RestMethod -Uri "https://astral.sh/uv/install.ps1" | Invoke-Expression
    }
    catch {
        Write-Err2 "Failed to download/run the uv installer."
        Write-Err2 $_.Exception.Message
        exit 1
    }

    # Make uv available in *this* session too (the installer only updates the
    # persistent user PATH, which new shells pick up — not the current one).
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path (Join-Path $uvBin "uv.exe")) {
        $env:Path = "$uvBin;$env:Path"
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Err2 "uv was installed but is not on PATH in this shell."
        Write-Err2 "Open a new PowerShell window and re-run this script."
        exit 1
    }
    Write-Ok "uv installed: $(uv --version)"
}

# ---------------------------------------------------------------------------
# 2. Python 3.13 (uv manages it)
# ---------------------------------------------------------------------------
Write-Step "Ensuring Python 3.13 is available to uv"
try {
    uv python install 3.13 | Out-Host
    Write-Ok "Python 3.13 ready"
}
catch {
    Write-Err2 "uv python install 3.13 failed."
    Write-Err2 $_.Exception.Message
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Project dependencies via uv sync
# ---------------------------------------------------------------------------
Write-Step "Installing project dependencies (uv sync)"
Write-Warn2 "This will download PyTorch + Ultralytics (~1-2 GB). Be patient."
try {
    uv sync | Out-Host
    Write-Ok "Dependencies installed into .venv"
}
catch {
    Write-Err2 "uv sync failed."
    Write-Err2 $_.Exception.Message
    exit 1
}

# ---------------------------------------------------------------------------
# 4. ffmpeg (optional but strongly recommended for broad video codec support)
# ---------------------------------------------------------------------------
if (-not $SkipFfmpegCheck) {
    Write-Step "Checking for ffmpeg"
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Ok "ffmpeg found"
    }
    else {
        Write-Warn2 "ffmpeg not found on PATH."
        Write-Warn2 "OpenCV can read most .mp4 files without it, but some codecs"
        Write-Warn2 "require it. Install with one of:"
        Write-Warn2 "    winget install Gyan.FFmpeg"
        Write-Warn2 "    choco install ffmpeg"
        Write-Warn2 "Or download a static build from https://www.gyan.dev/ffmpeg/builds/"
        Write-Warn2 "and add its bin\ directory to PATH."
    }
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Step "Installation complete"
Write-Host ""
Write-Host "To run BehaveAI:" -ForegroundColor White
Write-Host "    uv run python app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or activate the venv manually:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    python app.py" -ForegroundColor White
Write-Host ""