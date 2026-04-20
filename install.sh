#!/usr/bin/env bash
# install.sh
# -----------------------------------------------------------------------------
# BehaveAI installer for Linux (bash).
#
# What it does:
#   1. Installs `uv` (Python package manager) if it isn't already on PATH.
#   2. Uses uv to install Python 3.13 (the version pinned in .python-version).
#   3. Runs `uv sync` to create a .venv and install all project dependencies
#      from pyproject.toml + uv.lock.
#   4. Checks for ffmpeg and offers to install via the detected package
#      manager if missing.
#
# Run from the repo root:
#     bash install.sh
#
# Flags:
#     --skip-ffmpeg      don't check/install ffmpeg
#     --reinstall        force re-install of uv even if present
#     --yes              non-interactive: assume yes for ffmpeg install prompt
# -----------------------------------------------------------------------------

set -euo pipefail

# ---- flag parsing ---------------------------------------------------------
SKIP_FFMPEG=0
REINSTALL=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --skip-ffmpeg) SKIP_FFMPEG=1 ;;
        --reinstall)   REINSTALL=1 ;;
        --yes|-y)      ASSUME_YES=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

# ---- pretty output --------------------------------------------------------
if [[ -t 1 ]]; then
    C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_RESET=$'\033[0m'
else
    C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_RESET=''
fi
step() { printf '%s==> %s%s\n' "$C_CYAN"   "$*" "$C_RESET"; }
ok()   { printf '%s    %s%s\n' "$C_GREEN"  "$*" "$C_RESET"; }
warn() { printf '%s    %s%s\n' "$C_YELLOW" "$*" "$C_RESET"; }
err()  { printf '%s    %s%s\n' "$C_RED"    "$*" "$C_RESET" >&2; }

# ---- move to repo root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "pyproject.toml" ]]; then
    err "pyproject.toml not found in $SCRIPT_DIR."
    err "Run this script from the BehaveAI repository root."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. uv
# ---------------------------------------------------------------------------
step "Checking for uv"

# Make sure ~/.local/bin is on PATH for this session — it's where uv lands
# by default, and the user's shell rc may not have picked it up yet.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

if command -v uv >/dev/null 2>&1 && [[ $REINSTALL -eq 0 ]]; then
    ok "uv already installed ($(uv --version))"
else
    if [[ $REINSTALL -eq 1 ]]; then
        warn "Reinstalling uv (per --reinstall)"
    else
        warn "uv not found — installing"
    fi

    # Official Astral install script.
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        err "Neither curl nor wget is available; cannot download uv."
        err "Install one of them and re-run this script."
        exit 1
    fi

    # Re-probe PATH in this shell.
    export PATH="$HOME/.local/bin:$PATH"
    hash -r

    if ! command -v uv >/dev/null 2>&1; then
        err "uv installed but not visible on PATH in this shell."
        err "Open a new terminal and re-run this script."
        exit 1
    fi
    ok "uv installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 2. Python 3.13
# ---------------------------------------------------------------------------
step "Ensuring Python 3.13 is available to uv"
if ! uv python install 3.13; then
    err "uv python install 3.13 failed."
    exit 1
fi
ok "Python 3.13 ready"

# ---------------------------------------------------------------------------
# 3. Project dependencies via uv sync
# ---------------------------------------------------------------------------
step "Installing project dependencies (uv sync)"
warn "This will download PyTorch + CUDA wheels + Ultralytics (~2-4 GB). Be patient."
if ! uv sync; then
    err "uv sync failed."
    exit 1
fi
ok "Dependencies installed into .venv"

# ---------------------------------------------------------------------------
# 4. ffmpeg (optional)
# ---------------------------------------------------------------------------
install_ffmpeg() {
    # Detect package manager and install.
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm ffmpeg
    elif command -v zypper >/dev/null 2>&1; then
        sudo zypper install -y ffmpeg
    elif command -v apk >/dev/null 2>&1; then
        sudo apk add ffmpeg
    else
        warn "No known package manager detected."
        warn "Install ffmpeg manually from https://ffmpeg.org/download.html"
        return 1
    fi
}

if [[ $SKIP_FFMPEG -eq 0 ]]; then
    step "Checking for ffmpeg"
    if command -v ffmpeg >/dev/null 2>&1; then
        ok "ffmpeg found"
    else
        warn "ffmpeg not found."
        warn "OpenCV can read most .mp4 files without it, but broader codec"
        warn "support (e.g. some .mov, .mkv variants) needs it."
        if [[ $ASSUME_YES -eq 1 ]]; then
            reply="y"
        else
            printf '%s    Install ffmpeg now via your package manager? [y/N] %s' \
                "$C_YELLOW" "$C_RESET"
            read -r reply </dev/tty || reply="n"
        fi
        if [[ "$reply" =~ ^[Yy]$ ]]; then
            if install_ffmpeg; then
                ok "ffmpeg installed"
            else
                warn "ffmpeg install did not complete; carry on without it."
            fi
        else
            warn "Skipping ffmpeg install."
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
step "Installation complete"
echo
echo "To run BehaveAI:"
echo "    uv run python app.py"
echo
echo "Or activate the venv manually:"
echo "    source .venv/bin/activate"
echo "    python app.py"
echo