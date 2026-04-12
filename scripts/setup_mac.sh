#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup script — macOS (Intel & Apple Silicon)
# Run once:  bash scripts/setup_mac.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Homebrew ───────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    error "Homebrew not found. Install it from https://brew.sh then re-run this script."
fi
info "Homebrew found."

# ── 2. System packages ────────────────────────────────────────────────────────
info "Installing system packages via Homebrew..."
brew install ffmpeg libsndfile 2>/dev/null || brew upgrade ffmpeg libsndfile 2>/dev/null || true

# ── 3. Python ─────────────────────────────────────────────────────────────────
PYTHON_BIN=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || echo "")
if [ -z "$PYTHON_BIN" ]; then
    info "Installing Python 3.11 via Homebrew..."
    brew install python@3.11
    PYTHON_BIN=$(command -v python3.11 || error "Python 3.11 install failed.")
fi
PYTHON_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using Python $PYTHON_VER at $PYTHON_BIN"

# ── 4. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── 5. PyTorch ────────────────────────────────────────────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon detected — installing PyTorch with MPS support..."
    # MPS backend is included in standard macOS wheels
    pip install torch torchaudio -q
else
    info "Intel Mac detected — installing CPU PyTorch..."
    pip install torch torchaudio -q
fi

# ── 6. Main dependencies ──────────────────────────────────────────────────────
info "Installing project dependencies..."
pip install -r "$REPO_ROOT/requirements.txt" -q

# ── 7. .env ───────────────────────────────────────────────────────────────────
if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    warn ".env created — edit it and set HF_TOKEN before running."
fi

mkdir -p "$REPO_ROOT/data/output" "$REPO_ROOT/data/voice_samples"

echo ""
info "Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env: set HF_TOKEN=hf_your_token_here"
echo "  2. source .venv/bin/activate"
echo "  3. python run.py"
echo ""
