#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup script — Linux (Ubuntu/Debian)
# Run once on your server:  bash scripts/setup_linux.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
PY_MIN="3.10"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. System packages ────────────────────────────────────────────────────────
info "Updating apt and installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    python3-pip \
    python3-venv \
    libsndfile1 \
    libgomp1

# ── 2. Python version check ───────────────────────────────────────────────────
PYTHON_BIN=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || echo "")
if [ -z "$PYTHON_BIN" ]; then
    warn "Python >= ${PY_MIN} not found via python3.1x. Trying python3..."
    PYTHON_BIN=$(command -v python3 || error "Python 3 not found. Install python3.10+")
fi
PYTHON_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using Python $PYTHON_VER at $PYTHON_BIN"

# ── 3. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    info "Virtual environment already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── 4. PyTorch (CUDA auto-detect) ─────────────────────────────────────────────
info "Detecting CUDA..."
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || echo "")
    if [ -n "$NVCC_VER" ]; then
        # Map CUDA version to PyTorch index
        MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
        MINOR=$(echo "$NVCC_VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 12 ] && [ "$MINOR" -ge 4 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        elif [ "$MAJOR" -ge 12 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        elif [ "$MAJOR" -eq 11 ] && [ "$MINOR" -ge 8 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        fi
        info "CUDA $NVCC_VER detected — installing GPU PyTorch from $TORCH_INDEX"
        pip install torch torchaudio --index-url "$TORCH_INDEX" -q
    else
        warn "nvidia-smi found but nvcc missing. Installing CUDA 12.1 PyTorch..."
        pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cu121" -q
    fi
else
    warn "No GPU detected — installing CPU-only PyTorch."
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/cpu" -q
fi

# ── 5. Main dependencies ──────────────────────────────────────────────────────
info "Installing project dependencies..."
pip install -r "$REPO_ROOT/requirements.txt" -q

# ── 6. .env file ──────────────────────────────────────────────────────────────
if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    warn ".env created from .env.example — EDIT IT and set HF_TOKEN before running."
else
    info ".env already exists."
fi

# ── 7. Data directories ───────────────────────────────────────────────────────
mkdir -p "$REPO_ROOT/data/output" "$REPO_ROOT/data/voice_samples"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env and set HF_TOKEN=hf_your_token_here"
echo "  2. Activate the venv:  source .venv/bin/activate"
echo "  3. Launch the app:     python run.py"
echo ""
