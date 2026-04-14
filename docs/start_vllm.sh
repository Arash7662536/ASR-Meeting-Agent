#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start vLLM serving Whisper large-v3 on port 8001
# Usage:  bash docs/start_vllm.sh
#         nohup bash docs/start_vllm.sh > logs/vllm.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# Activate venv (prefer .venv-vllm for isolation, fall back to .venv)
if [ -d "$REPO_ROOT/.venv-vllm" ]; then
    source "$REPO_ROOT/.venv-vllm/bin/activate"
elif [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "[ERROR] No virtualenv found. Run scripts/setup_linux.sh first."
    exit 1
fi

# Config (overridable via env)
MODEL="${VLLM_MODEL:-openai/whisper-large-v3}"
PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.40}"
DTYPE="${VLLM_DTYPE:-bfloat16}"

echo "================================================================"
echo " Starting vLLM — $MODEL"
echo " Port       : $PORT"
echo " GPU util   : $GPU_UTIL"
echo " dtype      : $DTYPE"
echo "================================================================"
echo ""
echo "First run will download the model (~3 GB). This may take a few minutes."
echo ""

mkdir -p "$REPO_ROOT/logs"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len 448 \
    --gpu-memory-utilization "$GPU_UTIL" \
            
