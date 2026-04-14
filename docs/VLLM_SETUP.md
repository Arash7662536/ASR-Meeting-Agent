# vLLM Setup — Whisper large-v3 Inference Server

This guide sets up vLLM as an OpenAI-compatible Whisper inference server on port 8001.
The Meeting Transcription app sends every audio chunk to this server for transcription.

---

## Why vLLM?

- Runs Whisper on GPU with continuous batching — much faster than running locally in-process
- OpenAI-compatible `/v1/audio/transcriptions` endpoint
- Single GPU is shared between the vLLM server and the app (pyannote, Resemblyzer)
- Can be restarted independently without restarting the app

---

## Requirements

| Item | Minimum |
|---|---|
| GPU VRAM | 10 GB for large-v3 (A5000 / 3090 / A100 / H100) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| OS | Linux (Ubuntu 20.04+) |

> **Vast.ai**: Any instance with ≥24 GB VRAM (e.g. RTX 3090, A5000, A100) handles both vLLM and the app pipeline concurrently without issues.

---

## Installation

### Option A — Same virtualenv as the app

```bash
source .venv/bin/activate
pip install vllm
```

### Option B — Separate virtualenv (recommended for isolation)

```bash
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install vllm
```

> vLLM pins specific torch versions. If it conflicts with the app's torch, use Option B.

---

## First-time model download

vLLM downloads the model from Hugging Face on first launch. Set your token:

```bash
export HF_TOKEN=hf_your_token_here
```

The model (`openai/whisper-large-v3`, ~3 GB) will be cached in `~/.cache/huggingface/`.

---

## Launching the vLLM server

### Production launch (recommended)

```bash
bash docs/start_vllm.sh
```

### Manual launch

```bash
source .venv/bin/activate        # or .venv-vllm if using Option B

vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 448 \
    --gpu-memory-utilization 0.40 \
    --disable-log-requests
```

**Flag explanations:**

| Flag | Value | Why |
|---|---|---|
| `--dtype bfloat16` | bfloat16 | Half-precision — faster, less VRAM |
| `--max-model-len 448` | 448 | Whisper's encoder sequence length |
| `--gpu-memory-utilization` | 0.40 | Reserves 60% of VRAM for other models (pyannote, Resemblyzer). Increase to 0.55 if using a dedicated GPU for vLLM |
| `--disable-log-requests` | — | Cleaner logs; remove to see each request |

### Smaller/faster models

```bash
# medium — 2× faster, less accurate
vllm serve openai/whisper-medium \
    --host 0.0.0.0 --port 8001 \
    --dtype bfloat16 --max-model-len 448 \
    --gpu-memory-utilization 0.25

# large-v2 — slightly less accurate than v3 but well-tested
vllm serve openai/whisper-large-v2 \
    --host 0.0.0.0 --port 8001 \
    --dtype bfloat16 --max-model-len 448 \
    --gpu-memory-utilization 0.40
```

Update `VLLM_MODEL` in `.env` to match.

---

## Verifying the server is up

```bash
# Health check
curl http://localhost:8001/health

# List loaded models
curl http://localhost:8001/v1/models | python3 -m json.tool

# Quick transcription test (replace test.wav with any audio file)
curl http://localhost:8001/v1/audio/transcriptions \
    -H "Authorization: Bearer dummy" \
    -F "model=openai/whisper-large-v3" \
    -F "file=@test.wav" \
    -F "response_format=json"
```

Expected health response: `{"status":"ok"}`

---

## Running as a background service

### Using `nohup` (simplest)

```bash
nohup bash docs/start_vllm.sh > logs/vllm.log 2>&1 &
echo "vLLM PID: $!"
```

### Using `systemd` (production)

Create `/etc/systemd/system/vllm-whisper.service`:

```ini
[Unit]
Description=vLLM Whisper large-v3 inference server
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/Agent_1
EnvironmentFile=/path/to/Agent_1/.env
ExecStart=/path/to/Agent_1/.venv/bin/vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 448 \
    --gpu-memory-utilization 0.40 \
    --disable-log-requests
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-whisper
sudo systemctl start vllm-whisper
sudo systemctl status vllm-whisper
```

### Using `tmux` (Vast.ai / SSH sessions)

```bash
tmux new-session -d -s vllm 'bash docs/start_vllm.sh'
# Attach to view logs:
tmux attach -t vllm
```

---

## Troubleshooting

**`CUDA out of memory`**
→ Lower `--gpu-memory-utilization` (e.g. `0.30`) or use a smaller model (`whisper-medium`).

**Port 8001 already in use**
```bash
lsof -i :8001        # find who's using it
kill -9 <PID>        # kill it
```

**Model download fails / HF rate limit**
→ Ensure `HF_TOKEN` is set. Or pre-download manually:
```bash
huggingface-cli download openai/whisper-large-v3
```

**`ImportError: cannot import name ...` from vllm**
→ vLLM and the app may have conflicting torch versions. Use Option B (separate venv).

**App startup says `vLLM not reachable`**
→ vLLM hasn't finished loading yet (model download or init takes 1-3 min). Wait for
`Application startup complete` in the vLLM logs, then restart the app.

---

## Expected startup sequence

1. Start vLLM (wait until you see `Application startup complete`)
2. Start the app (`python run.py`)
3. App warmup checks vLLM → pyannote → Resemblyzer
4. Gradio public URL is printed
