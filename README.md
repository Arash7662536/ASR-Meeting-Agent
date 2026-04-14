# Meeting Transcription App

Production-grade meeting transcription with automatic speaker detection, optional voice-based speaker naming, and a Gradio web UI backed by a persistent SQLite database.

Whisper is served via **vLLM** as a separate GPU inference server. All models are pre-loaded on GPU and verified before the Gradio public URL is printed.

---

## Pipeline

```
Input: Upload file  OR  Paste URL (app downloads it)
             │
             ▼
[1] Audio Extraction (ffmpeg)          mono 16 kHz WAV
             │
             ▼
[2] Vocal Denoising (Demucs)           optional — skip for clean audio
             │  clean vocals WAV
             ▼
[3] Speaker Diarization (pyannote 3.1) pre-loaded on GPU at startup
             │  SPEAKER_00, SPEAKER_01, …
             ▼
[4] Speaker Identification (Resemblyzer)  optional — only if voice profiles saved
             │  SPEAKER_00 → "Alice", SPEAKER_01 → "Bob", …
             ▼
[5] Transcription (Whisper via vLLM)   HTTP POST to localhost:8001
             │  per-speaker text
             ▼
[6] SQLite DB  +  Gradio UI
             results stored, speaker names editable in browser
```

---

## Architecture overview

| Component | Technology | Notes |
|---|---|---|
| Voice extraction | ffmpeg + Demucs | Separates vocals from background |
| Speaker diarization | pyannote 3.1 | Pre-loaded at startup |
| Speaker identification | Resemblyzer | Optional; pre-loaded at startup |
| Transcription | Whisper via **vLLM** | Separate GPU server on port 8001 |
| Persistence | SQLite (WAL mode) | Single file, no server needed |
| UI | Gradio 4.x | 4 tabs, public link via Gradio tunnel |

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 or 3.12 recommended |
| CUDA GPU ≥ 10 GB VRAM | Needed for Whisper large-v3; pyannote and Resemblyzer share the rest |
| ffmpeg | Must be on `PATH` |
| Hugging Face token | Free — required for pyannote diarization |

> **Vast.ai**: Any instance with ≥24 GB VRAM (RTX 3090, A5000, A100, H100) runs everything comfortably.

### Hugging Face token setup (one-time)

1. Create a free account at <https://huggingface.co>
2. Generate a token at <https://huggingface.co/settings/tokens> (Read access)
3. Accept model conditions at:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
4. Paste the token into `.env`

---

## Installation

### Linux / Vast.ai server (recommended)

```bash
git clone <your-repo-url>
cd Agent_1
bash scripts/setup_linux.sh
```

The script auto-detects your CUDA version and installs the matching PyTorch wheel.

### macOS

```bash
bash scripts/setup_mac.sh
```

Requires [Homebrew](https://brew.sh). Uses MPS backend on Apple Silicon.

### Windows

```bat
scripts\setup_windows.bat
```

Requires Python 3.10+ and ffmpeg already on `PATH`.

### Manual (any platform)

```bash
# 1. Create venv
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install PyTorch with CUDA (adjust cu121 for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install app dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — set HF_TOKEN and verify VLLM_URL
```

---

## vLLM setup (Whisper inference server)

vLLM serves Whisper as an OpenAI-compatible API. It **must be running** before the app starts.

See **[docs/VLLM_SETUP.md](docs/VLLM_SETUP.md)** for the full guide.

### Quick start

```bash
# Install vLLM (same or separate venv — see VLLM_SETUP.md)
pip install vllm

# Launch (reads VLLM_MODEL, VLLM_PORT, VLLM_GPU_UTIL from .env)
bash docs/start_vllm.sh
```

Wait until you see `Application startup complete` in the vLLM output before starting the app.

### GPU memory split (example — 24 GB card)

| Process | VRAM |
|---|---|
| vLLM / Whisper large-v3 | ~9 GB (VLLM_GPU_UTIL=0.40) |
| pyannote diarization | ~4 GB |
| Resemblyzer | ~0.5 GB |
| Buffer | ~10 GB |

Adjust `VLLM_GPU_UTIL` in `.env` for your card.

---

## Configuration

Copy `.env.example` to `.env` and fill in your values.

```ini
# ── Required ──────────────────────────────────────────────────────────────────
HF_TOKEN=hf_your_token_here

# ── vLLM (Whisper server) ─────────────────────────────────────────────────────
VLLM_URL=http://localhost:8001/v1       # Must match --port in start_vllm.sh
VLLM_MODEL=openai/whisper-large-v3     # Must match the model vLLM was started with
VLLM_GPU_UTIL=0.40                     # Fraction of GPU VRAM for vLLM

# ── Optional ──────────────────────────────────────────────────────────────────
PORT=7860
SHARE=true                             # false = no public Gradio tunnel
DEMUCS_MODEL=htdemucs
SAVE_AUDIO_CHUNKS=true
```

Full variable reference in [.env.example](.env.example).

---

## Running

```bash
# Terminal 1 — start vLLM (keep running)
bash docs/start_vllm.sh

# Terminal 2 — start the app
source .venv/bin/activate
python run.py
```

The app prints a startup check table **before** the Gradio URL appears:

```
══════════════════════════════════════════════════════════════
 STARTUP: Pre-loading models on GPU...
══════════════════════════════════════════════════════════════
  [1/3] vLLM Whisper server ...  ✓  (0.3s)
  [2/3] pyannote diarization ...  ✓  (12.4s)
  [3/3] Resemblyzer encoder ...   ✓  (1.1s)
──────────────────────────────────────────────────────────────
✓  All systems ready.
══════════════════════════════════════════════════════════════

Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxx.gradio.live
```

If vLLM is not running, the app exits with a clear error message before any URL is printed.

### CLI flags

```bash
python run.py --host 0.0.0.0 --port 8080 --no-share --skip-warmup
```

| Flag | Effect |
|---|---|
| `--host` | Override bind address |
| `--port` | Override port |
| `--no-share` | Disable Gradio public tunnel |
| `--skip-warmup` | Skip model pre-loading (dev only — models load on first request) |

---

## UI Tabs

### 1. Process

**Two input methods:**

| Method | How |
|---|---|
| Upload File | Drag & drop or click — mp4, mkv, wav, mp3, flac, ogg, m4a, … |
| From URL | Paste a direct download link → click **Download from URL** |

URL validation before download:
- Must be `http://` or `https://`
- HEAD request checks `Content-Type` — rejects non-audio/video responses
- Size limit enforced (default 5 GB)
- Clear error shown for invalid URLs, wrong file type, HTTP errors, timeouts

**Settings:**

| Setting | Default | Notes |
|---|---|---|
| Language | auto | ISO code (en, fa, de, fr, …) or auto-detect |
| Skip Demucs | off | Enable for already-clean audio — saves time |
| Demucs Model | htdemucs | htdemucs_ft or mdx_extra for higher quality |
| HF Token | from `.env` | Override per-session |
| Exact speakers | 0 (auto) | Hint — speeds up diarization |
| Use Voice Library | off | Auto-match speakers against saved profiles |

Click **Process Meeting** — pipeline logs stream in real time.

### 2. Results

- Select any past session from the dropdown
- **Edit speaker names** — double-click any cell in the Display Name column
- Click **Save Names** — transcript re-renders instantly using the new names
- **Export Transcript** — downloads the final `.txt` file

### 3. Voice Library

Save named voice samples for automatic speaker identification.

- Upload a 5–30 s audio clip of each known speaker
- Resemblyzer embeddings are stored in the database — no re-processing on reuse
- **Voice samples are completely optional** — the pipeline works without them; rename speakers manually in the Results tab instead
- Enable **"Match against Voice Library"** in the Process tab to use saved profiles

### 4. History

- Browse all past sessions (date, file, status, model, duration)
- Delete sessions you no longer need

---

## Project structure

```
.
├── src/
│   ├── config.py          # Central config — reads .env, exposes cfg singleton
│   ├── database.py        # SQLite (sessions / speakers / chunks / voice_profiles)
│   ├── extract_voice.py   # ffmpeg audio extraction + Demucs vocal separation
│   ├── diarizer.py        # pyannote 3.1 diarization — pipeline cached at startup
│   ├── speaker_id.py      # Resemblyzer identification — encoder cached at startup
│   ├── transcriber.py     # Cuts audio → temp WAV → POST to vLLM API
│   ├── warmup.py          # Pre-loads all models before Gradio starts
│   ├── pipeline.py        # Orchestrator — wires all modules, writes to DB
│   └── app.py             # Gradio UI (4 tabs + URL downloader)
├── docs/
│   ├── VLLM_SETUP.md      # Full vLLM setup guide
│   └── start_vllm.sh      # Production vLLM launcher script
├── scripts/
│   ├── setup_linux.sh     # Auto-detects CUDA, creates venv, installs deps
│   ├── setup_mac.sh       # Homebrew + MPS support
│   └── setup_windows.bat  # GPU/CPU detection, creates venv
├── data/                  # Created at runtime (gitignored)
│   ├── meeting_transcription.db
│   ├── output/
│   │   ├── downloads/         # Files downloaded from URLs
│   │   └── session_N/
│   │       ├── transcript.txt
│   │       ├── timeline.txt
│   │       └── chunks/        # Per-speaker WAVs (SAVE_AUDIO_CHUNKS=true)
│   └── voice_samples/
├── run.py                 # Entry point
├── CLAUDE.md              # Architecture guide for Claude Code
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

> **Legacy files** — `extract_voice_mp4.py` and `speaker_diarization.py` are the original prototype scripts, kept for reference. All other root-level `.py` files (`app.py`, `pipeline.py`, etc.) are deprecated stubs that raise `ImportError`.

---

## Database schema

| Table | Purpose |
|---|---|
| `sessions` | One row per pipeline run — status, models used, elapsed time |
| `speakers` | Per-session speaker rows with editable `display_name` |
| `audio_chunks` | Per-segment rows — `start_time`, `end_time`, `transcript`, chunk WAV path |
| `voice_profiles` | Global speaker library — name + Resemblyzer embedding blob |

The database is a single file: `data/meeting_transcription.db`.
Back it up to preserve speaker name edits and voice profiles across reinstalls.

---

## Troubleshooting

**`vLLM not reachable` on startup**
→ vLLM hasn't finished loading yet (model download or init takes 1–3 min on first run).
Wait for `Application startup complete` in the vLLM terminal, then re-run the app.

**`HF_TOKEN` error on diarization**
→ Set `HF_TOKEN` in `.env` and accept both model conditions on Hugging Face (links in Requirements section above).

**CUDA out of memory**
→ Lower `VLLM_GPU_UTIL` in `.env` (e.g. `0.30`), or switch to a smaller model:
`VLLM_MODEL=openai/whisper-medium` and restart vLLM.

**Demucs OOM**
→ Use `htdemucs` (default). If still OOM, check "Skip Demucs" for already-clean recordings.

**URL download rejected — wrong content-type**
→ The server is returning a non-audio/video `Content-Type`. Try saving the file locally and uploading it instead.

**`torchcodec` / `torchaudio` import errors**
→ The diarizer uses `soundfile` to load audio, bypassing `torchcodec`. Ensure `soundfile` is installed: `pip install soundfile`.

**Resemblyzer not matching speakers**
→ Lower the threshold slider (default 0.75). Use a longer, cleaner sample (15–30 s, no background noise, no music).

**vLLM and app torch version conflict**
→ Install vLLM in a separate virtualenv (`.venv-vllm`). See [docs/VLLM_SETUP.md](docs/VLLM_SETUP.md) — Option B.

---

## License

MIT
