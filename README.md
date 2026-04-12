# Meeting Transcription App

Production-grade meeting transcription with automatic speaker detection, optional voice-based speaker naming, and a Gradio web UI backed by a persistent SQLite database.

---

## Pipeline

```
Video / Audio
     │
     ▼
[1] Audio Extraction (ffmpeg)
     │  mono 16 kHz WAV
     ▼
[2] Vocal Denoising (Demucs)        ← optional, disable with "Skip Demucs"
     │  clean vocals WAV
     ▼
[3] Speaker Diarization (pyannote 3.1)
     │  who spoke when → SPEAKER_00, SPEAKER_01, …
     ▼
[4] Speaker Identification (Resemblyzer)   ← optional, needs voice samples
     │  SPEAKER_00 → "Alice", SPEAKER_01 → "Bob", …
     ▼
[5] Transcription (Whisper large-v3)
     │  per-speaker text
     ▼
[6] SQLite Database  +  Gradio UI
     results stored, speaker names editable in browser
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.11 or 3.12 recommended |
| ffmpeg | Must be on `PATH` |
| CUDA GPU | Optional but strongly recommended for large-v3 |
| Hugging Face token | Free — required for pyannote diarization |

### Hugging Face token setup (one-time)

1. Create a free account at <https://huggingface.co>
2. Generate a token at <https://huggingface.co/settings/tokens> (Read access)
3. Accept model conditions at:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
4. Paste the token into `.env` (see below)

---

## Installation

### Linux / Vast.ai server (recommended)

```bash
git clone <your-repo-url>
cd meeting-transcription
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install PyTorch (adjust CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining deps
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and set HF_TOKEN
```

---

## Configuration

All settings live in `.env`:

```ini
HF_TOKEN=hf_your_token_here   # Required

# Optional overrides
WHISPER_MODEL=large-v3         # large-v3 | large-v2 | medium | small | base | tiny
DEMUCS_MODEL=htdemucs          # htdemucs | htdemucs_ft | mdx_extra
PORT=7860
SHARE=true                     # false = no public Gradio tunnel
SAVE_AUDIO_CHUNKS=true         # saves per-chunk WAV to data/output/session_N/chunks/
```

---

## Running

```bash
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python run.py
```

Additional CLI flags:

```
python run.py --host 0.0.0.0 --port 8080 --no-share
```

Open the printed URL in your browser. With `SHARE=true` a public `gradio.live` link is also printed — useful for Vast.ai.

---

## UI Tabs

### 1. Process
Upload a video or audio file and configure the pipeline:

| Setting | Default | Notes |
|---|---|---|
| Language | auto | ISO code (en, fa, de, …) or auto-detect |
| Whisper Model | large-v3 | Smaller models are faster but less accurate |
| Skip Demucs | off | Enable if your audio is already clean |
| HF Token | — | Overrides `.env` value per session |
| Exact speakers | 0 (auto) | Speeds up diarization if you know the count |
| Use Voice Library | off | Match against saved speaker profiles |

Click **Process Meeting** — logs stream in real time.

### 2. Results
- Select any past session from the dropdown
- **Edit speaker names** directly in the table (double-click a cell)
- Click **Save Names** — transcript re-renders instantly with the new names
- Download the final transcript with **Export Transcript**

### 3. Voice Library
- Save named voice samples (5–30 s clips) for automatic speaker identification
- Voice samples are **optional** — the app works without them; you can always rename speakers manually in Results
- Embeddings are stored in the database (no re-processing on re-use)

### 4. History
- Browse all past sessions with status, model, and processing time
- Delete sessions you no longer need

---

## Project Structure

```
.
├── src/
│   ├── config.py          # Central config (reads .env)
│   ├── database.py        # SQLite layer (sessions, speakers, chunks, voice profiles)
│   ├── extract_voice.py   # ffmpeg extraction + Demucs denoising
│   ├── diarizer.py        # pyannote speaker diarization
│   ├── speaker_id.py      # Resemblyzer speaker identification
│   ├── transcriber.py     # Whisper per-chunk transcription
│   ├── pipeline.py        # Orchestrator — wires all modules + writes to DB
│   └── app.py             # Gradio UI
├── scripts/
│   ├── setup_linux.sh
│   ├── setup_mac.sh
│   └── setup_windows.bat
├── data/                  # Created at runtime (gitignored)
│   ├── meeting_transcription.db
│   ├── output/
│   │   └── session_N/
│   │       ├── transcript.txt
│   │       ├── timeline.txt
│   │       └── chunks/        # per-speaker WAV chunks (if SAVE_AUDIO_CHUNKS=true)
│   └── voice_samples/
├── run.py                 # Entry point
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

> **Legacy files** (`extract_voice_mp4.py`, `speaker_diarization.py`) in the repo root are the original prototypes. The production code is entirely under `src/`.

---

## Database Schema

| Table | Purpose |
|---|---|
| `sessions` | One row per pipeline run (status, models used, elapsed time) |
| `speakers` | Per-session speaker rows with editable `display_name` |
| `audio_chunks` | Per-segment rows with `start_time`, `end_time`, `transcript`, path to chunk WAV |
| `voice_profiles` | Global speaker library — name + stored Resemblyzer embedding blob |

The database is a single file at `data/meeting_transcription.db`. Back it up to keep your speaker name edits and voice profiles.

---

## Troubleshooting

**`HF_TOKEN` error on diarization**
→ Set `HF_TOKEN` in `.env` and accept model conditions on Hugging Face (links above).

**Demucs OOM on GPU**
→ Use `htdemucs` (default). If still OOM, enable "Skip Demucs" for clean recordings.

**Whisper slow on CPU**
→ Switch model to `medium` or `small` in the UI. `large-v3` on CPU is very slow.

**`torchcodec` / `torchaudio` import errors**
→ The diarizer uses `soundfile` to load audio, bypassing `torchcodec` entirely. Ensure `soundfile` is installed (`pip install soundfile`).

**`resemblyzer` not matching speakers**
→ Lower the threshold slider (default 0.75). Try providing a longer, cleaner voice sample (15–30 s of uninterrupted speech with no background noise).

---

## License

MIT
