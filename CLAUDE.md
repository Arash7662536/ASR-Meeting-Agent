# CLAUDE.md — Project Guide for Claude Code

This file is read automatically at the start of every session. Update it whenever architecture or conventions change.

---

## What this repo is

**Meeting Transcription App** — a Gradio web app that takes a video or audio file (or a URL), runs it through a full AI pipeline, and returns a timestamped, speaker-attributed transcript stored in a local SQLite database.

---

## Full Pipeline

```
Input (upload or URL)
        │
        ▼
[1] Audio Extraction (ffmpeg)          src/extract_voice.py → prepare_audio()
        │
        ▼
[2] Vocal Denoising (Demucs)           src/extract_voice.py → denoise()
        │                              (optional — skip for clean audio)
        ▼
[3] Speaker Diarization (pyannote 3.1) src/diarizer.py → run_diarization()
        │                              Model is PRE-LOADED at startup
        ▼
[4] Speaker Identification             src/speaker_id.py → identify_speakers()
        │  (Resemblyzer)               Optional — only if voice profiles saved
        │                              Encoder is PRE-LOADED at startup
        ▼
[5] Transcription                      src/transcriber.py → transcribe_segments()
        │  (Whisper via vLLM)          Calls vLLM on port 8001 — NOT local
        ▼
[6] Persist to SQLite DB               src/database.py
        │
        ▼
[7] Gradio UI                          src/app.py
```

---

## Module Map

| File | Responsibility | Key public API |
|---|---|---|
| `src/config.py` | All config from `.env`. Single `cfg` singleton. | `cfg` |
| `src/database.py` | SQLite via stdlib sqlite3. WAL mode. | `Database` class |
| `src/extract_voice.py` | ffmpeg extraction + Demucs denoising | `prepare_audio()` |
| `src/diarizer.py` | pyannote diarization. `_pipeline_cache` pre-warmed at startup | `run_diarization()`, `Segment` |
| `src/speaker_id.py` | Resemblyzer identification. `_encoder_cache` pre-warmed at startup | `identify_speakers()`, `compute_embedding()` |
| `src/transcriber.py` | Cuts audio → temp WAV → POST to vLLM `/v1/audio/transcriptions` | `transcribe_segments()` |
| `src/warmup.py` | Pre-loads all models on GPU before Gradio starts. Prints errors to stdout | `run_startup_checks()` |
| `src/pipeline.py` | Orchestrates all modules, writes every artifact to DB | `run_pipeline()` → returns `session_id` |
| `src/app.py` | Gradio UI (4 tabs). URL download input. Calls pipeline | `build_ui()`, `main()` |
| `run.py` | Entry point: adds `src/` to sys.path, runs warmup, launches Gradio | — |

---

## Critical Design Decisions

### vLLM serves Whisper (NOT local)
Whisper is **not** loaded in-process. `transcriber.py` posts audio chunks to `http://localhost:8001/v1/audio/transcriptions` using the OpenAI Python client. vLLM must be running before starting the app.

- Config key: `VLLM_URL=http://localhost:8001/v1` and `VLLM_MODEL=openai/whisper-large-v3`
- See `docs/VLLM_SETUP.md` for how to start vLLM

### Models pre-loaded before Gradio starts
`run.py` calls `warmup.run_startup_checks()` before `app.launch()`. This:
1. Pings vLLM health endpoint — **fails fast if vLLM is down**
2. Loads pyannote pipeline on GPU and caches it in `diarizer._pipeline_cache`
3. Loads Resemblyzer encoder and caches it in `speaker_id._encoder_cache`

Errors are printed clearly to stdout before the Gradio public URL is shown.

### Flat imports within src/
All modules use flat imports (`from config import cfg`, `from database import Database`).
`run.py` inserts `src/` at the front of `sys.path` before any imports. Do not use relative imports.

### SQLite singleton DB
`pipeline.get_db()` returns a process-level singleton `Database` instance. All modules that need DB access import this function — never instantiate `Database` directly in modules other than `pipeline.py` and `app.py`.

### Voice samples are optional
If no voice profiles exist in the library, `identify_speakers()` returns the original `SPEAKER_XX` labels. Users can rename speakers manually in the Results tab. Resemblyzer is still loaded at warmup for fast response if profiles are added later.

### Speaker name edits persist to DB
`database.update_speaker_name()` is called from the Gradio Results tab. `database.rebuild_transcript()` re-renders the transcript using current `display_name` values from the DB. The on-disk `transcript.txt` is NOT automatically updated when names change (only regenerated on export).

---

## Environment Variables (`.env`)

| Key | Required | Default | Notes |
|---|---|---|---|
| `HF_TOKEN` | YES | — | pyannote diarization |
| `VLLM_URL` | YES | `http://localhost:8001/v1` | Whisper vLLM endpoint |
| `VLLM_MODEL` | no | `openai/whisper-large-v3` | Model name sent to vLLM |
| `VLLM_API_KEY` | no | `dummy` | vLLM doesn't enforce auth |
| `WHISPER_MODEL` | no | `large-v3` | Used in UI label only |
| `DEMUCS_MODEL` | no | `htdemucs` | |
| `HOST` | no | `0.0.0.0` | |
| `PORT` | no | `7860` | |
| `SHARE` | no | `true` | Gradio public tunnel |
| `SAVE_AUDIO_CHUNKS` | no | `true` | Save per-chunk WAVs |

---

## How to run

```bash
# 1. Start vLLM Whisper server (separate terminal)
bash docs/start_vllm.sh

# 2. Start the app
source .venv/bin/activate
python run.py
```

Warmup output before Gradio starts:
```
============================================================
STARTUP: Pre-loading models on GPU...
============================================================
[1/3] vLLM (Whisper) ...  ✓ ready (model: openai/whisper-large-v3)
[2/3] pyannote diarization ...  ✓ loaded on CUDA
[3/3] Resemblyzer encoder ...  ✓ loaded
------------------------------------------------------------
All systems ready. Launching Gradio on port 7860 ...
```

---

## Legacy files (root level)

`extract_voice_mp4.py` and `speaker_diarization.py` are the **original prototype scripts** from before the `src/` package was built. They are self-contained CLI tools, kept for reference. Do not import them.

All other root-level `.py` files (`app.py`, `pipeline.py`, etc.) are **deprecated stubs** that raise `ImportError`. They exist only because files cannot be deleted via this toolchain.

---

## Common tasks

**Add a new language to the UI dropdown**
→ Edit the `language` `gr.Dropdown` choices list in `src/app.py`.

**Change the default Whisper model**
→ Set `VLLM_MODEL=openai/whisper-large-v2` in `.env` and restart vLLM with the new model.

**Add a new pipeline step**
→ Write a new module in `src/`, add it to the import list in `src/pipeline.py`, insert it between existing steps in `run_pipeline()`, add warmup logic in `src/warmup.py` if it needs pre-loading.

**Add a new DB column**
→ Alter the DDL in `src/database.py` (`_DDL` string) and add the corresponding helper method. SQLite allows `ALTER TABLE ... ADD COLUMN` for backwards compatibility.
