"""
Pipeline Orchestrator
Coordinates all modules and persists every artifact to the database.
"""

import logging
import time
from pathlib import Path
from typing import Callable

from config import cfg
from database import Database
from extract_voice import prepare_audio
from diarizer import run_diarization, speaker_durations, merge_segments
from transcriber import transcribe_segments, format_transcript
from speaker_id import identify_speakers

logger = logging.getLogger(__name__)

# Singleton DB — shared across the app process
_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database(cfg.db_path)
    return _db


# ── Progress reporter ─────────────────────────────────────────────────────────

class ProgressReporter:
    """Collects stage updates and forwards to an optional callback."""

    def __init__(self, callback: Callable[[str, str], None] | None = None):
        self._cb = callback
        self.log: list[str] = []

    def __call__(self, stage: str, detail: str = "") -> None:
        msg = f"[{stage}] {detail}" if detail else f"[{stage}]"
        self.log.append(msg)
        logger.info(msg)
        if self._cb:
            self._cb(stage, detail)


# ── Session output directory ──────────────────────────────────────────────────

def _session_dir(session_id: int) -> Path:
    d = cfg.output_dir / f"session_{session_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    *,
    demucs_model: str = None,
    skip_demucs: bool = False,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    hf_token: str = None,
    whisper_model: str = None,
    language: str = None,
    voice_profiles: list[dict] | None = None,
    similarity_threshold: float | None = None,
    progress_cb: Callable[[str, str], None] | None = None,
) -> int:
    """
    Run the full meeting transcription pipeline.

    Parameters
    ----------
    input_path        : Path to video or audio file.
    demucs_model      : Demucs model ('htdemucs', 'htdemucs_ft', 'mdx_extra').
    skip_demucs       : Skip vocal separation (use raw audio).
    num_speakers      : Exact speaker count hint.
    min/max_speakers  : Speaker count range hints.
    hf_token          : HuggingFace token for pyannote.
    whisper_model     : Whisper model size.
    language          : ISO language code or None for auto.
    voice_profiles    : List of {"name": str, "embedding": ndarray | None,
                                 "audio_path": str | None}.
                        Omit or pass [] to skip identification.
    similarity_threshold : Resemblyzer cosine threshold.
    progress_cb       : Optional callable(stage, detail) for UI updates.

    Returns
    -------
    session_id : int — the database session ID.
    """
    db = get_db()
    progress = ProgressReporter(progress_cb)
    t0 = time.time()

    # Resolve defaults from config
    demucs_model = demucs_model or cfg.default_demucs_model
    whisper_model = whisper_model or cfg.default_whisper_model
    threshold = similarity_threshold if similarity_threshold is not None else cfg.default_similarity_threshold
    token = hf_token or cfg.hf_token

    # ── Create session record ─────────────────────────────────────────────────
    # output_dir is unknown until we have the session_id, so pass None and update below.
    session_id = db.create_session(
        input_path=input_path,
        demucs_model=demucs_model,
        skip_demucs=skip_demucs,
        whisper_model=whisper_model,
        language=language,
        num_speakers=num_speakers,
        output_dir=None,
    )
    out_dir = _session_dir(session_id)
    db.update_session(session_id, output_dir=str(out_dir))

    try:
        # ── Step 1: Audio extraction + denoising ──────────────────────────────
        progress("extraction", f"Extracting audio (skip_demucs={skip_demucs}) ...")
        audio_path = prepare_audio(
            input_path=input_path,
            out_dir=out_dir,
            demucs_model=demucs_model,
            skip_demucs=skip_demucs,
        )
        progress("extraction", f"Clean audio: {audio_path.name}")

        # ── Step 2: Speaker diarization ───────────────────────────────────────
        progress("diarization", "Identifying speakers ...")
        raw_segs = run_diarization(
            audio_path=audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=token,
        )
        segs = merge_segments(raw_segs, gap=cfg.chunk_merge_gap)
        durations = speaker_durations(segs)
        progress("diarization", f"{len(durations)} speakers, {len(segs)} segments")

        # ── Step 3: Speaker identification (optional) ─────────────────────────
        if voice_profiles:
            progress("identification", f"Matching {len(voice_profiles)} voice profiles ...")
            speaker_map = identify_speakers(
                audio_path=audio_path,
                segments=segs,
                voice_profiles=voice_profiles,
                threshold=threshold,
            )
        else:
            speaker_map = {lbl: lbl for lbl in durations}
        progress("identification", f"Speaker map: {speaker_map}")

        # ── Persist speakers to DB ────────────────────────────────────────────
        speaker_id_map: dict[str, int] = {}
        for orig_label, display_name in speaker_map.items():
            spk_id = db.upsert_speaker(
                session_id=session_id,
                original_label=orig_label,
                display_name=display_name,
                total_duration=durations.get(orig_label, 0.0),
            )
            speaker_id_map[orig_label] = spk_id

        # ── Step 4: Transcription ─────────────────────────────────────────────
        progress("transcription", f"Transcribing with Whisper {whisper_model} ...")
        chunks_dir = out_dir / "chunks" if cfg.save_audio_chunks else None

        transcribed = transcribe_segments(
            audio_path=audio_path,
            segments=segs,
            language=language if language and language != "auto" else None,
            model_name=whisper_model,
            merge_gap=cfg.chunk_merge_gap,
            min_duration=cfg.min_chunk_duration,
            save_chunks_dir=chunks_dir,
        )

        # ── Persist chunks to DB ──────────────────────────────────────────────
        for chunk in transcribed:
            # Find matching audio chunk file (if saved)
            chunk_path = None
            if chunks_dir:
                candidate = chunks_dir / f"{chunk.speaker}_{chunk.start:.3f}_{chunk.end:.3f}.wav"
                if candidate.exists():
                    chunk_path = str(candidate)

            db.insert_chunk(
                session_id=session_id,
                speaker_id=speaker_id_map[chunk.speaker],
                start_time=chunk.start,
                end_time=chunk.end,
                chunk_audio_path=chunk_path,
                transcript=chunk.text,
                language_detected=chunk.language,
            )

        # ── Save transcript text file ─────────────────────────────────────────
        transcript_text = format_transcript(transcribed, speaker_map)
        (out_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")

        # Save timeline
        timeline_lines = [
            f"[{s.start:07.3f}s - {s.end:07.3f}s]  {speaker_map.get(s.speaker, s.speaker)}"
            for s in segs
        ]
        (out_dir / "timeline.txt").write_text("\n".join(timeline_lines), encoding="utf-8")

        # ── Finalize session ──────────────────────────────────────────────────
        elapsed = time.time() - t0
        db.update_session(
            session_id,
            status="done",
            elapsed_seconds=elapsed,
        )
        progress("done", f"Finished in {elapsed:.1f}s → session {session_id}")

    except Exception as exc:
        elapsed = time.time() - t0
        db.update_session(session_id, status="error", error_message=str(exc))
        logger.exception(f"Pipeline failed for session {session_id}")
        raise

    return session_id
