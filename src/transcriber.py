"""
Transcription Module
Cuts the diarized audio into per-speaker chunks and transcribes each one
with Whisper large-v3 (or any other Whisper model).
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscribedChunk:
    """One transcribed speaker turn."""
    start: float
    end: float
    speaker: str
    text: str
    language: str = ""


# ── Whisper model loader (cached per process) ─────────────────────────────────

_whisper_cache: dict = {}


def _load_whisper(model_name: str):
    if model_name not in _whisper_cache:
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper '{model_name}' on {device.upper()} ...")
        _whisper_cache[model_name] = whisper.load_model(model_name, device=device)
        logger.info("Whisper ready.")
    return _whisper_cache[model_name]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _load_mono_16k(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 numpy array at native sample rate."""
    import soundfile as sf
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _resample(data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    import resampy
    return resampy.resample(data, orig_sr, target_sr)


def _cut(data: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return data[int(start * sr): int(end * sr)]


# ── Merging helper ────────────────────────────────────────────────────────────

def _merge_for_transcription(segments: list, gap: float) -> list:
    """Merge consecutive same-speaker segments within `gap` seconds."""
    from diarizer import Segment
    if not segments:
        return []
    merged = [Segment(segments[0].start, segments[0].end, segments[0].speaker)]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and (seg.start - prev.end) <= gap:
            merged[-1] = Segment(prev.start, seg.end, prev.speaker)
        else:
            merged.append(Segment(seg.start, seg.end, seg.speaker))
    return merged


# ── Core transcription ────────────────────────────────────────────────────────

def transcribe_segments(
    audio_path: str | Path,
    segments: list,
    language: str = None,
    model_name: str = "large-v3",
    merge_gap: float = 1.0,
    min_duration: float = 0.3,
    save_chunks_dir: Path = None,
    progress_callback=None,
) -> list[TranscribedChunk]:
    """
    Transcribe diarized audio per speaker turn.

    Args:
        audio_path:       Path to full WAV audio.
        segments:         List of diarizer.Segment.
        language:         Whisper language code or None for auto-detect.
        model_name:       Whisper model size.
        merge_gap:        Seconds gap for merging same-speaker segments.
        min_duration:     Skip chunks shorter than this (seconds).
        save_chunks_dir:  If set, write each chunk WAV to this directory.
        progress_callback: Optional callable(current: int, total: int).

    Returns:
        List of TranscribedChunk sorted by start time.
    """
    import whisper
    import soundfile as sf

    model = _load_whisper(model_name)

    # Load and resample to 16kHz
    data, sr = _load_mono_16k(audio_path)
    data16 = _resample(data, sr, 16000)

    merged = _merge_for_transcription(segments, merge_gap)
    total = len(merged)
    results: list[TranscribedChunk] = []

    if save_chunks_dir:
        save_chunks_dir = Path(save_chunks_dir)
        save_chunks_dir.mkdir(parents=True, exist_ok=True)

    fp16 = model.device.type == "cuda"
    decode_opts = dict(
        language=language or None,
        fp16=fp16,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        verbose=False,
    )

    logger.info(f"Transcribing {total} merged segments with Whisper {model_name} ...")

    for i, seg in enumerate(merged):
        if seg.duration < min_duration:
            logger.debug(f"Skipping short segment ({seg.duration:.2f}s)")
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        chunk = _cut(data16, 16000, seg.start, seg.end)
        if len(chunk) < 160:  # < 10ms
            continue

        # Optionally persist chunk WAV
        chunk_path = None
        if save_chunks_dir:
            chunk_path = save_chunks_dir / f"{seg.speaker}_{seg.start:.3f}_{seg.end:.3f}.wav"
            sf.write(str(chunk_path), chunk, 16000)

        try:
            result = model.transcribe(chunk, **decode_opts)
            text = result["text"].strip()
            lang = result.get("language", "")
        except Exception as e:
            logger.warning(f"Whisper failed on segment {i} ({seg.speaker}): {e}")
            text, lang = "", ""

        if text:
            results.append(TranscribedChunk(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker,
                text=text,
                language=lang,
            ))

        if progress_callback:
            progress_callback(i + 1, total)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info(f"  Transcribed {i + 1}/{total}")

    logger.info(f"Transcription complete: {len(results)} segments with text.")
    return results


# ── Formatting ────────────────────────────────────────────────────────────────

def format_transcript(
    chunks: list[TranscribedChunk],
    speaker_map: dict[str, str] = None,
) -> str:
    """
    Render transcript as human-readable text.

    Args:
        chunks: Transcribed segments.
        speaker_map: Optional {original_label: display_name}.

    Returns:
        Formatted string suitable for display or export.
    """
    lines = []
    for c in chunks:
        name = (speaker_map or {}).get(c.speaker, c.speaker)
        lines.append(
            f"[{c.start:07.3f}s - {c.end:07.3f}s]  {name}:\n  {c.text}\n"
        )
    return "\n".join(lines)
