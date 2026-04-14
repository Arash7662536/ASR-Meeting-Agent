"""
Transcription Module
Cuts diarized audio into per-speaker WAV chunks and transcribes each via vLLM's
OpenAI-compatible /v1/audio/transcriptions endpoint (Whisper large-v3).

vLLM must be running before this module is used.
See docs/VLLM_SETUP.md for setup instructions.
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class TranscribedChunk:
    """One transcribed speaker turn."""
    start: float
    end: float
    speaker: str
    text: str
    language: str = ""


# ── vLLM client (lazy singleton) ─────────────────────────────────────────────

_vllm_client = None


def _get_client():
    """Return (and cache) the OpenAI client pointed at vLLM."""
    global _vllm_client
    if _vllm_client is None:
        from openai import OpenAI
        _vllm_client = OpenAI(
            base_url=cfg.vllm_url,
            api_key=cfg.vllm_api_key,
        )
    return _vllm_client


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _load_mono_16k(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio file as mono float32 numpy array at native sample rate."""
    import soundfile as sf
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _resample_to_16k(data: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == 16000:
        return data
    import resampy
    return resampy.resample(data, orig_sr, 16000)


def _cut(data: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return data[int(start * sr): int(end * sr)]


# ── vLLM transcription ────────────────────────────────────────────────────────

def _transcribe_chunk(
    chunk: np.ndarray,
    language: str | None,
    chunk_save_path: Path | None = None,
) -> tuple[str, str]:
    """
    Transcribe a mono 16 kHz numpy chunk via vLLM.

    Writes the chunk to a temp WAV, POSTs it to vLLM, then deletes the temp file.
    If chunk_save_path is given the WAV is saved there instead (no deletion).

    Returns (text, detected_language).
    """
    import soundfile as sf

    # Write WAV to disk (vLLM API requires a file object)
    if chunk_save_path is not None:
        wav_path = chunk_save_path
        sf.write(str(wav_path), chunk, 16000, subtype="PCM_16")
        delete_after = False
    else:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(tmp), chunk, 16000, subtype="PCM_16")
        wav_path = tmp
        delete_after = True

    try:
        client = _get_client()
        with open(wav_path, "rb") as f:
            kwargs: dict = {
                "model": cfg.vllm_model,
                "file": f,
                "response_format": "json",
            }
            if language:
                kwargs["language"] = language

            response = client.audio.transcriptions.create(**kwargs)

        text = (response.text or "").strip()
        # vLLM may expose language in the response object
        lang = getattr(response, "language", "") or ""
        return text, lang

    except Exception as e:
        logger.warning(f"vLLM transcription request failed: {e}")
        raise
    finally:
        if delete_after:
            wav_path.unlink(missing_ok=True)


# ── Segment merging ───────────────────────────────────────────────────────────

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


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_segments(
    audio_path: str | Path,
    segments: list,
    language: str = None,
    merge_gap: float = None,
    min_duration: float = None,
    save_chunks_dir: Path = None,
    progress_callback=None,
) -> list[TranscribedChunk]:
    """
    Transcribe diarized audio segments via vLLM Whisper endpoint.

    Each merged speaker turn is:
      1. Cut from the full audio as a mono 16 kHz numpy array
      2. Written to a temp (or persistent) WAV file
      3. POSTed to vLLM /v1/audio/transcriptions
      4. Text result is collected

    Args:
        audio_path:       Full cleaned WAV (output of Demucs or extraction).
        segments:         List of diarizer.Segment.
        language:         ISO language code ('en', 'fa', …) or None for auto-detect.
        merge_gap:        Max gap (s) to merge same-speaker segments. Default: cfg value.
        min_duration:     Skip segments shorter than this. Default: cfg value.
        save_chunks_dir:  If set, WAV chunks are saved here permanently.
        progress_callback: Optional callable(current: int, total: int).

    Returns:
        List of TranscribedChunk sorted by start time.
    """
    merge_gap    = merge_gap    if merge_gap    is not None else cfg.chunk_merge_gap
    min_duration = min_duration if min_duration is not None else cfg.min_chunk_duration

    # Load + resample full audio once
    data, sr = _load_mono_16k(audio_path)
    data16 = _resample_to_16k(data, sr)

    merged = _merge_for_transcription(segments, merge_gap)
    total = len(merged)
    results: list[TranscribedChunk] = []

    if save_chunks_dir:
        save_chunks_dir = Path(save_chunks_dir)
        save_chunks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {total} segments via vLLM ({cfg.vllm_url}) ...")

    for i, seg in enumerate(merged):
        if seg.duration < min_duration:
            logger.debug(f"  Skipping short segment {seg.speaker} ({seg.duration:.2f}s)")
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        chunk = _cut(data16, 16000, seg.start, seg.end)
        if len(chunk) < 160:  # < 10 ms
            continue

        # Determine save path (permanent) vs temp (deleted after request)
        chunk_save_path = None
        if save_chunks_dir:
            chunk_save_path = save_chunks_dir / f"{seg.speaker}_{seg.start:.3f}_{seg.end:.3f}.wav"

        try:
            text, lang = _transcribe_chunk(chunk, language, chunk_save_path)
        except Exception as e:
            logger.error(
                f"  Segment {i+1}/{total} ({seg.speaker} {seg.start:.1f}s-{seg.end:.1f}s) "
                f"transcription failed: {e}"
            )
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
            logger.info(f"  {i + 1}/{total} segments done")

    logger.info(f"Transcription complete: {len(results)}/{total} segments had speech.")
    return results


# ── Formatting ────────────────────────────────────────────────────────────────

def format_transcript(
    chunks: list[TranscribedChunk],
    speaker_map: dict[str, str] = None,
) -> str:
    """Render transcript as human-readable text with optional name substitution."""
    lines = []
    for c in chunks:
        name = (speaker_map or {}).get(c.speaker, c.speaker)
        lines.append(
            f"[{c.start:07.3f}s - {c.end:07.3f}s]  {name}:\n  {c.text}\n"
        )
    return "\n".join(lines)
