"""
Speaker Diarization
Uses pyannote/speaker-diarization-3.1 to produce a timeline of who spoke when.
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single speaker turn."""
    start: float
    end: float
    speaker: str   # e.g. "SPEAKER_00"

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── Audio loader (avoids torchcodec issues) ───────────────────────────────────

def _load_audio(audio_path: Path) -> dict:
    import soundfile as sf
    import torch

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channels, time)

    if sr != 16000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    logger.info(f"Audio loaded: {waveform.shape[1] / sr:.1f}s @ {sr} Hz")
    return {"waveform": waveform, "sample_rate": sr}


# ── pyannote iterator (supports both old and new API) ─────────────────────────

def _iter_turns(diarization):
    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            yield turn.start, turn.end, speaker
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield turn.start, turn.end, speaker
    else:
        raise RuntimeError(f"Unknown diarization output type: {type(diarization)}")


# ── Public API ────────────────────────────────────────────────────────────────

def run_diarization(
    audio_path: str | Path,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    hf_token: str = None,
) -> list[Segment]:
    """
    Run pyannote speaker diarization.

    Args:
        audio_path: Path to a 16kHz mono WAV.
        num_speakers: Exact number of speakers if known.
        min_speakers / max_speakers: Hints for automatic detection.
        hf_token: HuggingFace token (falls back to HF_TOKEN env var).

    Returns:
        List of Segment sorted by start time.

    Requires:
        HF token accepted at:
          https://huggingface.co/pyannote/speaker-diarization-3.1
          https://huggingface.co/pyannote/segmentation-3.0
    """
    import torch
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError(
            "Hugging Face token required for diarization.\n"
            "Set HF_TOKEN env var or provide it in the UI."
        )

    audio_path = Path(audio_path)
    logger.info("Loading pyannote/speaker-diarization-3.1 ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )

    device = "cpu"
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            pipeline.to(torch.device("cuda"))
            device = "cuda"
    except RuntimeError:
        logger.warning("CUDA not available, falling back to CPU.")
    logger.info(f"Diarization device: {device.upper()}")

    audio_input = _load_audio(audio_path)

    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    logger.info("Running diarization (may take a few minutes) ...")
    diarization = pipeline(audio_input, **kwargs)

    segments = [
        Segment(start=s, end=e, speaker=sp)
        for s, e, sp in _iter_turns(diarization)
    ]
    segments.sort(key=lambda x: x.start)
    logger.info(f"Diarization done: {len(segments)} raw segments.")
    return segments


def speaker_durations(segments: list[Segment]) -> dict[str, float]:
    """Total speaking time (seconds) per speaker label."""
    totals: dict[str, float] = defaultdict(float)
    for seg in segments:
        totals[seg.speaker] += seg.duration
    return dict(sorted(totals.items()))


def merge_segments(segments: list[Segment], gap: float = 0.5) -> list[Segment]:
    """
    Merge consecutive turns from the same speaker when gap ≤ `gap` seconds.
    Reduces the number of chunks fed to Whisper without losing boundaries.
    """
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
