"""
Audio Extraction + Denoising
Extracts audio from video and optionally separates vocals with Demucs.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


# ── ffmpeg ────────────────────────────────────────────────────────────────────

def _check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg not found. Install it:\n"
            "  Ubuntu : sudo apt install ffmpeg\n"
            "  macOS  : brew install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


def extract_audio(video_path: Path, out_dir: Path, sample_rate: int = 16000) -> Path:
    """
    Extract mono WAV at `sample_rate` from any video file.
    Returns path to the extracted WAV. Skips if output already exists.
    """
    _check_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{video_path.stem}_audio.wav"

    if wav_path.exists():
        logger.info(f"Audio already extracted: {wav_path}")
        return wav_path

    logger.info(f"Extracting audio from {video_path.name} ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{proc.stderr}")

    logger.info(f"Audio saved → {wav_path}")
    return wav_path


# ── Demucs ────────────────────────────────────────────────────────────────────

def denoise(audio_path: Path, out_dir: Path, model: str = "htdemucs") -> Path:
    """
    Separate vocals from background using Demucs.
    Returns path to vocals.wav.
    """
    logger.info(f"Running Demucs ({model}) on {audio_path.name} ...")
    demucs_out = out_dir / "demucs"
    demucs_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-o", str(demucs_out),
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{proc.stderr}")

    vocals_path = demucs_out / model / audio_path.stem / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(
            f"Demucs output not found at expected path: {vocals_path}"
        )
    logger.info(f"Vocals isolated → {vocals_path}")
    return vocals_path


# ── Public entry point ────────────────────────────────────────────────────────

def prepare_audio(
    input_path: str | Path,
    out_dir: str | Path,
    demucs_model: str = "htdemucs",
    skip_demucs: bool = False,
) -> Path:
    """
    Full extraction pipeline: video → WAV → (optionally) Demucs vocals.

    Args:
        input_path: Path to any video or audio file.
        out_dir: Working directory for intermediate/output files.
        demucs_model: Demucs model name.
        skip_demucs: If True, return raw extracted audio without denoising.

    Returns:
        Path to the clean audio WAV ready for diarization.
    """
    input_path = Path(input_path).resolve()
    out_dir = Path(out_dir).resolve()
    temp_dir = out_dir / "temp"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    # If already a WAV, use directly (no extraction needed)
    if suffix == ".wav":
        audio_path = input_path
    elif suffix in _AUDIO_EXTENSIONS:
        # Convert non-WAV audio to WAV for uniform downstream handling
        audio_path = extract_audio(input_path, temp_dir)
    elif suffix in _VIDEO_EXTENSIONS:
        audio_path = extract_audio(input_path, temp_dir)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if skip_demucs:
        return audio_path

    return denoise(audio_path, out_dir, demucs_model)
