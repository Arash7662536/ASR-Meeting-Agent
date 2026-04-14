"""
Central configuration — reads from environment variables / .env file.
All modules import from here; never read os.environ directly elsewhere.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    # ---------- Paths ----------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def voice_samples_dir(self) -> Path:
        return self.data_dir / "voice_samples"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "meeting_transcription.db"

    # ---------- Auth ----------
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN"))

    # ---------- vLLM (Whisper inference server) ----------
    # Full URL including /v1 suffix, e.g. http://localhost:8001/v1
    vllm_url: str = field(default_factory=lambda: _env("VLLM_URL", "http://localhost:8001/v1"))
    vllm_model: str = field(default_factory=lambda: _env("VLLM_MODEL", "openai/whisper-large-v3"))
    vllm_api_key: str = field(default_factory=lambda: _env("VLLM_API_KEY", "dummy"))
    # Health endpoint is derived by stripping /v1 from vllm_url
    @property
    # def vllm_health_url(self) -> str:
    #     return self.vllm_url.rstrip("/v1").rstrip("/") + "/health"
    def vllm_health_url(self) -> str:
        base = self.vllm_url.removesuffix("/v1").removesuffix("/")
        return base + "/health"
    # ---------- Gradio ----------
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 7860))
    share: bool = field(default_factory=lambda: _env_bool("SHARE", True))

    # ---------- Model defaults (UI labels / fallbacks) ----------
    default_whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "large-v3"))
    default_demucs_model: str = field(default_factory=lambda: _env("DEMUCS_MODEL", "htdemucs"))
    default_similarity_threshold: float = 0.75

    # ---------- Pipeline ----------
    save_audio_chunks: bool = field(default_factory=lambda: _env_bool("SAVE_AUDIO_CHUNKS", True))
    chunk_merge_gap: float = 1.0    # merge same-speaker segments within this gap (seconds)
    min_chunk_duration: float = 0.3  # skip segments shorter than this (seconds)

    # ---------- URL download ----------
    max_download_size_gb: float = 5.0
    download_timeout_s: int = 300   # 5 minutes for large files

    def ensure_dirs(self) -> None:
        """Create all required data directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "downloads").mkdir(parents=True, exist_ok=True)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
cfg = Config()
cfg.ensure_dirs()
