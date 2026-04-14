"""
Startup Warmup — Pre-loads all GPU models before Gradio begins accepting traffic.

Called once by run.py immediately after argument parsing and before app.launch().
Results are printed to stdout so errors are visible BEFORE the public URL appears.

Model singletons are stored in the respective module's _cache variables so that
pipeline.py reuses them without re-loading.
"""

import logging
import sys
import time
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_vllm() -> Optional[str]:
    """
    Verify vLLM is up and the Whisper model is loaded.
    Returns None on success, error string on failure.
    """
    import requests

    health_url = cfg.vllm_health_url
    models_url = cfg.vllm_url.rstrip("/") + "/models"

    try:
        resp = requests.get(health_url, timeout=10)
        if resp.status_code != 200:
            return f"Health check failed: HTTP {resp.status_code} from {health_url}"
    except requests.exceptions.ConnectionError:
        return (
            f"Cannot connect to vLLM at {health_url}.\n"
            f"    Start it with:  bash docs/start_vllm.sh\n"
            f"    Then wait for 'Application startup complete' before retrying."
        )
    except requests.exceptions.Timeout:
        return f"Timeout connecting to vLLM at {health_url} (10s)."
    except Exception as e:
        return f"Unexpected error reaching vLLM: {e}"

    # List loaded models
    try:
        resp = requests.get(models_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            if cfg.vllm_model not in model_ids:
                return (
                    f"vLLM is running but '{cfg.vllm_model}' is not loaded.\n"
                    f"    Loaded models: {model_ids}\n"
                    f"    Check VLLM_MODEL in .env matches the model vLLM was started with."
                )
    except Exception:
        pass  # Model listing is best-effort

    return None


def load_pyannote() -> Optional[str]:
    """
    Load pyannote speaker-diarization pipeline onto GPU and cache it in
    diarizer._pipeline_cache so run_diarization() skips the load step.
    Returns None on success, error string on failure.
    """
    import torch
    import diarizer  # noqa: local import after sys.path is set

    token1 = cfg.hf_token
    if not token1:
        return (
            "HF_TOKEN is not set — pyannote cannot be loaded.\n"
            "    Set it in .env: HF_TOKEN=hf_your_token_here"
        )

    try:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token1,
            # use_auth_token = token
        )

        device = "cpu"
        try:
            if torch.cuda.is_available():
                torch.zeros(1).cuda()  # probe CUDA
                pipeline.to(torch.device("cuda"))
                device = "cuda"
        except RuntimeError as e:
            logger.warning(f"CUDA probe failed ({e}), using CPU for pyannote.")

        diarizer._pipeline_cache = pipeline
        return None  # success, also report device
    except Exception as e:
        return f"pyannote failed to load: {e}"


def load_resemblyzer() -> Optional[str]:
    """
    Load Resemblyzer VoiceEncoder and cache it in speaker_id._encoder_cache.
    Returns None on success, error string on failure.
    """
    import speaker_id  # noqa

    try:
        from resemblyzer import VoiceEncoder
        speaker_id._encoder_cache = VoiceEncoder()
        return None
    except Exception as e:
        return f"Resemblyzer failed to load: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class _Color:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def _ok(msg: str = "") -> str:
    return f"{_Color.GREEN}✓{_Color.RESET} {msg}"


def _fail(msg: str = "") -> str:
    return f"{_Color.RED}✗{_Color.RESET} {msg}"


def _warn(msg: str = "") -> str:
    return f"{_Color.YELLOW}!{_Color.RESET} {msg}"


def run_startup_checks(*, abort_on_vllm_failure: bool = True) -> list[str]:
    """
    Pre-load all models and verify external services.

    Prints a live status table to stdout.
    Returns a list of error messages (empty list = everything OK).

    Args:
        abort_on_vllm_failure: If True, sys.exit(1) when vLLM is unreachable.
    """
    width = 62
    print()
    print(_Color.BOLD + "=" * width + _Color.RESET)
    print(_Color.BOLD + " STARTUP: Pre-loading models on GPU..." + _Color.RESET)
    print(_Color.BOLD + "=" * width + _Color.RESET)

    errors: list[str] = []

    checks = [
        ("vLLM Whisper server",   check_vllm,        True),   # (label, fn, required)
        ("pyannote diarization",  load_pyannote,     True),
        ("Resemblyzer encoder",   load_resemblyzer,  False),   # non-fatal: works without voice profiles
    ]

    for i, (label, fn, required) in enumerate(checks, 1):
        print(f"  [{i}/{len(checks)}] {label} ...", end="", flush=True)
        t0 = time.time()
        err = fn()
        elapsed = time.time() - t0

        if err is None:
            print(f"\r  [{i}/{len(checks)}] {label}  {_ok()}  ({elapsed:.1f}s)")
        else:
            marker = _fail() if required else _warn()
            print(f"\r  [{i}/{len(checks)}] {label}  {marker}")
            # Indent the error detail
            for line in err.splitlines():
                print(f"              {line}")
            errors.append(f"[{label}] {err.splitlines()[0]}")

            if required and label.startswith("vLLM") and abort_on_vllm_failure:
                print()
                print(_Color.RED + _Color.BOLD +
                      "  FATAL: vLLM must be running before the app starts.\n"
                      "  See docs/VLLM_SETUP.md" + _Color.RESET)
                print()
                sys.exit(1)

    print("-" * width)
    if errors:
        print(_warn(f"  {len(errors)} warning(s) — check logs above."))
    else:
        print(_ok("  All systems ready."))
    print("=" * width)
    print()

    return errors
