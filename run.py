"""
Entry point — adds src/ to sys.path, runs GPU warmup checks, then launches Gradio.

Usage:
    python run.py [--host 0.0.0.0] [--port 7860] [--no-share] [--skip-warmup]
"""

import argparse
import sys
from pathlib import Path

# ── src/ must be on sys.path before any local imports ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Meeting Transcription App")
    parser.add_argument("--host",         default=None,  help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",         type=int, default=None, help="Port (default: 7860)")
    parser.add_argument("--no-share",     action="store_true", help="Disable Gradio public tunnel")
    parser.add_argument("--skip-warmup",  action="store_true",
                        help="Skip model pre-loading (not recommended for production)")
    args = parser.parse_args()

    # Apply CLI overrides to config singleton
    from config import cfg
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.no_share:
        cfg.share = False

    # ── Warmup: verify vLLM + pre-load pyannote & Resemblyzer ─────────────────
    if not args.skip_warmup:
        from warmup import run_startup_checks
        # abort_on_vllm_failure=True: exits with code 1 if vLLM is down.
        # Pyannote / Resemblyzer failures are warnings (non-fatal).
        run_startup_checks(abort_on_vllm_failure=True)
    else:
        print("[WARNING] Warmup skipped — models will load on first request (may be slow).")

    # ── Launch Gradio ──────────────────────────────────────────────────────────
    from app import main as launch
    launch()


if __name__ == "__main__":
    main()
