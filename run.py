"""
Entry point — adds src/ to sys.path then launches the Gradio app.
Usage:
    python run.py [--host 0.0.0.0] [--port 7860] [--no-share]
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Meeting Transcription App")
    parser.add_argument("--host",     default=None, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port",     type=int, default=None, help="Port (default: 7860)")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio public tunnel")
    args = parser.parse_args()

    # Override config from CLI flags
    from config import cfg
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.no_share:
        cfg.share = False

    from app import main as run_app
    run_app()


if __name__ == "__main__":
    main()
