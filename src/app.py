"""
Gradio UI — Meeting Transcription App

Tabs
----
1. Process       — Upload file OR paste URL, configure, run pipeline
2. Results       — View transcript, edit speaker names (persisted to DB)
3. Voice Library — Manage global named speaker profiles (optional)
4. History       — Browse past sessions
"""

import shutil
import urllib.parse
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests

from config import cfg
from database import Database
from pipeline import get_db, run_pipeline
from speaker_id import compute_embedding

# ── Helpers ───────────────────────────────────────────────────────────────────

def _db() -> Database:
    return get_db()


def _ts(unix: float) -> str:
    return datetime.fromtimestamp(unix).strftime("%Y-%m-%d %H:%M")


def _dur(seconds: float) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ── URL download ──────────────────────────────────────────────────────────────

_SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
}

_SUPPORTED_MIMETYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/flac", "audio/ogg", "audio/mp4", "audio/aac",
    "audio/x-m4a", "audio/webm",
    "video/mp4", "video/x-matroska", "video/webm",
    "video/quicktime", "video/x-msvideo", "video/mpeg",
    "application/octet-stream",  # many servers send this for any binary
}


def download_from_url(url: str) -> tuple[str | None, str]:
    """
    Download audio/video from a URL.

    Returns:
        (file_path, status_message)
        file_path is None on failure.
    """
    url = (url or "").strip()
    if not url:
        return None, "Please enter a URL."

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None, "URL must start with http:// or https://"

    max_bytes = int(cfg.max_download_size_gb * 1024 ** 3)

    # ── HEAD — check content-type & size without downloading ──────────────────
    try:
        head = requests.head(url, timeout=15, allow_redirects=True)
        content_type = head.headers.get("content-type", "").split(";")[0].strip().lower()
        content_length = int(head.headers.get("content-length", 0) or 0)

        if content_length and content_length > max_bytes:
            gb = content_length / 1024 ** 3
            return None, f"File too large ({gb:.1f} GB). Maximum is {cfg.max_download_size_gb:.0f} GB."

        path_ext = Path(parsed.path).suffix.lower()

        # Reject clearly wrong content types (skip check for octet-stream — let extension decide)
        if content_type and content_type not in _SUPPORTED_MIMETYPES:
            if path_ext not in _SUPPORTED_EXTENSIONS:
                return None, (
                    f"URL does not appear to point to an audio or video file.\n"
                    f"  Content-Type: {content_type}\n"
                    f"  URL path: {parsed.path}\n"
                    f"  Supported formats: mp4, mkv, wav, mp3, flac, ogg, m4a, …"
                )
    except requests.exceptions.Timeout:
        return None, "HEAD request timed out (15s). The URL may be slow or unreachable."
    except requests.exceptions.ConnectionError as e:
        return None, f"Cannot connect to URL: {e}"
    except Exception:
        pass  # HEAD may not be supported; proceed to GET

    # ── GET — stream download ─────────────────────────────────────────────────
    download_dir = cfg.output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    path_ext = Path(parsed.path).suffix.lower()
    suffix = path_ext if path_ext in _SUPPORTED_EXTENSIONS else ".mp4"

    # Use the filename from URL if available
    url_filename = Path(parsed.path).name
    if not url_filename or "." not in url_filename:
        url_filename = "download" + suffix
    save_path = download_dir / url_filename

    try:
        with requests.get(url, stream=True, timeout=cfg.download_timeout_s, allow_redirects=True) as r:
            r.raise_for_status()

            # Re-check content-type from GET response
            content_type = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type and content_type not in _SUPPORTED_MIMETYPES:
                path_ext = Path(parsed.path).suffix.lower()
                if path_ext not in _SUPPORTED_EXTENSIONS:
                    return None, (
                        f"Server returned unsupported content-type: '{content_type}'.\n"
                        f"  Expected audio/* or video/*."
                    )

            downloaded = 0
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):  # 2 MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        save_path.unlink(missing_ok=True)
                        return None, (
                            f"Download aborted: file exceeded {cfg.max_download_size_gb:.0f} GB limit."
                        )

        size_mb = downloaded / (1024 * 1024)
        return str(save_path), f"Downloaded {size_mb:.1f} MB  →  {url_filename}"

    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e.response.status_code} {e.response.reason}"
    except requests.exceptions.Timeout:
        return None, f"Download timed out after {cfg.download_timeout_s}s."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"
    except Exception as e:
        return None, f"Download failed: {e}"


# ── Tab 1: Process ─────────────────────────────────────────────────────────────

def process_meeting(
    # Input
    input_file,
    url_file_path,        # gr.State — path of URL-downloaded file
    # Transcription
    language,
    # Denoising
    skip_demucs,
    demucs_model,
    # Diarization
    hf_token,
    num_speakers_raw,
    min_speakers_raw,
    max_speakers_raw,
    # Speaker identification
    use_voice_library,
    similarity_threshold,
    progress=gr.Progress(track_tqdm=True),
):
    # Resolve the input — prefer URL download if available
    final_input = url_file_path or input_file
    if not final_input:
        raise gr.Error("Please upload a file or paste a valid URL and click Download.")

    num_speakers = int(num_speakers_raw) if num_speakers_raw and int(num_speakers_raw) > 0 else None
    min_speakers = int(min_speakers_raw) if min_speakers_raw and int(min_speakers_raw) > 0 else None
    max_speakers = int(max_speakers_raw) if max_speakers_raw and int(max_speakers_raw) > 0 else None
    token = hf_token.strip() if hf_token and hf_token.strip() else None

    voice_profiles = []
    if use_voice_library:
        db = _db()
        for prof in db.get_voice_profiles():
            emb = db.get_voice_profile_embedding(prof["id"])
            voice_profiles.append({"name": prof["name"], "embedding": emb, "audio_path": prof.get("audio_path")})

    log_lines: list[str] = []

    def _cb(stage: str, detail: str):
        log_lines.append(f"[{stage}] {detail}")
        progress(0, desc=f"{stage}: {detail}")

    try:
        session_id = run_pipeline(
            input_path=final_input,
            demucs_model=demucs_model,
            skip_demucs=skip_demucs,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=token,
            language=language if language != "auto" else None,
            voice_profiles=voice_profiles or None,
            similarity_threshold=similarity_threshold,
            progress_cb=_cb,
        )
        log_lines.append(f"\nSession {session_id} complete.")
        return (
            f"Done — Session ID: {session_id}",
            "\n".join(log_lines),
            str(session_id),
        )
    except Exception as exc:
        log_lines.append(f"\nERROR: {exc}")
        raise gr.Error(str(exc))


# ── Tab 2: Results ─────────────────────────────────────────────────────────────

def _session_choices() -> list[str]:
    return [
        f"[{s['id']}] {s['input_filename']} — {_ts(s['created_at'])} ({s['status']})"
        for s in _db().list_sessions()
    ]


def load_results(session_choice: str):
    if not session_choice:
        return "", "", []
    session_id = int(session_choice.split("]")[0].lstrip("["))
    db = _db()
    transcript = db.rebuild_transcript(session_id)
    speakers = db.get_speakers(session_id)
    speaker_table = [[s["id"], s["original_label"], s["display_name"], _dur(s["total_duration"])]
                     for s in speakers]
    chunks = db.get_chunks(session_id)
    timeline = "\n".join(
        f"[{c['start_time']:07.3f}s - {c['end_time']:07.3f}s]  {c['display_name']}"
        for c in chunks
    )
    return transcript, timeline, speaker_table


def save_speaker_names(session_choice: str, speaker_table):
    if not session_choice or not speaker_table:
        return "Nothing to save."
    session_id = int(session_choice.split("]")[0].lstrip("["))
    db = _db()
    for row in speaker_table:
        spk_id, _orig, new_name, _dur = row
        if new_name and str(new_name).strip():
            db.update_speaker_name(int(spk_id), str(new_name).strip())
    return db.rebuild_transcript(session_id)


def export_transcript(session_choice: str):
    if not session_choice:
        return None
    session_id = int(session_choice.split("]")[0].lstrip("["))
    db = _db()
    session = db.get_session(session_id)
    if not session or not session.get("output_dir"):
        return None
    txt_path = Path(session["output_dir"]) / "transcript.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(db.rebuild_transcript(session_id), encoding="utf-8")
    return str(txt_path)


# ── Tab 3: Voice Library ───────────────────────────────────────────────────────

def load_voice_library():
    return [[p["id"], p["name"], p.get("audio_path") or "—", _ts(p["created_at"])]
            for p in _db().get_voice_profiles()]


def add_voice_profile(name: str, audio_file):
    if not name or not name.strip():
        raise gr.Error("Please enter a speaker name.")
    name = name.strip()
    db = _db()
    audio_path = None
    embedding = None
    if audio_file:
        dst = cfg.voice_samples_dir / f"{name.replace(' ', '_')}.wav"
        shutil.copy2(audio_file, dst)
        audio_path = str(dst)
        embedding = compute_embedding(audio_path)
        if embedding is None:
            raise gr.Error("Could not compute voice embedding — clip may be too short. Use 5–30s of clear speech.")
    db.upsert_voice_profile(name=name, audio_path=audio_path, embedding=embedding)
    return f"Profile '{name}' saved.", load_voice_library()


def delete_voice_profile(profile_table, selected_index):
    if not profile_table:
        return "Nothing to delete.", load_voice_library()
    try:
        profile_id = int(profile_table[int(selected_index)][0])
        name = profile_table[int(selected_index)][1]
        _db().delete_voice_profile(profile_id)
        return f"Deleted '{name}'.", load_voice_library()
    except (IndexError, TypeError, ValueError):
        return "Select a valid row index.", load_voice_library()


# ── Tab 4: History ─────────────────────────────────────────────────────────────

def load_history():
    return [
        [s["id"], _ts(s["created_at"]), s["input_filename"], s["status"],
         s.get("whisper_model") or "—", s.get("language") or "auto",
         _dur(s.get("elapsed_seconds"))]
        for s in _db().list_sessions()
    ]


def delete_session_action(history_table, selected_index):
    if not history_table:
        return "Nothing to delete.", load_history()
    try:
        session_id = int(history_table[int(selected_index)][0])
        _db().delete_session(session_id)
        return f"Deleted session {session_id}.", load_history()
    except (IndexError, TypeError, ValueError):
        return "Select a valid row index.", load_history()


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Meeting Transcription", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# Meeting Transcription App\n"
            "**Pipeline:** Audio Extraction → Demucs denoising → "
            "Pyannote diarization → Resemblyzer identification → **Whisper via vLLM**"
        )

        current_session_id = gr.State("")
        url_file_path      = gr.State(None)   # path of URL-downloaded file

        # ════════════════════════════════════════════════════════════════════
        # Tab 1 — Process
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Process"):
            with gr.Row():

                # ── Left: inputs ──────────────────────────────────────────
                with gr.Column(scale=1):

                    # ── Input source ──────────────────────────────────────
                    with gr.Tabs():
                        with gr.Tab("Upload File"):
                            input_file = gr.File(
                                label="Video or Audio File",
                                file_types=[
                                    ".mp4", ".mkv", ".avi", ".mov", ".webm",
                                    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
                                ],
                                type="filepath",
                            )

                        with gr.Tab("From URL"):
                            gr.Markdown(
                                "Paste a **direct download link** to a video or audio file. "
                                "The app will download it before processing."
                            )
                            url_input      = gr.Textbox(
                                label="Download URL",
                                placeholder="https://example.com/meeting.mp4",
                            )
                            download_btn   = gr.Button("Download from URL", variant="secondary")
                            download_status = gr.Textbox(
                                label="Download status",
                                lines=2,
                                interactive=False,
                            )

                    # ── Transcription settings ────────────────────────────
                    with gr.Accordion("Transcription", open=True):
                        language = gr.Dropdown(
                            label="Language",
                            choices=["auto", "en", "fa", "de", "fr", "es", "it",
                                     "pt", "nl", "ru", "zh", "ja", "ko", "ar",
                                     "tr", "pl", "uk", "cs", "sv", "da", "fi",
                                     "hu", "ro", "bg", "he", "hi", "id", "vi"],
                            value="auto",
                        )
                        gr.Markdown(
                            f"*Whisper model in use: **{cfg.vllm_model}** served by vLLM at `{cfg.vllm_url}`*"
                        )

                    # ── Denoising ─────────────────────────────────────────
                    with gr.Accordion("Denoising (Demucs)", open=False):
                        skip_demucs = gr.Checkbox(
                            label="Skip Demucs — audio already clean",
                            value=False,
                        )
                        demucs_model = gr.Dropdown(
                            label="Demucs Model",
                            choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                            value=cfg.default_demucs_model,
                        )

                    # ── Diarization ───────────────────────────────────────
                    with gr.Accordion("Diarization (pyannote)", open=False):
                        hf_token = gr.Textbox(
                            label="Hugging Face Token",
                            placeholder="hf_… (or set HF_TOKEN in .env)",
                            type="password",
                        )
                        num_speakers = gr.Number(label="Exact speakers (0 = auto)", value=0, precision=0)
                        with gr.Row():
                            min_speakers = gr.Number(label="Min speakers", value=0, precision=0)
                            max_speakers = gr.Number(label="Max speakers", value=0, precision=0)

                    # ── Speaker identification ────────────────────────────
                    with gr.Accordion("Speaker Identification (optional)", open=False):
                        use_voice_library = gr.Checkbox(
                            label="Match against Voice Library",
                            value=False,
                            info="Manage speaker profiles in the 'Voice Library' tab.",
                        )
                        similarity_threshold = gr.Slider(
                            label="Match Threshold",
                            minimum=0.5, maximum=0.95,
                            value=cfg.default_similarity_threshold,
                            step=0.05,
                        )

                    run_btn = gr.Button("Process Meeting", variant="primary", size="lg")

                # ── Right: outputs ────────────────────────────────────────
                with gr.Column(scale=1):
                    process_status = gr.Textbox(label="Status", lines=2, interactive=False)
                    process_logs   = gr.Textbox(label="Logs", lines=22, interactive=False)

            # Wire download
            download_btn.click(
                fn=download_from_url,
                inputs=[url_input],
                outputs=[url_file_path, download_status],
            )

            # Wire process
            run_btn.click(
                fn=process_meeting,
                inputs=[
                    input_file, url_file_path,
                    language,
                    skip_demucs, demucs_model,
                    hf_token, num_speakers, min_speakers, max_speakers,
                    use_voice_library, similarity_threshold,
                ],
                outputs=[process_status, process_logs, current_session_id],
            )

        # ════════════════════════════════════════════════════════════════════
        # Tab 2 — Results
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Results"):
            with gr.Row():
                session_dd    = gr.Dropdown(label="Select Session", choices=_session_choices(), scale=4)
                refresh_btn   = gr.Button("Refresh", scale=1)
                export_btn    = gr.Button("Export Transcript", scale=1)

            export_file = gr.File(label="Download Transcript", visible=False)

            with gr.Row():
                with gr.Column(scale=2):
                    transcript_box = gr.Textbox(
                        label="Transcript",
                        lines=28,
                        show_copy_button=True,
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    timeline_box = gr.Textbox(
                        label="Speaker Timeline",
                        lines=20,
                        interactive=False,
                    )
                    gr.Markdown(
                        "### Edit Speaker Names\n"
                        "Double-click a **Display Name** cell, then click **Save Names**."
                    )
                    speaker_table = gr.Dataframe(
                        headers=["ID", "Original Label", "Display Name", "Duration"],
                        datatype=["number", "str", "str", "str"],
                        col_count=(4, "fixed"),
                        interactive=True,
                        label="Speakers",
                        wrap=True,
                    )
                    save_names_btn = gr.Button("Save Names", variant="primary")

            session_dd.change(
                fn=load_results,
                inputs=[session_dd],
                outputs=[transcript_box, timeline_box, speaker_table],
            )
            refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=_session_choices()),
                inputs=[],
                outputs=[session_dd],
            )
            save_names_btn.click(
                fn=save_speaker_names,
                inputs=[session_dd, speaker_table],
                outputs=[transcript_box],
            )
            export_btn.click(
                fn=lambda s: gr.File(value=export_transcript(s), visible=True),
                inputs=[session_dd],
                outputs=[export_file],
            )

        # ════════════════════════════════════════════════════════════════════
        # Tab 3 — Voice Library
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Voice Library"):
            gr.Markdown(
                "Save named voice samples here. Enable **'Match against Voice Library'** "
                "in the Process tab to auto-identify speakers.\n\n"
                "**Voice samples are optional** — you can always rename speakers manually in Results."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Add / Update Profile")
                    new_name  = gr.Textbox(label="Speaker Name", placeholder="e.g. Alice")
                    new_audio = gr.Audio(
                        label="Voice Sample (5–30s of clear speech, optional)",
                        type="filepath",
                    )
                    add_btn    = gr.Button("Save Profile", variant="primary")
                    add_status = gr.Textbox(label="Status", lines=1, interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### Saved Profiles")
                    profile_table = gr.Dataframe(
                        headers=["ID", "Name", "Audio Path", "Created"],
                        datatype=["number", "str", "str", "str"],
                        col_count=(4, "fixed"),
                        interactive=False,
                    )
                    with gr.Row():
                        del_idx = gr.Number(label="Row index to delete (0-based)", value=0, precision=0)
                        del_btn = gr.Button("Delete", variant="stop")
                    del_status = gr.Textbox(label="Status", lines=1, interactive=False)

            demo.load(fn=load_voice_library, inputs=[], outputs=[profile_table])
            add_btn.click(
                fn=add_voice_profile,
                inputs=[new_name, new_audio],
                outputs=[add_status, profile_table],
            )
            del_btn.click(
                fn=delete_voice_profile,
                inputs=[profile_table, del_idx],
                outputs=[del_status, profile_table],
            )

        # ════════════════════════════════════════════════════════════════════
        # Tab 4 — History
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("History"):
            refresh_hist_btn = gr.Button("Refresh")
            history_table = gr.Dataframe(
                headers=["ID", "Date", "File", "Status", "Model", "Language", "Duration"],
                datatype=["number", "str", "str", "str", "str", "str", "str"],
                col_count=(7, "fixed"),
                interactive=False,
            )
            with gr.Row():
                del_sess_idx = gr.Number(label="Row index to delete (0-based)", value=0, precision=0)
                del_sess_btn = gr.Button("Delete Session", variant="stop")
            hist_status = gr.Textbox(label="Status", lines=1, interactive=False)

            demo.load(fn=load_history, inputs=[], outputs=[history_table])
            refresh_hist_btn.click(fn=load_history, inputs=[], outputs=[history_table])
            del_sess_btn.click(
                fn=delete_session_action,
                inputs=[history_table, del_sess_idx],
                outputs=[hist_status, history_table],
            )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = build_ui()
    app.launch(
        server_name=cfg.host,
        server_port=cfg.port,
        share=cfg.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
