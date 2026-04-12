"""
Gradio UI — Meeting Transcription App

Tabs
----
1. Process     — Upload file, configure, run pipeline
2. Results     — View transcript, edit speaker names (persisted to DB)
3. Voice Library — Manage global named speaker profiles (optional)
4. History     — Browse past sessions
"""

import shutil
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

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


# ── Tab 1: Process ────────────────────────────────────────────────────────────

def process_meeting(
    input_file,
    language,
    whisper_model,
    skip_demucs,
    demucs_model,
    num_speakers_raw,
    min_speakers_raw,
    max_speakers_raw,
    hf_token,
    use_voice_library,
    similarity_threshold,
    progress=gr.Progress(track_tqdm=True),
):
    if input_file is None:
        raise gr.Error("Please upload a video or audio file.")

    num_speakers = int(num_speakers_raw) if num_speakers_raw and int(num_speakers_raw) > 0 else None
    min_speakers = int(min_speakers_raw) if min_speakers_raw and int(min_speakers_raw) > 0 else None
    max_speakers = int(max_speakers_raw) if max_speakers_raw and int(max_speakers_raw) > 0 else None
    token = hf_token.strip() if hf_token and hf_token.strip() else None

    # Gather voice profiles from the library (optional)
    voice_profiles = []
    if use_voice_library:
        db = _db()
        for prof in db.get_voice_profiles():
            emb = db.get_voice_profile_embedding(prof["id"])
            voice_profiles.append({
                "name": prof["name"],
                "embedding": emb,
                "audio_path": prof.get("audio_path"),
            })

    log_lines: list[str] = []

    def _cb(stage: str, detail: str):
        log_lines.append(f"[{stage}] {detail}")
        progress(0, desc=f"{stage}: {detail}")

    try:
        session_id = run_pipeline(
            input_path=input_file,
            demucs_model=demucs_model,
            skip_demucs=skip_demucs,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=token,
            whisper_model=whisper_model,
            language=language if language != "auto" else None,
            voice_profiles=voice_profiles or None,
            similarity_threshold=similarity_threshold,
            progress_cb=_cb,
        )
        log_lines.append(f"\nSession ID: {session_id} — Done!")
        return (
            f"Processing complete. Session ID: {session_id}",
            "\n".join(log_lines),
            str(session_id),
        )

    except Exception as exc:
        log_lines.append(f"\nERROR: {exc}")
        raise gr.Error(str(exc))


# ── Tab 2: Results ────────────────────────────────────────────────────────────

def _load_session_list() -> list[str]:
    sessions = _db().list_sessions()
    return [f"[{s['id']}] {s['input_filename']} — {_ts(s['created_at'])} ({s['status']})"
            for s in sessions]


def load_results(session_choice: str):
    if not session_choice:
        return "", "", []

    session_id = int(session_choice.split("]")[0].lstrip("["))
    db = _db()
    transcript = db.rebuild_transcript(session_id)

    speakers = db.get_speakers(session_id)
    speaker_table = [[s["id"], s["original_label"], s["display_name"],
                      _dur(s["total_duration"])] for s in speakers]

    chunks = db.get_chunks(session_id)
    timeline_lines = [
        f"[{c['start_time']:07.3f}s - {c['end_time']:07.3f}s]  {c['display_name']}"
        for c in chunks
    ]
    timeline = "\n".join(timeline_lines)

    return transcript, timeline, speaker_table


def save_speaker_names(session_choice: str, speaker_table):
    if not session_choice or speaker_table is None:
        return "Nothing to save."

    session_id = int(session_choice.split("]")[0].lstrip("["))
    db = _db()

    for row in speaker_table:
        spk_id, orig, new_name, _ = row[0], row[1], row[2], row[3]
        if new_name and str(new_name).strip():
            db.update_speaker_name(int(spk_id), str(new_name).strip())

    # Rebuild transcript with new names
    transcript = db.rebuild_transcript(session_id)
    return transcript


def refresh_sessions_dropdown():
    choices = _load_session_list()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def export_transcript(session_choice: str):
    if not session_choice:
        return None
    session_id = int(session_choice.split("]")[0].lstrip("["))
    db = _db()
    session = db.get_session(session_id)
    if not session or not session.get("output_dir"):
        return None
    txt_path = Path(session["output_dir"]) / "transcript.txt"
    if txt_path.exists():
        return str(txt_path)
    # Fallback: write on the fly
    text = db.rebuild_transcript(session_id)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    return str(txt_path)


# ── Tab 3: Voice Library ──────────────────────────────────────────────────────

def load_voice_library():
    profiles = _db().get_voice_profiles()
    return [[p["id"], p["name"], p.get("audio_path", "—"),
             _ts(p["created_at"])] for p in profiles]


def add_voice_profile(name: str, audio_file):
    if not name or not name.strip():
        raise gr.Error("Please enter a speaker name.")
    name = name.strip()
    db = _db()

    audio_path = None
    embedding = None

    if audio_file:
        # Copy into voice_samples_dir for permanent storage
        dst = cfg.voice_samples_dir / f"{name.replace(' ', '_')}.wav"
        shutil.copy2(audio_file, dst)
        audio_path = str(dst)
        embedding = compute_embedding(audio_path)
        if embedding is None:
            raise gr.Error("Could not compute embedding — audio may be too short. Use 5-30s clips.")

    db.upsert_voice_profile(name=name, audio_path=audio_path, embedding=embedding)
    return f"Saved profile '{name}'.", load_voice_library()


def delete_voice_profile(profile_table, selected_index):
    if profile_table is None or len(profile_table) == 0:
        return "Nothing to delete.", load_voice_library()
    try:
        row = profile_table[selected_index]
        profile_id = int(row[0])
        name = row[1]
        _db().delete_voice_profile(profile_id)
        return f"Deleted '{name}'.", load_voice_library()
    except (IndexError, TypeError):
        return "Select a row to delete.", load_voice_library()


# ── Tab 4: History ────────────────────────────────────────────────────────────

def load_history():
    sessions = _db().list_sessions()
    rows = []
    for s in sessions:
        rows.append([
            s["id"],
            _ts(s["created_at"]),
            s["input_filename"],
            s["status"],
            s.get("whisper_model", "—"),
            s.get("language") or "auto",
            _dur(s.get("elapsed_seconds")),
        ])
    return rows


def delete_session_action(history_table, selected_index):
    if history_table is None or len(history_table) == 0:
        return "Nothing to delete.", load_history()
    try:
        row = history_table[selected_index]
        session_id = int(row[0])
        _db().delete_session(session_id)
        return f"Deleted session {session_id}.", load_history()
    except (IndexError, TypeError):
        return "Select a row to delete.", load_history()


# ── UI layout ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    session_choices = _load_session_list()

    with gr.Blocks(title="Meeting Transcription", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# Meeting Transcription App\n"
            "**Pipeline:** Video → Demucs denoising → Pyannote diarization "
            "→ Resemblyzer identification → Whisper transcription"
        )

        # ── Shared state ──────────────────────────────────────────────────────
        current_session_id = gr.State("")

        # ════════════════════════════════════════════════════════════════════
        # Tab 1 — Process
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Process"):

            with gr.Row():
                with gr.Column(scale=1):
                    input_file = gr.File(
                        label="Upload Meeting Video / Audio",
                        file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm",
                                    ".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                        type="filepath",
                    )

                    with gr.Accordion("Transcription", open=True):
                        language = gr.Dropdown(
                            label="Language",
                            choices=["auto", "en", "fa", "de", "fr", "es", "it",
                                     "pt", "nl", "ru", "zh", "ja", "ko", "ar",
                                     "tr", "pl", "uk", "cs", "sv", "da", "fi",
                                     "hu", "ro", "bg", "he", "hi", "id", "vi"],
                            value="auto",
                        )
                        whisper_model = gr.Dropdown(
                            label="Whisper Model",
                            choices=["large-v3", "large-v2", "medium", "small", "base", "tiny"],
                            value=cfg.default_whisper_model,
                        )

                    with gr.Accordion("Denoising", open=False):
                        skip_demucs = gr.Checkbox(
                            label="Skip Demucs (audio already clean)",
                            value=False,
                        )
                        demucs_model = gr.Dropdown(
                            label="Demucs Model",
                            choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                            value=cfg.default_demucs_model,
                        )

                    with gr.Accordion("Diarization", open=False):
                        hf_token = gr.Textbox(
                            label="Hugging Face Token",
                            placeholder="hf_… (or set HF_TOKEN in .env)",
                            type="password",
                        )
                        with gr.Row():
                            num_speakers = gr.Number(label="Exact speakers (0=auto)", value=0, precision=0)
                        with gr.Row():
                            min_speakers = gr.Number(label="Min speakers", value=0, precision=0)
                            max_speakers = gr.Number(label="Max speakers", value=0, precision=0)

                    with gr.Accordion("Speaker Identification (optional)", open=False):
                        use_voice_library = gr.Checkbox(
                            label="Use Voice Library for auto-identification",
                            value=False,
                            info="Manage profiles in the 'Voice Library' tab.",
                        )
                        similarity_threshold = gr.Slider(
                            label="Match Threshold",
                            minimum=0.5, maximum=0.95,
                            value=cfg.default_similarity_threshold,
                            step=0.05,
                        )

                    run_btn = gr.Button("Process Meeting", variant="primary", size="lg")

                with gr.Column(scale=1):
                    process_status = gr.Textbox(label="Status", lines=2, interactive=False)
                    process_logs = gr.Textbox(label="Logs", lines=20, interactive=False)

            run_btn.click(
                fn=process_meeting,
                inputs=[
                    input_file, language, whisper_model, skip_demucs, demucs_model,
                    num_speakers, min_speakers, max_speakers,
                    hf_token, use_voice_library, similarity_threshold,
                ],
                outputs=[process_status, process_logs, current_session_id],
            )

        # ════════════════════════════════════════════════════════════════════
        # Tab 2 — Results
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("Results"):
            with gr.Row():
                session_dd = gr.Dropdown(
                    label="Select Session",
                    choices=session_choices,
                    value=session_choices[0] if session_choices else None,
                    scale=4,
                )
                refresh_btn = gr.Button("Refresh", scale=1)
                export_btn  = gr.Button("Export Transcript", scale=1)

            export_file = gr.File(label="Download Transcript", visible=False)

            with gr.Row():
                with gr.Column(scale=2):
                    transcript_box = gr.Textbox(
                        label="Transcript (updates when you save names)",
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
                    gr.Markdown("### Edit Speaker Names")
                    gr.Markdown(
                        "Double-click a **Display Name** cell to rename a speaker. "
                        "Click **Save Names** to persist and re-render the transcript."
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
                fn=refresh_sessions_dropdown,
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
                "Save named voice samples here. When processing a meeting, enable "
                "**'Use Voice Library'** in the Process tab to auto-identify speakers."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Add / Update Profile")
                    new_profile_name  = gr.Textbox(label="Speaker Name", placeholder="e.g. Alice")
                    new_profile_audio = gr.Audio(
                        label="Voice Sample (5-30s recommended, optional)",
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
                        label="Voice Profiles",
                    )
                    with gr.Row():
                        selected_profile_idx = gr.Number(
                            label="Row index to delete (0-based)", value=0, precision=0
                        )
                        delete_profile_btn = gr.Button("Delete Selected", variant="stop")
                    delete_status = gr.Textbox(label="Status", lines=1, interactive=False)

            # Load profiles on tab open
            demo.load(fn=load_voice_library, inputs=[], outputs=[profile_table])

            add_btn.click(
                fn=add_voice_profile,
                inputs=[new_profile_name, new_profile_audio],
                outputs=[add_status, profile_table],
            )
            delete_profile_btn.click(
                fn=delete_voice_profile,
                inputs=[profile_table, selected_profile_idx],
                outputs=[delete_status, profile_table],
            )

        # ════════════════════════════════════════════════════════════════════
        # Tab 4 — History
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("History"):
            refresh_history_btn = gr.Button("Refresh")
            history_table = gr.Dataframe(
                headers=["ID", "Date", "File", "Status", "Model", "Language", "Duration"],
                datatype=["number", "str", "str", "str", "str", "str", "str"],
                col_count=(7, "fixed"),
                interactive=False,
                label="Past Sessions",
            )
            with gr.Row():
                selected_session_idx = gr.Number(
                    label="Row index to delete (0-based)", value=0, precision=0
                )
                delete_session_btn = gr.Button("Delete Session", variant="stop")
            history_status = gr.Textbox(label="Status", lines=1, interactive=False)

            demo.load(fn=load_history, inputs=[], outputs=[history_table])
            refresh_history_btn.click(fn=load_history, inputs=[], outputs=[history_table])
            delete_session_btn.click(
                fn=delete_session_action,
                inputs=[history_table, selected_session_idx],
                outputs=[history_status, history_table],
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
