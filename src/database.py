"""
Database layer — SQLite via the stdlib sqlite3 module.

Schema
------
sessions      : one row per pipeline run
speakers      : one row per unique speaker per session (display_name editable)
audio_chunks  : one row per diarized segment (transcript, audio path)
voice_profiles: global library of named speakers with stored embeddings
"""

import sqlite3
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

# ── DDL ──────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      REAL    NOT NULL DEFAULT (unixepoch('now')),
    input_filename  TEXT    NOT NULL,
    input_path      TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    demucs_model    TEXT,
    skip_demucs     INTEGER NOT NULL DEFAULT 0,
    whisper_model   TEXT,
    language        TEXT,
    num_speakers    INTEGER,
    output_dir      TEXT,
    total_duration  REAL,
    elapsed_seconds REAL
);

CREATE TABLE IF NOT EXISTS speakers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    original_label  TEXT    NOT NULL,
    display_name    TEXT    NOT NULL,
    total_duration  REAL    NOT NULL DEFAULT 0,
    voice_sample_path TEXT,
    UNIQUE(session_id, original_label)
);

CREATE TABLE IF NOT EXISTS audio_chunks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    speaker_id        INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    start_time        REAL    NOT NULL,
    end_time          REAL    NOT NULL,
    chunk_audio_path  TEXT,
    transcript        TEXT,
    language_detected TEXT
);

CREATE TABLE IF NOT EXISTS voice_profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now')),
    name        TEXT    NOT NULL UNIQUE,
    audio_path  TEXT,
    embedding   BLOB
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

class Database:
    """Thin wrapper around sqlite3 with helpers for each table."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init()

    def _init(self):
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(
        self,
        input_path: str,
        demucs_model: str = None,
        skip_demucs: bool = False,
        whisper_model: str = None,
        language: str = None,
        num_speakers: int = None,
        output_dir: str = None,
    ) -> int:
        input_path = str(input_path)
        filename = Path(input_path).name
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (input_filename, input_path, status,
                    demucs_model, skip_demucs, whisper_model,
                    language, num_speakers, output_dir)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (filename, input_path, "processing",
                 demucs_model, int(skip_demucs), whisper_model,
                 language, num_speakers, output_dir),
            )
            return cur.lastrowid

    def update_session(self, session_id: int, **kwargs) -> None:
        if not kwargs:
            return
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [session_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE sessions SET {cols} WHERE id=?", vals)

    def get_session(self, session_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ── Speakers ──────────────────────────────────────────────────────────────

    def upsert_speaker(
        self,
        session_id: int,
        original_label: str,
        display_name: str,
        total_duration: float = 0.0,
        voice_sample_path: str = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO speakers
                   (session_id, original_label, display_name, total_duration, voice_sample_path)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(session_id, original_label)
                   DO UPDATE SET
                       display_name=excluded.display_name,
                       total_duration=excluded.total_duration,
                       voice_sample_path=excluded.voice_sample_path""",
                (session_id, original_label, display_name, total_duration, voice_sample_path),
            )
            # Return the id (INSERT or existing)
            row = conn.execute(
                "SELECT id FROM speakers WHERE session_id=? AND original_label=?",
                (session_id, original_label),
            ).fetchone()
            return row["id"]

    def update_speaker_name(self, speaker_id: int, display_name: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE speakers SET display_name=? WHERE id=?",
                (display_name, speaker_id),
            )

    def get_speakers(self, session_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM speakers WHERE session_id=? ORDER BY original_label",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Audio Chunks ──────────────────────────────────────────────────────────

    def insert_chunk(
        self,
        session_id: int,
        speaker_id: int,
        start_time: float,
        end_time: float,
        chunk_audio_path: str = None,
        transcript: str = None,
        language_detected: str = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO audio_chunks
                   (session_id, speaker_id, start_time, end_time,
                    chunk_audio_path, transcript, language_detected)
                   VALUES (?,?,?,?,?,?,?)""",
                (session_id, speaker_id, start_time, end_time,
                 chunk_audio_path, transcript, language_detected),
            )
            return cur.lastrowid

    def get_chunks(self, session_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, s.original_label, s.display_name
                   FROM audio_chunks c
                   JOIN speakers s ON c.speaker_id = s.id
                   WHERE c.session_id=?
                   ORDER BY c.start_time""",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_chunk_transcript(self, chunk_id: int, transcript: str, language_detected: str = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE audio_chunks SET transcript=?, language_detected=? WHERE id=?",
                (transcript, language_detected, chunk_id),
            )

    # ── Voice Profiles (global speaker library) ───────────────────────────────

    def upsert_voice_profile(
        self, name: str, audio_path: str = None, embedding: np.ndarray = None
    ) -> int:
        emb_bytes = pickle.dumps(embedding) if embedding is not None else None
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO voice_profiles (name, audio_path, embedding)
                   VALUES (?,?,?)
                   ON CONFLICT(name) DO UPDATE SET
                       audio_path=excluded.audio_path,
                       embedding=excluded.embedding""",
                (name, audio_path, emb_bytes),
            )
            row = conn.execute(
                "SELECT id FROM voice_profiles WHERE name=?", (name,)
            ).fetchone()
            return row["id"]

    def get_voice_profiles(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, name, audio_path FROM voice_profiles ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_voice_profile_embedding(self, profile_id: int) -> Optional[np.ndarray]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM voice_profiles WHERE id=?", (profile_id,)
            ).fetchone()
            if row and row["embedding"]:
                return pickle.loads(row["embedding"])
            return None

    def delete_voice_profile(self, profile_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM voice_profiles WHERE id=?", (profile_id,))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def build_speaker_map(self, session_id: int) -> dict[str, str]:
        """Return {original_label: display_name} for a session."""
        return {s["original_label"]: s["display_name"] for s in self.get_speakers(session_id)}

    def rebuild_transcript(self, session_id: int) -> str:
        """Re-render transcript using current display_names (reflects edits)."""
        chunks = self.get_chunks(session_id)
        lines = []
        for c in chunks:
            if not c.get("transcript"):
                continue
            name = c["display_name"]
            lines.append(
                f"[{c['start_time']:07.3f}s - {c['end_time']:07.3f}s]  {name}:\n"
                f"  {c['transcript']}\n"
            )
        return "\n".join(lines)
