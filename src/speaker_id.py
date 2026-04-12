"""
Speaker Identification via Resemblyzer
Matches diarized speaker labels to named voice profiles using cosine similarity.
Voice samples are optional — if none are provided, generic labels are kept.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.75


# ── Embedding utilities ───────────────────────────────────────────────────────

def _get_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


def compute_embedding(audio_path: str | Path) -> np.ndarray | None:
    """
    Compute a d-vector embedding for a single audio file.
    Returns None if the file is too short or fails.
    """
    try:
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(Path(audio_path))
        if len(wav) < 1600:
            logger.warning(f"Audio too short for embedding: {audio_path}")
            return None
        encoder = _get_encoder()
        return encoder.embed_utterance(wav)
    except Exception as e:
        logger.warning(f"Embedding failed for {audio_path}: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Extract speaker audio from full recording ─────────────────────────────────

def _extract_speaker_chunks(
    audio_data: np.ndarray, sr: int, segments: list, speaker_label: str
) -> np.ndarray:
    """Concatenate all audio chunks belonging to `speaker_label`."""
    chunks = []
    for seg in segments:
        if seg.speaker == speaker_label:
            s = int(seg.start * sr)
            e = int(seg.end * sr)
            chunk = audio_data[s:e]
            if len(chunk) > 0:
                chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ── Main identification function ──────────────────────────────────────────────

def identify_speakers(
    audio_path: str | Path,
    segments: list,
    voice_profiles: list[dict],       # list of {"name": str, "embedding": np.ndarray | None, "audio_path": str | None}
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, str]:
    """
    Match diarized speaker labels to named profiles.

    Args:
        audio_path: Full audio file used for diarization.
        segments: List of diarizer.Segment.
        voice_profiles: Known speakers. Each dict must have "name" key and
                        either "embedding" (np.ndarray) or "audio_path" (str).
        threshold: Cosine similarity threshold. No match → keep original label.

    Returns:
        Dict {original_label: display_name}.
        Unmatched speakers map to their original label.
    """
    import soundfile as sf

    unique_labels = sorted(set(seg.speaker for seg in segments))

    # Fast-path: no profiles
    if not voice_profiles:
        return {lbl: lbl for lbl in unique_labels}

    # Load known embeddings (compute on-the-fly if only audio_path given)
    known: dict[str, np.ndarray] = {}
    for prof in voice_profiles:
        name = prof["name"]
        emb = prof.get("embedding")
        if emb is None and prof.get("audio_path"):
            emb = compute_embedding(prof["audio_path"])
        if emb is not None:
            known[name] = emb
            logger.info(f"Loaded embedding for '{name}'")
        else:
            logger.warning(f"No usable embedding for profile '{name}', skipping.")

    if not known:
        return {lbl: lbl for lbl in unique_labels}

    # Load full audio
    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    encoder = _get_encoder()
    speaker_map: dict[str, str] = {}
    used_names: set[str] = set()

    for label in unique_labels:
        speaker_audio = _extract_speaker_chunks(audio_data, sr, segments, label)

        if len(speaker_audio) < 1600:
            logger.warning(f"{label}: too little audio for identification.")
            speaker_map[label] = label
            continue

        # Write temp WAV so resemblyzer can preprocess it properly
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            import soundfile as sf2
            sf2.write(str(tmp), speaker_audio, sr)
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(tmp)
            spk_emb = encoder.embed_utterance(wav)
        except Exception as e:
            logger.warning(f"Failed embedding for {label}: {e}")
            speaker_map[label] = label
            continue
        finally:
            tmp.unlink(missing_ok=True)

        best_name, best_score = label, -1.0
        for name, emb in known.items():
            if name in used_names:
                continue
            score = cosine_similarity(spk_emb, emb)
            logger.info(f"  {label} vs '{name}': {score:.3f}")
            if score > best_score:
                best_score = score
                if score >= threshold:
                    best_name = name

        speaker_map[label] = best_name
        if best_name != label:
            used_names.add(best_name)
            logger.info(f"  → {label} identified as '{best_name}' (score={best_score:.3f})")
        else:
            logger.info(f"  → {label} unmatched (best score={best_score:.3f})")

    return speaker_map
