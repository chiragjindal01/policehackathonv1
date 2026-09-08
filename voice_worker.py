"""
voice_worker.py
----------------
Voice-note evidence ingestion for TETCP.

Mirrors the OCR fork (Fork A) pattern: turn a raw seized artefact (here,
an audio recording instead of a screenshot) into normalized evidence lines
that flow through the *existing* storage.parse_and_ingest_file() pipeline,
so entity extraction, FTS5 search, triage leads and the link graph all
pick it up automatically with zero changes to those subsystems.

Air-gap law respected: no network calls. Transcription uses a local
faster-whisper model if installed; if not, we don't block ingestion —
we store the audio, log a placeholder line, and flag it for manual
transcription (same philosophy as the "heuristic fallback" already used
for the local LLM in server.py).

Dependencies (all optional, all local):
    pip install faster-whisper      # CPU-friendly local speech-to-text
No dependency is required for the code to run — it degrades gracefully.
"""

import hashlib
import os
import time
import wave
import contextlib

import storage  # your existing storage.py — must expose get_db() and
                 # parse_and_ingest_file(case_id, filename, bytes_content)

VOICE_NOTE_DIR = os.path.join("data", "raw", "voice_notes")
os.makedirs(VOICE_NOTE_DIR, exist_ok=True)

# Lazily-loaded whisper model (only loaded the first time it's needed,
# so the app still starts instantly if faster-whisper isn't installed).
_whisper_model = None
_whisper_load_attempted = False


def _get_whisper_model():
    """Load a local faster-whisper model once and cache it. Returns None
    if faster-whisper isn't installed — caller must handle that."""
    global _whisper_model, _whisper_load_attempted
    if _whisper_load_attempted:
        return _whisper_model
    _whisper_load_attempted = True
    try:
        from faster_whisper import WhisperModel
        # "base" is a good size/accuracy trade-off for police laptops.
        # "tiny" is faster but less accurate; "small"/"medium" are slower
        # but better on noisy audio. int8 keeps CPU-only inference fast.
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        _whisper_model = None
    return _whisper_model


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_audio_file(case_id: str, audio_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """Persist the raw audio to disk under a content-addressed name.
    Returns (absolute_path, sha256_hash)."""
    file_hash = _sha256_bytes(audio_bytes)
    ext = os.path.splitext(original_filename)[1].lower() or ".webm"
    safe_name = f"{case_id}_{file_hash[:16]}{ext}"
    path = os.path.join(VOICE_NOTE_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path, file_hash


def _get_duration_seconds(path: str) -> float:
    """Best-effort duration read; only works for WAV. Non-fatal if it fails
    (webm/ogg from MediaRecorder won't parse here — that's fine, duration
    is a nice-to-have for the UI, not a requirement)."""
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0


def transcribe_audio(path: str) -> tuple[str, bool]:
    """Transcribe an audio file locally.

    Returns (transcript_text, was_transcribed).
    If no local model is available, returns a placeholder and False,
    so the caller can flag the record for manual review instead of
    silently dropping it.
    """
    model = _get_whisper_model()
    if model is None:
        return (
            "[PENDING MANUAL TRANSCRIPTION — no local speech-to-text model "
            "available. Install faster-whisper or transcribe manually and "
            "re-ingest.]",
            False,
        )

    try:
        segments, info = model.transcribe(path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            text = "[TRANSCRIPTION EMPTY — silence or unsupported audio]"
        return text, True
    except Exception as e:
        return f"[TRANSCRIPTION FAILED: {e}]", False


def process_voice_note(audio_bytes: bytes, filename: str, case_id: str,
                        sender_id: str = "UNKNOWN") -> dict:
    """
    Main entry point — call this from the /api/upload_voice handler in
    server.py.

    1. Saves the raw audio (content-addressed, SHA-256 hashed — satisfies
       the same Section 63 BSA chain-of-custody requirement as file uploads).
    2. Transcribes it locally (or flags for manual transcription).
    3. Feeds the transcript into the existing evidence pipeline via
       storage.parse_and_ingest_file(), so entity extraction / FTS5 / leads
       all work exactly as they do for any other ingested file.
    4. Links the resulting evidence_files row to the audio file on disk so
       the frontend can render an <audio> player next to the transcript.

    Returns a dict describing what happened — mirror this shape in the
    server.py JSON response.
    """
    audio_path, audio_hash = _save_audio_file(case_id, audio_bytes, filename)
    duration = _get_duration_seconds(audio_path)
    transcript, was_transcribed = transcribe_audio(audio_path)

    # Build a single evidence "line" the same shape your other ingestors
    # produce: one timestamped, sender-tagged line of text. Voice notes are
    # a single utterance, so we ingest it as one line, prefixed so it's
    # unmistakably distinguishable from typed chat text during triage.
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[VOICE NOTE {timestamp}] {sender_id}: {transcript}"

    # Re-use your existing ingestion pipeline exactly as file uploads do —
    # this is what wires entity extraction, records_fts, and Panel 1/2/3
    # up automatically with no further code.
    ingest_result = storage.parse_and_ingest_file(
        case_id, filename, line.encode("utf-8")
    )

    # Attach audio metadata onto the resulting evidence_files row so the
    # frontend can play back the original recording next to the transcript.
    # Requires the small storage.py patch below (adds audio_path column).
    file_id = ingest_result.get("file_id") if isinstance(ingest_result, dict) else None
    if file_id is not None:
        storage.attach_audio_to_file(file_id, audio_path, audio_hash, duration)

    return {
        "file_id": file_id,
        "audio_hash_sha256": audio_hash,
        "audio_path": audio_path,
        "duration_seconds": duration,
        "transcript": transcript,
        "transcribed_locally": was_transcribed,
        "ingest_result": ingest_result,
    }
