"""
voice_worker.py - Offline Voice Note Ingestion & Transcription
Chandigarh Police Hackathon 2026 - PS-3
"""

import os
import hashlib
from datetime import datetime
import storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "data", "raw", "voice_notes")

_whisper_model = None


def _get_whisper_model():
    """Lazy-loads faster-whisper model only once, if installed."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        return _whisper_model
    except Exception as e:
        print(f"[voice_worker] faster-whisper not available: {e}")
        return None


def process_voice_note(audio_bytes: bytes, filename: str, case_id: str, sender_id: str = "FIELD_OFFICER"):
    """
    Saves the recorded audio, attempts local offline transcription,
    and ingests the transcript through the existing evidence pipeline.
    """
    os.makedirs(VOICE_DIR, exist_ok=True)

    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    safe_filename = f"{audio_hash[:12]}_{filename}"
    audio_path_abs = os.path.join(VOICE_DIR, safe_filename)

    with open(audio_path_abs, "wb") as f:
        f.write(audio_bytes)

    # Relative path (from repo root) so frontend can request it via static server
    audio_path_rel = os.path.relpath(audio_path_abs, BASE_DIR).replace("\\", "/")

    transcribed_locally = False
    duration_seconds = 0.0
    transcript_text = ""

    model = _get_whisper_model()
    if model is not None:
        try:
            segments, info = model.transcribe(audio_path_abs, beam_size=5)
            duration_seconds = round(getattr(info, "duration", 0.0), 2)
            transcript_text = " ".join(seg.text.strip() for seg in segments).strip()
            transcribed_locally = True
        except Exception as e:
            print(f"[voice_worker] Transcription failed: {e}")
            transcript_text = ""

    if not transcript_text:
        transcript_text = "[PENDING MANUAL TRANSCRIPTION - no local STT model available or transcription failed]"

    now_str = datetime.utcnow().isoformat() + "Z"
    transcript_filename = filename.rsplit(".", 1)[0] + "_voice_transcript.txt"
    transcript_content = f"[VOICE NOTE from {sender_id} at {now_str}]\n{transcript_text}"

    # Ingest transcript through the normal evidence pipeline
    ingest_result = storage.parse_and_ingest_file(
        case_id,
        transcript_filename,
        transcript_content.encode("utf-8")
    )

    file_id = ingest_result.get("file_id")

    # Link the audio file to that evidence_files row
    if file_id:
        storage.attach_audio_to_file(
            file_id=file_id,
            audio_path=audio_path_rel,
            audio_hash=audio_hash,
            duration=duration_seconds
        )

    storage.log_audit(
        case_id,
        "VOICE_NOTE_INGESTED",
        f"Voice note '{filename}' ingested ({duration_seconds}s, transcribed_locally={transcribed_locally}) -> {file_id}",
        performed_by=sender_id
    )

    return {
        "status": "success",
        "file_id": file_id,
        "filename": transcript_filename,
        "audio_path": audio_path_rel,
        "audio_hash": audio_hash,
        "duration_seconds": duration_seconds,
        "transcribed_locally": transcribed_locally,
        "transcript_preview": transcript_text[:200],
        "ingest_data": ingest_result
    }
