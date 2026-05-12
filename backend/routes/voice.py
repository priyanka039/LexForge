# ─────────────────────────────────────────────
# routes/voice.py  ·  Sarvam AI voice layer
#
#   POST /api/voice/transcribe   (multipart audio)   → { transcript }
#   POST /api/voice/speak        (json text+role)    → { audio_b64, format }
#   GET  /api/voice/languages                        → supported list
#
# All Sarvam network calls happen here / in
# sarvam.py — the API key is never exposed to the
# browser. Errors are swallowed so the user UX
# stays "silent".
# ─────────────────────────────────────────────

import base64
import os
from typing import Optional

import httpx
from fastapi  import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from sarvam import (
    SUPPORTED_LANGUAGES,
    SPEAKER_MAP,
    SARVAM_BASE_URL,
    SARVAM_HEADER_NAME,
    TTS_MODEL,
    STT_MODEL,
    TRANSLATE_MODEL,
    transcribe          as sarvam_transcribe,
    speak_with_translate,
    normalize_lang,
)

router = APIRouter()


# ── /transcribe ──────────────────────────────
@router.post("/api/voice/transcribe")
async def voice_transcribe(
    audio:    UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """
    Accepts audio (any browser MediaRecorder blob) and returns Sarvam STT
    transcript. Returns empty transcript on any failure — never raises.
    """
    try:
        data = await audio.read()
    except Exception:
        return {"transcript": ""}

    transcript = await sarvam_transcribe(
        audio_bytes = data,
        filename    = audio.filename or "audio.webm",
        mime        = audio.content_type or "audio/webm",
        language    = language,
    )
    return {"transcript": transcript}


# ── /speak ───────────────────────────────────
class SpeakRequest(BaseModel):
    text: str
    role: Optional[str] = "default"
    lang: Optional[str] = "en-IN"


@router.post("/api/voice/speak")
async def voice_speak(req: SpeakRequest):
    """
    Pipeline:
      1. If `lang` != en-IN  → translate(text, en-IN → lang)  [silent]
      2. text-to-speech with role-mapped speaker, in `lang`
      3. Return base64 WAV.

    On any failure returns an empty audio string so the frontend can
    silently no-op.
    """
    audio_bytes = await speak_with_translate(
        text        = req.text or "",
        role        = req.role,
        target_lang = req.lang or "en-IN",
    )
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else ""
    return {
        "audio_b64": audio_b64,
        "format":    "wav",
        "lang":      normalize_lang(req.lang),
        # Hint for the browser console — never shown in the UI.
        "ok":        bool(audio_b64),
        "reason":    "" if audio_b64 else (
            "missing_api_key" if not os.getenv("SARVAM_API_KEY", "").strip()
            else "sarvam_error_check_server_log_or_voice_health"
        ),
    }


# ── /languages ───────────────────────────────
@router.get("/api/voice/languages")
def voice_languages():
    return {
        "languages": SUPPORTED_LANGUAGES,
        "default":   "en-IN",
        "roles":     list(SPEAKER_MAP.keys()),
    }


# ── /health ──────────────────────────────────
# A diagnostic endpoint. Hit GET /api/voice/health from the browser to see
# why TTS / STT might be failing without digging through server logs.
@router.get("/api/voice/health")
async def voice_health():
    key = os.getenv("SARVAM_API_KEY", "").strip()
    out: dict = {
        "key_present":  bool(key),
        "key_length":   len(key),
        "key_preview":  (key[:6] + "…" + key[-4:]) if len(key) >= 12 else "",
        "stt_model":    STT_MODEL,
        "translate_model": TRANSLATE_MODEL,
        "tts_model":    TTS_MODEL,
        "tts_test":     None,
    }
    if not key:
        out["tts_test"] = {
            "ok":    False,
            "error": "SARVAM_API_KEY missing — create lexforge/.env with SARVAM_API_KEY=...",
        }
        return out

    # One-shot live TTS so the user sees the actual Sarvam response.
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{SARVAM_BASE_URL}/text-to-speech",
                headers={SARVAM_HEADER_NAME: key, "Content-Type": "application/json"},
                json={
                    "text":                 "Hello from LexForge.",
                    "target_language_code": "en-IN",
                    "speaker":              "aditya",
                    "model":                TTS_MODEL,
                },
            )
        out["tts_test"] = {
            "ok":          200 <= r.status_code < 300,
            "status_code": r.status_code,
            "body_preview": r.text[:400],
        }
    except Exception as e:
        out["tts_test"] = {
            "ok":    False,
            "error": f"{type(e).__name__}: {e}",
        }
    return out
