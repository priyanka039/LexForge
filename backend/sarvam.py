# ─────────────────────────────────────────────
# sarvam.py
# Thin async client around Sarvam AI:
#   - speech-to-text          (saarika:v2)
#   - translate               (mayura:v1)
#   - text-to-speech          (bulbul:v3)
#
# Loaded lazily so the rest of the app boots
# even if SARVAM_API_KEY is missing or the network
# is offline. All public functions return safe
# fallbacks (empty string / empty bytes) on any
# error — callers must never see a Sarvam
# exception bubble up to the user.
# ─────────────────────────────────────────────

from __future__ import annotations

import base64
import os
from typing import Optional

import httpx

SARVAM_BASE_URL    = "https://api.sarvam.ai"
SARVAM_HEADER_NAME = "api-subscription-key"

STT_MODEL          = "saarika:v2.5"
TRANSLATE_MODEL    = "mayura:v1"
TTS_MODEL          = "bulbul:v3"
TTS_TEXT_LIMIT     = 2400        # Sarvam bulbul:v3 caps at 2500 chars; leave buffer.

# Lightweight log helper — never crashes the app, prints in the dev terminal.
def _log(msg: str) -> None:
    try:
        print(f"[sarvam] {msg}", flush=True)
    except Exception:
        pass

# Petitioner / opposition / bench / researcher voices.
# Backend-only — never exposed in any API response.
SPEAKER_MAP = {
    "petitioner": "rahul",
    "opposition": "priya",
    "judge":      "aditya",
    "researcher": "neha",
    "default":    "aditya",
}

SUPPORTED_LANGUAGES = [
    {"code": "en-IN", "label": "English (India)"},
    {"code": "hi-IN", "label": "हिन्दी"},
    {"code": "ta-IN", "label": "தமிழ்"},
    {"code": "te-IN", "label": "తెలుగు"},
    {"code": "kn-IN", "label": "ಕನ್ನಡ"},
    {"code": "ml-IN", "label": "മലയാളം"},
    {"code": "bn-IN", "label": "বাংলা"},
    {"code": "mr-IN", "label": "मराठी"},
    {"code": "gu-IN", "label": "ગુજરાતી"},
    {"code": "pa-IN", "label": "ਪੰਜਾਬੀ"},
]
SUPPORTED_LANG_CODES = {l["code"] for l in SUPPORTED_LANGUAGES}


def _api_key() -> str:
    return os.getenv("SARVAM_API_KEY", "").strip()


def _headers() -> dict:
    return {SARVAM_HEADER_NAME: _api_key()}


def resolve_speaker(role: Optional[str]) -> str:
    if not role:
        return SPEAKER_MAP["default"]
    return SPEAKER_MAP.get(role.strip().lower(), SPEAKER_MAP["default"])


def normalize_lang(lang: Optional[str]) -> str:
    """Returns a Sarvam BCP-47 code, defaulting to en-IN."""
    if not lang:
        return "en-IN"
    code = lang.strip()
    if code in SUPPORTED_LANG_CODES:
        return code
    # Allow plain "hi", "ta" etc. and map to *-IN
    short = code.split("-")[0].lower()
    for c in SUPPORTED_LANG_CODES:
        if c.split("-")[0] == short:
            return c
    return "en-IN"


# ═════════════════════════════════════════════
# 1. SPEECH-TO-TEXT
# ═════════════════════════════════════════════
async def transcribe(
    audio_bytes: bytes,
    filename:    str = "audio.wav",
    mime:        str = "audio/wav",
    language:    Optional[str] = None,
) -> str:
    """
    POST /speech-to-text. Returns the transcript or "" on any failure.
    `language` is optional — Sarvam supports auto-detect; if a code is
    given it is used as a hint.
    """
    if not audio_bytes:
        _log("transcribe: empty audio bytes")
        return ""
    if not _api_key():
        _log("transcribe: SARVAM_API_KEY not set")
        return ""
    try:
        data = {"model": STT_MODEL}
        if language:
            data["language_code"] = normalize_lang(language)
        files = {"file": (filename, audio_bytes, mime)}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{SARVAM_BASE_URL}/speech-to-text",
                headers=_headers(),
                data=data,
                files=files,
            )
        if r.status_code >= 400:
            _log(f"transcribe: HTTP {r.status_code} body={r.text[:300]}")
            return ""
        payload = r.json() or {}
        return (payload.get("transcript") or "").strip()
    except Exception as e:
        _log(f"transcribe: exception {type(e).__name__}: {e}")
        return ""


# ═════════════════════════════════════════════
# 2. TRANSLATE  (English → target Indian language)
# ═════════════════════════════════════════════
async def translate(text: str, target_lang: str, source_lang: str = "en-IN") -> str:
    """
    POST /translate. Returns translated text. On ANY error returns the
    original text — callers must fall back silently.
    """
    if not text:
        return ""
    target = normalize_lang(target_lang)
    if target == "en-IN":
        return text
    if not _api_key():
        _log("translate: SARVAM_API_KEY not set — returning original text")
        return text
    try:
        body = {
            "input":                 text,
            "source_language_code":  normalize_lang(source_lang),
            "target_language_code":  target,
            "model":                 TRANSLATE_MODEL,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{SARVAM_BASE_URL}/translate",
                headers={**_headers(), "Content-Type": "application/json"},
                json=body,
            )
        if r.status_code >= 400:
            _log(f"translate: HTTP {r.status_code} body={r.text[:300]}")
            return text
        payload = r.json() or {}
        translated = (payload.get("translated_text") or "").strip()
        return translated or text
    except Exception as e:
        _log(f"translate: exception {type(e).__name__}: {e}")
        return text


# ═════════════════════════════════════════════
# 3. TEXT-TO-SPEECH
# ═════════════════════════════════════════════
async def speak(text: str, speaker: str, lang: str = "en-IN") -> bytes:
    """
    POST /text-to-speech. Returns raw WAV bytes; b"" on any failure.
    Sarvam returns base64-encoded audio in `audios[0]`.
    """
    if not text:
        _log("speak: empty text")
        return b""
    if not _api_key():
        _log("speak: SARVAM_API_KEY not set")
        return b""
    try:
        # bulbul:v3 max 2500 chars — clamp to be safe.
        clamped = text[:TTS_TEXT_LIMIT]
        body = {
            "text":                 clamped,
            "target_language_code": normalize_lang(lang),
            "speaker":              (speaker or SPEAKER_MAP["default"]).lower(),
            "model":                TTS_MODEL,
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                f"{SARVAM_BASE_URL}/text-to-speech",
                headers={**_headers(), "Content-Type": "application/json"},
                json=body,
            )
        if r.status_code >= 400:
            _log(f"speak: HTTP {r.status_code} body={r.text[:300]}")
            return b""
        payload = r.json() or {}
        audios = payload.get("audios") or []
        if not audios:
            _log(f"speak: empty audios in response keys={list(payload.keys())}")
            return b""
        try:
            return base64.b64decode(audios[0])
        except Exception as e:
            _log(f"speak: base64 decode failed: {e}")
            return b""
    except Exception as e:
        _log(f"speak: exception {type(e).__name__}: {e}")
        return b""


# ═════════════════════════════════════════════
# Convenience: full speak pipeline with optional
# silent translation (English in → translate →
# TTS in target language). Used by routes/voice.py.
# ═════════════════════════════════════════════
async def speak_with_translate(
    text:        str,
    role:        Optional[str],
    target_lang: str = "en-IN",
) -> bytes:
    if not text:
        return b""
    speaker = resolve_speaker(role)
    target  = normalize_lang(target_lang)
    body_text = text
    if target != "en-IN":
        try:
            body_text = await translate(text, target_lang=target, source_lang="en-IN")
        except Exception:
            body_text = text  # silent fallback
    audio = await speak(body_text, speaker=speaker, lang=target)
    if not audio and target != "en-IN":
        # Final silent fallback: try English TTS so the user still gets audio.
        audio = await speak(text, speaker=speaker, lang="en-IN")
    return audio
