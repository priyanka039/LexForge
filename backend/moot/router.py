# ─────────────────────────────────────────────
# moot/router.py
# Moot Chamber transport layer.
#
#   WS   /api/moot/ws                        live argument session
#   GET  /api/moot/meta                      judges / courts / languages
#   GET  /api/moot/session/{id}/debrief      scores + feedback + transcript
#   POST /api/moot/session/{id}/save-to-matter
#
# WebSocket protocol
#   client → server
#     JSON   {"type":"start_session","config":{...}}
#     BINARY WAV bytes of one complete utterance (16k mono PCM)
#     JSON   {"type":"interrupt"} | {"type":"end_session"}
#   server → client
#     JSON   {"event_type": "...", "data": {...}}
#     BINARY WAV bytes of TTS audio (always preceded by its agent_response)
# ─────────────────────────────────────────────

import asyncio
import json
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from sarvam import transcribe as sarvam_transcribe
from sarvam import translate as sarvam_translate
from sarvam import speak as sarvam_speak

from . import llm, session_store
from .normalize import legal_normalize
from .agents import Orchestrator
from .config import (
    COURT_ADDRESS_FORMS,
    DEFAULT_JUDGE_VOICE,
    JUDGE_VOICE_MAP,
    SUPPORTED_SESSION_LANGUAGES,
)
from .models import SessionConfig, SessionState
from .score import compute_final_score, generate_feedback

router = APIRouter(prefix="/api/moot", tags=["moot-chamber"])

_orchestrator = Orchestrator()


def _log(msg: str) -> None:
    try:
        print(f"[moot] {msg}", flush=True)
    except Exception:
        pass


# ═══ JUDGE OPENING LINES (templated — zero latency) ═══
_OPENING_LINES = {
    "verma":        "Yes, Counsel. I have read the petition with some care. You may open — and I would ask you to begin with the constitutional foundation of your claim.",
    "mehta":        "Counsel, before anything else — is this petition maintainable? Satisfy me on that first.",
    "krishnaswamy": "Counsel, I have read your papers. This matter raises questions this court takes seriously. Proceed.",
    "sinha":        "Counsel. I have read your petition. I am not persuaded it discloses what you say it discloses. Proceed — carefully.",
    "kaul":         "Counsel, I have your papers. Let us not spend the morning on theory — tell me what happened and what you want this court to do.",
}


async def _send_json(ws: WebSocket, event_type: str, data: dict) -> None:
    await ws.send_text(json.dumps({"event_type": event_type, "data": data}, ensure_ascii=False))


async def _tts(text: str, voice: str, lang: str) -> bytes:
    """Translate (if vernacular) + synthesize. Returns b'' on any failure."""
    if not text:
        return b""
    body = text
    if lang and lang != "en-IN" and _looks_english(text):
        # Agent responded in English despite a vernacular session (provider
        # fallback can cause this) — translate silently before speaking.
        body = await sarvam_translate(text, target_lang=lang, source_lang="en-IN")
    audio = await sarvam_speak(body, speaker=voice, lang=lang or "en-IN")
    if not audio and lang != "en-IN":
        audio = await sarvam_speak(text, speaker=voice, lang="en-IN")
    return audio


_SENTENCE_SPLIT = re.compile(r"(?<=[.?!।])\s+")


async def _stream_tts(ws: WebSocket, text: str, voice: str, lang: str) -> None:
    """
    Sentence-level TTS pipelining: split the reply into sentences,
    synthesize them CONCURRENTLY, and ship each WAV to the client as
    soon as it (and every sentence before it) is ready. The client
    starts hearing the first sentence while later ones are still
    being synthesized.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return
    if len(sentences) == 1:
        audio = await _tts(sentences[0], voice, lang)
        if audio:
            await ws.send_bytes(audio)
        else:
            _log(f"TTS produced no audio for voice={voice} lang={lang} — check SARVAM_API_KEY / voice name")
        return

    tasks = [asyncio.create_task(_tts(s, voice, lang)) for s in sentences]
    sent = 0
    for task in tasks:           # in-order delivery, out-of-order synthesis
        audio = await task
        if audio:
            await ws.send_bytes(audio)
            sent += 1
    if sent == 0:
        _log(f"TTS produced no audio across {len(sentences)} sentences (voice={voice} lang={lang})")


def _looks_english(text: str) -> bool:
    """Cheap script check: mostly-ASCII text is treated as English."""
    if not text:
        return True
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (non_ascii / max(len(text), 1)) < 0.15


# ═══ WEBSOCKET — THE CHAMBER ══════════════════

@router.websocket("/ws")
async def moot_websocket(ws: WebSocket):
    await ws.accept()
    state: SessionState | None = None
    busy = asyncio.Lock()   # one utterance processed at a time

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── JSON control frames ──────────
            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue
                mtype = payload.get("type", "")

                if mtype == "start_session":
                    try:
                        config = SessionConfig.model_validate(payload.get("config") or {})
                    except Exception as e:
                        await _send_json(ws, "error", {"message": f"Invalid configuration: {e}"})
                        continue
                    state = SessionState(config=config)
                    session_store.save(state)
                    _log(f"session {state.session_id[:8]} started — "
                         f"{config.judge_personality.value}/{config.court_level.value}/"
                         f"{config.experience_level.value}/{config.language} "
                         f"providers={llm.active_providers()}")

                    await _send_json(ws, "session_state", {
                        "session_id": state.session_id,
                        "providers":  llm.active_providers(),
                    })

                    # Judge calls the matter — templated, spoken immediately.
                    persona = config.judge_personality.value
                    opening = _OPENING_LINES.get(persona, _OPENING_LINES["sinha"])
                    voice   = JUDGE_VOICE_MAP.get(persona, DEFAULT_JUDGE_VOICE)
                    state.add_history("judge", opening)
                    session_store.save(state)
                    await _send_json(ws, "agent_response", {
                        "agent":     "Judge",
                        "text":      opening,
                        "spoken":    opening,
                        "citations": [],
                        "metadata":  {"judge_personality": persona, "opening": True},
                    })
                    await _stream_tts(ws, opening, voice, config.language)

                elif mtype == "end_session":
                    break

                elif mtype == "interrupt":
                    # TTS is request-based server-side; playback stops client-side.
                    pass

                continue

            # ── Binary frames: one WAV utterance ──
            audio_bytes = message.get("bytes")
            if not audio_bytes:
                continue
            if state is None:
                await _send_json(ws, "error", {"message": "Session not started."})
                continue

            async with busy:
                await _process_utterance(ws, state, audio_bytes)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        _log(f"ws error: {type(e).__name__}: {e}")
    finally:
        if state is not None:
            _finalize_session(state)


async def _process_utterance(ws: WebSocket, state: SessionState, audio_bytes: bytes) -> None:
    cfg = state.config
    sid = state.session_id[:8]
    t0 = time.perf_counter()
    _log(f"[{sid}] utterance received: {len(audio_bytes)} bytes WAV")

    # 1. STT
    await _send_json(ws, "agent_status", {"agent": "stt", "status": "processing"})
    raw = await sarvam_transcribe(
        audio_bytes = audio_bytes,
        filename    = "utterance.wav",
        mime        = "audio/wav",
        language    = cfg.language,
    )
    raw = (raw or "").strip()
    t_stt = time.perf_counter()
    if not raw:
        _log(f"[{sid}] STT returned EMPTY ({t_stt - t0:.2f}s) — "
             f"check SARVAM_API_KEY / audio not silent")
        await _send_json(ws, "transcript", {"text": "", "empty": True})
        return

    # Repair legal vocabulary and citations BEFORE agents see the text.
    transcript = legal_normalize(raw)
    if transcript != raw:
        _log(f"[{sid}] STT: {raw!r} -> normalized {transcript!r} ({t_stt - t0:.2f}s)")
    else:
        _log(f"[{sid}] STT: {transcript!r} ({t_stt - t0:.2f}s)")

    await _send_json(ws, "transcript", {"text": transcript})

    # 2. Orchestrate agents
    await _send_json(ws, "agent_status", {"agent": "judge", "status": "processing"})
    try:
        responses = await _orchestrator.handle_utterance(transcript, state)
    except Exception as e:
        _log(f"[{sid}] orchestrator error: {type(e).__name__}: {e}")
        await _send_json(ws, "error", {"message": "The bench is momentarily unavailable. Continue."})
        return
    t_llm = time.perf_counter()
    _log(f"[{sid}] agents -> {[r.agent_name for r in responses] or 'none'} "
         f"(LLM {t_llm - t_stt:.2f}s)")

    # 3. Stream responses (text first, then audio for spoken ones)
    for resp in responses:
        if resp.metadata.get("error") == "no_llm_response":
            _log(f"[{sid}] {resp.agent_name}: NO LLM RESPONSE — all providers failed")
            await _send_json(ws, "error", {
                "message": "No language model responded. Add an API key to .env or start Ollama.",
            })
            continue

        _log(f"[{sid}] {resp.agent_name}: {(resp.spoken_text or resp.text or '')[:80]!r}")
        await _send_json(ws, "agent_response", {
            "agent":     resp.agent_name,
            "text":      resp.text,
            "spoken":    resp.spoken_text,
            "citations": resp.citations,
            "metadata":  resp.metadata,
        })

        if resp.spoken_text:
            voice = resp.metadata.get("tts_voice") or JUDGE_VOICE_MAP.get(
                cfg.judge_personality.value, DEFAULT_JUDGE_VOICE
            )
            await _stream_tts(ws, resp.spoken_text, voice, cfg.language)

    _log(f"[{sid}] turn complete in {time.perf_counter() - t0:.2f}s "
         f"(STT {t_stt - t0:.2f} + LLM {t_llm - t_stt:.2f} + TTS {time.perf_counter() - t_llm:.2f})")

    # 4. Stats + persistence
    await _send_json(ws, "session_stats", {
        "exchanges":      state.mooter_exchange_count,
        "citations_used": len(state.citations_used),
        "flags":          len(state.citation_flags) + len(state.weaknesses_flagged),
    })
    session_store.save(state)


def _finalize_session(state: SessionState) -> None:
    """Score, persist to Redis and SQLite. Safe to call once per session."""
    from datetime import datetime, timezone
    if not state.is_active:
        return
    state.is_active = False
    state.ended_at  = datetime.now(timezone.utc).isoformat()
    state.score     = compute_final_score(state)
    session_store.save(state)

    # Persist a permanent record in SQLite (works with or without a matter).
    try:
        from database import save_session
        feedback = generate_feedback(state)
        save_session(
            session_type = "moot_chamber",
            title        = f"Moot: {state.config.case_name[:100] or 'Untitled matter'}",
            input_data   = state.config.model_dump(mode="json"),
            output_data  = {
                "session_id":     state.session_id,
                "score":          state.score.model_dump(),
                "overall":        state.score.overall(),
                "feedback":       feedback,
                "transcript":     state.argument_history,
                "citations_used": state.citations_used,
                "cases_surfaced": state.cases_surfaced,
                "weaknesses":     state.weaknesses_flagged,
                "citation_flags": state.citation_flags,
                "duration_seconds": state.duration_seconds(),
            },
            case_id = state.config.matter_id,
        )
        _log(f"session {state.session_id[:8]} finalised — overall {state.score.overall()}")
    except Exception as e:
        _log(f"sqlite persist failed: {type(e).__name__}: {e}")


# ═══ REST ═════════════════════════════════════

@router.get("/meta")
def moot_meta():
    """Everything the setup screen needs in one call."""
    return {
        "judges": [
            {"id": "verma",        "name": "Justice A.K. Verma",      "style": "The Constitutional Philosopher",
             "best_for": "Constitutional law, fundamental rights, PIL"},
            {"id": "mehta",        "name": "Justice S.K. Mehta",      "style": "The Technocrat",
             "best_for": "Civil procedure, evidence, jurisdiction"},
            {"id": "sinha",        "name": "Justice R.P. Sinha",      "style": "The Skeptic",
             "best_for": "Hard cases, criminal law, competition prep"},
            {"id": "krishnaswamy", "name": "Justice M. Krishnaswamy", "style": "The Activist",
             "best_for": "PIL, human rights, environment, minorities"},
            {"id": "kaul",         "name": "Justice P. Kaul",         "style": "The Pragmatist",
             "best_for": "Commercial, property, service, contract"},
        ],
        "court_levels": [
            {"id": "district",   "name": "District Court",  "desc": "Fundamental skills. Strict procedure.",
             "address": COURT_ADDRESS_FORMS["district"]["judge"]},
            {"id": "high_court", "name": "High Court",       "desc": "Constitutional arguments. Bench intervention.",
             "address": COURT_ADDRESS_FORMS["high_court"]["judge"]},
            {"id": "supreme",    "name": "Supreme Court",    "desc": "Constitutional morality. Jurisprudential depth.",
             "address": COURT_ADDRESS_FORMS["supreme"]["judge"]},
        ],
        "experience_levels": [
            {"id": "student", "name": "Law Student",     "desc": "Competition prep. Learning the ropes."},
            {"id": "junior",  "name": "Junior Advocate", "desc": "Sharpening submissions. Building instinct."},
            {"id": "senior",  "name": "Senior Advocate", "desc": "Full intensity. No quarter."},
        ],
        "languages": SUPPORTED_SESSION_LANGUAGES,
        "providers": llm.active_providers(),
    }


@router.get("/session/{session_id}/debrief")
def session_debrief(session_id: str):
    state = session_store.load(session_id)
    if state is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    if state.is_active:
        # Debrief requested before the socket closed — finalise now.
        _finalize_session(state)

    score    = state.score
    feedback = generate_feedback(state)
    return {
        "session_id":       state.session_id,
        "case_name":        state.config.case_name,
        "duration_seconds": state.duration_seconds(),
        "exchange_count":   state.mooter_exchange_count,
        "score": {
            "structure":      score.structure,
            "authority":      score.authority,
            "responsiveness": score.responsiveness,
            "precision":      score.precision,
            "coherence":      score.coherence,
            "overall":        score.overall(),
        },
        "feedback":       feedback,
        "transcript":     state.argument_history,
        "citations_used": state.citations_used,
        "cases_surfaced": state.cases_surfaced,
        "weaknesses":     state.weaknesses_flagged,
        "citation_flags": state.citation_flags,
    }


@router.post("/session/{session_id}/save-to-matter")
def save_to_matter(session_id: str, matter_id: int):
    state = session_store.load(session_id)
    if state is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    try:
        from database import get_case, save_session
        if get_case(matter_id) is None:
            return JSONResponse({"error": "Matter not found"}, status_code=404)
        feedback = generate_feedback(state)
        record = save_session(
            session_type = "moot_chamber",
            title        = f"Moot: {state.config.case_name[:100] or 'Untitled matter'}",
            input_data   = state.config.model_dump(mode="json"),
            output_data  = {
                "session_id": state.session_id,
                "score":      state.score.model_dump(),
                "overall":    state.score.overall(),
                "feedback":   feedback,
                "transcript": state.argument_history,
            },
            case_id = matter_id,
        )
        return {"ok": True, "session_record_id": record["id"]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
