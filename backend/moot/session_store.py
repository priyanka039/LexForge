# ─────────────────────────────────────────────
# moot/session_store.py
# Session persistence: Redis when reachable,
# transparent in-memory fallback when not.
# Redis is probed once at first use; if it goes
# away mid-session we degrade silently.
# ─────────────────────────────────────────────

from __future__ import annotations

from typing import Optional

from .config import REDIS_URL, SESSION_TTL_SECONDS
from .models import SessionState

_KEY_PREFIX = "moot:session:"

# In-memory fallback store (also a write-through cache for active sessions).
_memory: dict[str, SessionState] = {}

_redis = None
_redis_checked = False


def _log(msg: str) -> None:
    try:
        print(f"[moot-store] {msg}", flush=True)
    except Exception:
        pass


def _get_redis():
    """Lazily connect to Redis. Returns None if unavailable."""
    global _redis, _redis_checked
    if _redis_checked:
        return _redis
    _redis_checked = True
    try:
        import redis  # type: ignore
        client = redis.Redis.from_url(
            REDIS_URL,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
            decode_responses=True,
            protocol=2,   # old Redis builds (e.g. 5.x on Windows) lack RESP3 HELLO
        )
        client.ping()
        _redis = client
        _log(f"Redis connected at {REDIS_URL}")
    except Exception as e:
        _redis = None
        _log(f"Redis unavailable ({type(e).__name__}) — using in-memory store")
    return _redis


def save(state: SessionState) -> None:
    _memory[state.session_id] = state
    r = _get_redis()
    if r is not None:
        try:
            r.setex(
                _KEY_PREFIX + state.session_id,
                SESSION_TTL_SECONDS,
                state.model_dump_json(),
            )
        except Exception as e:
            _log(f"Redis save failed ({type(e).__name__}) — memory copy kept")


def load(session_id: str) -> Optional[SessionState]:
    if session_id in _memory:
        return _memory[session_id]
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(_KEY_PREFIX + session_id)
            if raw:
                state = SessionState.model_validate_json(raw)
                _memory[session_id] = state
                return state
        except Exception as e:
            _log(f"Redis load failed ({type(e).__name__})")
    return None


def drop_from_memory(session_id: str) -> None:
    """Free the in-memory copy once a session is finished (Redis keeps it for the debrief)."""
    save_state = _memory.pop(session_id, None)
    # Make sure the final state made it to Redis before dropping.
    if save_state is not None:
        r = _get_redis()
        if r is None:
            # No Redis — keep it in memory after all, or the debrief breaks.
            _memory[session_id] = save_state
