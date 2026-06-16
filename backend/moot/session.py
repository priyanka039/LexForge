# ─────────────────────────────────────────────
# moot/session.py
# Session store: Redis-backed when available,
# transparent in-memory fallback otherwise.
# Redis is NEVER a hard dependency — the chamber
# must work on a machine with nothing but Python.
# ─────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .config import REDIS_URL, SESSION_TTL_SECONDS
from .models import SessionState


def _log(msg: str) -> None:
    print(f"[moot.session] {msg}", flush=True)


class SessionStore:
    """
    get/put keyed by session_id. Values are SessionState serialized
    as JSON. The in-memory dict also acts as a write-through cache so
    a flaky Redis mid-session does not lose the argument record.
    """

    _KEY_PREFIX = "moot:session:"

    def __init__(self):
        self._memory: dict[str, str] = {}
        self._redis = None
        self._redis_failed = False

    async def _get_redis(self):
        if self._redis is not None or self._redis_failed:
            return self._redis
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(
                REDIS_URL, decode_responses=True,
                socket_connect_timeout=2, socket_timeout=2,
            )
            await client.ping()
            self._redis = client
            _log(f"connected to Redis at {REDIS_URL}")
        except Exception as e:
            self._redis_failed = True
            _log(f"Redis unavailable ({type(e).__name__}) — using in-memory store")
        return self._redis

    async def save(self, state: SessionState) -> None:
        raw = state.model_dump_json()
        self._memory[state.session_id] = raw
        r = await self._get_redis()
        if r is not None:
            try:
                await r.set(self._KEY_PREFIX + state.session_id, raw, ex=SESSION_TTL_SECONDS)
            except Exception as e:
                _log(f"Redis save failed ({type(e).__name__}) — memory copy retained")

    async def load(self, session_id: str) -> Optional[SessionState]:
        raw = self._memory.get(session_id)
        if raw is None:
            r = await self._get_redis()
            if r is not None:
                try:
                    raw = await r.get(self._KEY_PREFIX + session_id)
                except Exception:
                    raw = None
        if not raw:
            return None
        try:
            return SessionState.model_validate_json(raw)
        except Exception as e:
            _log(f"failed to deserialize session {session_id}: {e}")
            return None

    async def end(self, session_id: str) -> Optional[SessionState]:
        state = await self.load(session_id)
        if state is None:
            return None
        state.is_active = False
        state.ended_at = datetime.now(timezone.utc)
        await self.save(state)
        return state


# Module-level singleton — one store per process.
_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
