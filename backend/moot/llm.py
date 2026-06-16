# ─────────────────────────────────────────────
# moot/llm.py
# Provider-agnostic LLM client for live agents.
#
# Priority chain (first available wins, failures
# fall through to the next):
#   1. Anthropic   (ANTHROPIC_API_KEY)
#   2. OpenAI      (OPENAI_API_KEY)
#   3. Gemini      (GEMINI_API_KEY / GOOGLE_API_KEY)
#   4. Ollama      (local qwen3:8b — always last)
#
# All cloud calls go through httpx — no vendor
# SDKs required. Nothing here raises: on total
# failure an empty string is returned and the
# caller decides what to do.
# ─────────────────────────────────────────────

from __future__ import annotations

import asyncio
import os
import re

import httpx

from .config import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    GEMINI_MODEL,
    LLM_TIMEOUT_SECONDS,
)


def _log(msg: str) -> None:
    try:
        print(f"[moot-llm] {msg}", flush=True)
    except Exception:
        pass


def _clean(text: str) -> str:
    """Strip reasoning tags, markdown fences, and stage directions."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip())
    # Models sometimes emit stage directions like *leans forward* — strip them.
    text = re.sub(r"\*[^*\n]{1,60}\*", "", text)
    return text.strip()


# ═══ PROVIDERS ════════════════════════════════

async def _call_anthropic(system: str, user: str, max_tokens: int, temperature: float) -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no key")
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":       ANTHROPIC_MODEL,
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "system":      system,
                "messages":    [{"role": "user", "content": user}],
            },
        )
    r.raise_for_status()
    blocks = r.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


async def _call_openai(system: str, user: str, max_tokens: int, temperature: float) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no key")
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model":       OPENAI_MODEL,
                "max_tokens":  max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
        )
    r.raise_for_status()
    choices = r.json().get("choices", [])
    return choices[0]["message"]["content"] if choices else ""


async def _call_gemini(system: str, user: str, max_tokens: int, temperature: float) -> str:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("no key")
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            headers={"x-goog-api-key": key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature":     temperature,
                },
            },
        )
    r.raise_for_status()
    candidates = r.json().get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


async def _call_ollama(system: str, user: str, max_tokens: int, temperature: float) -> str:
    # Reuse the project's existing sync Ollama helper off the event loop.
    from utils import call_qwen
    return await asyncio.to_thread(
        call_qwen, user, max_tokens=max_tokens, system=system
    )


_CHAIN = [
    ("anthropic", _call_anthropic),
    ("openai",    _call_openai),
    ("gemini",    _call_gemini),
    ("ollama",    _call_ollama),
]


async def generate(
    system: str,
    user: str,
    max_tokens: int = 300,
    temperature: float = 0.6,
) -> str:
    """
    Run the provider chain. Returns clean text or "" if every provider failed.
    """
    for name, fn in _CHAIN:
        try:
            raw = await fn(system, user, max_tokens, temperature)
            text = _clean(raw)
            if text:
                return text
            _log(f"{name}: empty response, trying next provider")
        except RuntimeError:
            continue   # key not configured — silent skip
        except Exception as e:
            _log(f"{name}: {type(e).__name__}: {str(e)[:200]} — trying next provider")
    _log("all providers failed")
    return ""


def active_providers() -> list[str]:
    """Names of providers that have keys configured (Ollama always listed)."""
    out = []
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        out.append(f"anthropic:{ANTHROPIC_MODEL}")
    if os.getenv("OPENAI_API_KEY", "").strip():
        out.append(f"openai:{OPENAI_MODEL}")
    if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip():
        out.append(f"gemini:{GEMINI_MODEL}")
    out.append("ollama:qwen3:8b")
    return out
