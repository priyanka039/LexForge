# ─────────────────────────────────────────────
# moot/agents/base.py
# ─────────────────────────────────────────────

from .. import llm
from ..models import AgentResponse, SessionState


class BaseAgent:
    name = "Agent"

    async def process(self, query: str, state: SessionState) -> AgentResponse:
        raise NotImplementedError

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 250,
        temperature: float = 0.6,
    ) -> str:
        return await llm.generate(
            system=system_prompt,
            user=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _truncate_for_speech(text: str, max_sentences: int = 3) -> str:
        """Keep spoken output short — TTS reads everything it gets."""
        if not text:
            return ""
        parts, count, out = text.replace("…", "."), 0, []
        for sentence in parts.split(". "):
            out.append(sentence)
            count += 1
            if count >= max_sentences:
                break
        spoken = ". ".join(out).strip()
        if spoken and not spoken.endswith((".", "?", "!", "।")):
            spoken += "."
        return spoken
