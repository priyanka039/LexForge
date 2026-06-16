# ─────────────────────────────────────────────
# moot/agents/counter.py
# Opposing counsel. Steelmans the other side,
# grounded in real precedent retrieved from the
# project's ChromaDB corpus where possible.
# ─────────────────────────────────────────────

import asyncio

from .base import BaseAgent
from ..config import language_directive
from ..models import AgentResponse, ExperienceLevel, SessionState

COUNTER_PROMPT_STUDENT = """You are opposing counsel in an Indian moot court. The mooter is a law student preparing for competition. You argue for the {opposing_side} in {case_name}.

THE MOOTER JUST ARGUED: {last_argument}
PROVISIONS IN ISSUE: {statutes}
AUTHORITIES FROM THE RECORD (may be relevant):
{context}

Deliver a counter-argument in exactly 3 sentences:
1. Name the flaw in their argument — clearly and educationally.
2. State the counter-principle with a real Indian case (prefer one from the authorities above; otherwise a well-known Supreme Court case).
3. State how this undermines their submission.

This is for learning: be clear and instructional, but still adversarial. Never invent citations.{language_directive}"""

COUNTER_PROMPT_JUNIOR = """You are opposing counsel in a {court_level} matter in India. The mooter appears for the {side_arguing}; you appear for the {opposing_side} in {case_name}.

THEIR ARGUMENT: {last_argument}
THEIR CITATIONS SO FAR: {citations_used}
PROVISIONS IN ISSUE: {statutes}
AUTHORITIES FROM THE RECORD (may be relevant):
{context}

Deliver your counter-argument in 3 sentences:
1. The precise flaw or over-reach in their submission.
2. Your best authority that cuts against their position — a real Indian precedent (prefer one from the record above). Never invent citations.
3. The consequence for their case.

Be adversarial. No hints, no validation, no softening.{language_directive}"""

COUNTER_PROMPT_SENIOR = """You are senior opposing counsel in a {court_level} matter in India. The mooter appears for the {side_arguing} in {case_name}; you appear for the {opposing_side}.

THEIR ARGUMENT: {last_argument}
THEIR CITATIONS SO FAR: {citations_used}
PROVISIONS IN ISSUE: {statutes}
AUTHORITIES FROM THE RECORD (may be relevant):
{context}

Deliver a counter-argument a senior advocate would actually make in court. Go beyond the obvious:
- Concede-and-distinguish: "Even taking their proposition at its highest, it does not apply because..."
- Second-order consequences of accepting their argument
- Jurisdictional or procedural traps they have left open
- Authority they have conspicuously NOT cited that damages them

3-4 sentences. Analytically rigorous. Real Indian cases only — never invent citations.{language_directive}"""


class CounterAgent(BaseAgent):
    name = "Counter"

    PROMPTS = {
        ExperienceLevel.STUDENT: COUNTER_PROMPT_STUDENT,
        ExperienceLevel.JUNIOR:  COUNTER_PROMPT_JUNIOR,
        ExperienceLevel.SENIOR:  COUNTER_PROMPT_SENIOR,
    }

    COURT_NAMES = {
        "district":   "District Court",
        "high_court": "High Court",
        "supreme":    "Supreme Court",
    }

    async def _retrieve_context(self, query: str) -> str:
        """Ground the counter in the actual corpus. Soft-fails to empty."""
        try:
            from utils import search_chromadb, build_context
            chunks = await asyncio.to_thread(
                lambda: search_chromadb(
                    query, 3, exclude_doc_types=["statute"], min_score=0.30
                )
            )
            return build_context(chunks)[:1800] or "none on record"
        except Exception:
            return "none on record"

    async def process(self, query: str, state: SessionState) -> AgentResponse:
        cfg      = state.config
        opposing = "respondent" if cfg.side_arguing in ("petitioner", "appellant") else "petitioner"
        last_arg = next(
            (h["text"] for h in reversed(state.argument_history) if h["role"] == "mooter"),
            query,
        )

        context = await self._retrieve_context(f"{cfg.case_statement} {last_arg}"[:400])
        template = self.PROMPTS.get(cfg.experience_level, COUNTER_PROMPT_JUNIOR)

        system = template.format(
            case_name          = cfg.case_name or "the present matter",
            side_arguing       = cfg.side_arguing,
            opposing_side      = opposing,
            last_argument      = last_arg[:600],
            statutes           = ", ".join(cfg.relevant_statutes) or "the provisions under discussion",
            citations_used     = ", ".join(state.citations_used[-8:]) or "none cited yet",
            court_level        = self.COURT_NAMES.get(cfg.court_level.value, "High Court"),
            context            = context,
            language_directive = language_directive(cfg.language),
        )

        max_tokens = 320 if cfg.experience_level == ExperienceLevel.SENIOR else 220
        text = await self._call_llm(
            system_prompt = system,
            user_message  = f"Counter this argument now, as opposing counsel on your feet: {query}",
            max_tokens    = max_tokens,
            temperature   = 0.7,
        )

        if not text:
            return AgentResponse(agent_name=self.name, metadata={"error": "no_llm_response"})

        return AgentResponse(
            agent_name         = self.name,
            text               = text,
            spoken_text        = self._truncate_for_speech(text, 4),
            score_contribution = {"coherence": +0.2},
            metadata           = {"tts_voice": "priya"},   # opposing counsel voice
        )
