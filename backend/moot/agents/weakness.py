# ─────────────────────────────────────────────
# moot/agents/weakness.py
# Structural-gap auditor. Fires every N counsel
# exchanges (config). Output style depends on
# experience level; for seniors it never speaks
# during the session — findings appear only in
# the post-session debrief.
# ─────────────────────────────────────────────

from .base import BaseAgent
from ..config import language_directive, WEAKNESS_TRIGGER_INTERVAL, WEAKNESS_MIN_EXCHANGES
from ..models import AgentResponse, ExperienceLevel, SessionState

WEAKNESS_PROMPT_STUDENT = """You are a moot court coach reviewing a law student's live oral argument in an Indian moot.

FULL ARGUMENT SO FAR:
{full_argument}

PROVISIONS IN ISSUE: {statutes}
CITATIONS USED: {citations_used}
ALREADY FLAGGED (do not repeat): {already_flagged}

TASK: Identify the single MOST CRITICAL gap, then explain it for a student.
Format exactly:
"Your argument on [topic] has not addressed [specific gap]. A court would expect [specific test/principle/case]. Try: '[principle] was established in [case], and applies here because [link to facts].'"

One gap only. Educational, specific, no padding.{language_directive}"""

WEAKNESS_PROMPT_JUNIOR = """You are auditing a practising advocate's live oral argument in an Indian court for structural gaps.

ARGUMENT RECORD:
{full_argument}

PROVISIONS IN ISSUE: {statutes}
CITATIONS USED: {citations_used}
ALREADY FLAGGED (do not repeat): {already_flagged}

Identify ONE specific, actionable gap not previously flagged. Check for:
- Legal tests cited but not applied (proportionality, intelligible differentia, nexus, rarest of rare...)
- Provisions invoked but never argued
- Cases named but holdings never extracted
- The prayer not anchored in the argument
- Obvious counter-arguments not pre-empted

Format: "Your [issue/provision] argument has not addressed [specific gap]."
ONE sentence. Precise. No explanation — this is for a practitioner.{language_directive}"""

WEAKNESS_PROMPT_SENIOR = """You are conducting a strategic audit of a senior advocate's live oral argument in an Indian court.

ARGUMENT RECORD:
{full_argument}

PROVISIONS IN ISSUE: {statutes}
CITATIONS USED: {citations_used}
ALREADY FLAGGED (do not repeat): {already_flagged}

Identify the most strategically significant weakness — case theory, not technicality. Consider:
- Is the burden of proof actually being carried?
- Is the argument framed too broadly, inviting adverse consequences?
- Is the prayer properly anchored in law?
- Is there an unused concede-and-distinguish that would be stronger?

ONE sentence. Strategic, not pedagogical.{language_directive}"""


class WeaknessAgent(BaseAgent):
    name = "Weakness"

    PROMPTS = {
        ExperienceLevel.STUDENT: WEAKNESS_PROMPT_STUDENT,
        ExperienceLevel.JUNIOR:  WEAKNESS_PROMPT_JUNIOR,
        ExperienceLevel.SENIOR:  WEAKNESS_PROMPT_SENIOR,
    }

    @staticmethod
    def should_trigger(state: SessionState) -> bool:
        if not state.config.weakness_alerts:
            return False
        n = state.mooter_exchange_count
        return n >= WEAKNESS_MIN_EXCHANGES and n % WEAKNESS_TRIGGER_INTERVAL == 0

    def build_system_prompt(self, state: SessionState) -> str:
        cfg         = state.config
        mooter_args = [h["text"] for h in state.argument_history if h["role"] == "mooter"]
        full_text   = " | ".join(mooter_args[-12:])[:2500] or "No argument recorded yet."

        template = self.PROMPTS.get(cfg.experience_level, WEAKNESS_PROMPT_JUNIOR)
        return template.format(
            full_argument      = full_text,
            statutes           = ", ".join(cfg.relevant_statutes) or "not specified",
            citations_used     = ", ".join(state.citations_used[-10:]) or "none",
            already_flagged    = "; ".join(state.weaknesses_flagged[-3:]) or "none",
            language_directive = language_directive(cfg.language),
        )

    async def process(self, query: str, state: SessionState) -> AgentResponse:
        cfg = state.config
        max_tokens = 170 if cfg.experience_level == ExperienceLevel.STUDENT else 90

        text = await self._call_llm(
            system_prompt = self.build_system_prompt(state),
            user_message  = "Identify the next most important gap in the argument now.",
            max_tokens    = max_tokens,
            temperature   = 0.4,
        )
        if not text:
            return AgentResponse(agent_name=self.name)

        state.weaknesses_flagged.append(text)

        # Senior level: record the finding, but stay silent until the debrief.
        in_session_text = "" if cfg.experience_level == ExperienceLevel.SENIOR else text

        return AgentResponse(
            agent_name         = self.name,
            text               = in_session_text,
            spoken_text        = "",   # weakness notes are read, never spoken
            score_contribution = {"structure": -0.3},
        )
