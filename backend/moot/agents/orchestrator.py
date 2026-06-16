# ─────────────────────────────────────────────
# moot/agents/orchestrator.py
# Routes each counsel utterance to the right
# agents and updates session state.
#
# Routing is deliberately heuristic (regex), not
# an LLM call: it is instant, free, predictable,
# and never burns a rate-limited request just to
# decide who should answer.
#
# Per utterance:
#   1. Citation agent   (always, local, silent)
#   2. Primary agent    (judge | precedent | counter)
#   3. Weakness agent   (every N counsel exchanges)
# ─────────────────────────────────────────────

import asyncio
import re

from .citation import CitationAgent
from .counter import CounterAgent
from .judge import JudgeAgent
from .precedent import PrecedentAgent
from .weakness import WeaknessAgent
from ..models import AgentResponse, Intent, SessionState

# Explicit research requests — addressed to the system, not the bench.
_PRECEDENT_RX = re.compile(
    r"(case\s*law|precedent|authorit|cite\s+me|cases?\s+on|give\s+me\s+(a\s+)?cases?"
    r"|find\s+(me\s+)?(a\s+)?cases?|judgments?\s+on|ruling\s+on"
    r"|कोई\s*केस|केस\s*(बताओ|बताइए|दो|दीजिए)|नज़ीर|उदाहरण\s*दो"
    r"|case\s*(batao|bataiye|do|dijiye)|nazeer|nazir)",
    re.IGNORECASE,
)

# Explicit requests to hear the other side.
_COUNTER_RX = re.compile(
    r"(what\s+would\s+(the\s+)?(other\s+side|opposition|respondent|petitioner|state)\s+(say|argue)"
    r"|counter[\s-]*argument|opposing\s+counsel|steelman|their\s+best\s+(argument|case)"
    r"|argue\s+against\s+me|push\s+back"
    r"|विरोधी\s*(पक्ष|वकील)|दूसरा\s*पक्ष|सामने\s*वाले?"
    r"|virodhi|doosra\s*paksh|saamne\s*wal[ae])",
    re.IGNORECASE,
)


def classify_intent(text: str) -> Intent:
    if _PRECEDENT_RX.search(text):
        return Intent.PRECEDENT_SEARCH
    if _COUNTER_RX.search(text):
        return Intent.COUNTER_ARGUMENT
    return Intent.JUDGE


class Orchestrator:
    def __init__(self):
        self.judge     = JudgeAgent()
        self.precedent = PrecedentAgent()
        self.counter   = CounterAgent()
        self.citation  = CitationAgent()
        self.weakness  = WeaknessAgent()

    async def handle_utterance(self, transcript: str, state: SessionState) -> list[AgentResponse]:
        """
        Process one counsel utterance. Returns the ordered agent responses
        (caller streams each to the client). State is mutated in place;
        the caller persists it.
        """
        responses: list[AgentResponse] = []
        transcript = (transcript or "").strip()
        if not transcript:
            return responses

        # 1. Citation check BEFORE history append (works on the raw utterance).
        citation_resp = await self.citation.process(transcript, state)
        if citation_resp.text:
            responses.append(citation_resp)
        self._apply_score(citation_resp, state)

        # 2. Record counsel's words.
        state.add_history("mooter", transcript)

        # 3. Primary agent + (optional) opposing counsel + weakness audit all
        #    run CONCURRENTLY. None of them need each other's output for this
        #    turn, so running them in parallel keeps latency to a single LLM
        #    round-trip instead of stacking them.
        intent = classify_intent(transcript)
        if intent == Intent.PRECEDENT_SEARCH:
            primary_task = self.precedent.process(transcript, state)
        elif intent == Intent.COUNTER_ARGUMENT:
            primary_task = self.counter.process(transcript, state)
        else:
            primary_task = self.judge.process(transcript, state)

        tasks  = [primary_task]
        labels = ["primary"]

        # Opposing counsel proactively attacks substantive submissions to the
        # bench — a real moot is two-on-one. Skip on short/procedural utterances
        # and when the user explicitly asked for precedent/counter (already handled).
        run_counter = (
            state.config.opposing_counsel_active
            and intent == Intent.JUDGE
            and self._is_substantive(transcript)
        )
        if run_counter:
            tasks.append(self.counter.process(transcript, state))
            labels.append("counter")

        run_weakness = self.weakness.should_trigger(state)
        if run_weakness:
            tasks.append(self.weakness.process(transcript, state))
            labels.append("weakness")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        by_label = dict(zip(labels, results))

        primary = by_label["primary"]
        if isinstance(primary, Exception):
            raise primary
        if primary.text or primary.metadata.get("error"):
            responses.append(primary)
        self._apply_score(primary, state)

        if primary.agent_name == "Judge" and primary.text:
            state.add_history(
                "judge", primary.text,
                score_contribution=primary.score_contribution or None,
            )
        elif primary.text:
            state.add_history(primary.agent_name.lower(), primary.text)

        # Opposing counsel speaks after the bench, then is recorded so the
        # mooter (and later agents) can engage with what was actually said.
        counter = by_label.get("counter")
        if counter is not None and not isinstance(counter, Exception) and counter.text:
            responses.append(counter)
            self._apply_score(counter, state)
            state.add_history("counter", counter.text)

        weak = by_label.get("weakness")
        if weak is not None and not isinstance(weak, Exception) and weak.text:
            responses.append(weak)
            self._apply_score(weak, state)

        return responses

    @staticmethod
    def _is_substantive(text: str) -> bool:
        """A real submission worth attacking — not 'yes my lord' / 'as I said'."""
        words = text.split()
        if len(words) < 8:
            return False
        filler = {"yes", "no", "thank", "thanks", "okay", "ok", "right", "my", "lord", "lordship"}
        meaningful = [w for w in words if w.lower().strip(".,") not in filler]
        return len(meaningful) >= 6

    @staticmethod
    def _apply_score(resp: AgentResponse, state: SessionState) -> None:
        """Apply bounded score deltas as the session progresses."""
        for dim, delta in (resp.score_contribution or {}).items():
            if hasattr(state.score, dim):
                cur = getattr(state.score, dim)
                setattr(state.score, dim, max(1.0, min(9.5, cur + delta)))
