# ─────────────────────────────────────────────
# moot/agents/citation.py
# Silent citation checker — pure Python, no LLM,
# so it costs nothing and never adds latency.
#
# - Well-formed citations are recorded silently.
# - Malformed reporter references are flagged:
#     student  → flag + explanation of the format
#     junior   → flag only
#     senior   → flag only when materially wrong
# ─────────────────────────────────────────────

import re

from .base import BaseAgent
from ..config import CITATION_PATTERNS, CITATION_LOOSE_PATTERN
from ..models import AgentResponse, ExperienceLevel, SessionState

_EXACT = [re.compile(p, re.IGNORECASE) for p in CITATION_PATTERNS]
_LOOSE = re.compile(CITATION_LOOSE_PATTERN, re.IGNORECASE)

_FORMAT_HELP = (
    "Correct formats — Supreme Court: (Year) Volume SCC Page, e.g. (2017) 10 SCC 1; "
    "or AIR Year SC Page, e.g. AIR 1978 SC 597."
)


class CitationAgent(BaseAgent):
    name = "Citation"

    async def process(self, query: str, state: SessionState) -> AgentResponse:
        cfg = state.config

        # 1. Record well-formed citations.
        valid_spans = []
        for rx in _EXACT:
            for m in rx.finditer(query):
                cite = re.sub(r"\s+", " ", m.group(0)).strip()
                valid_spans.append((m.start(), m.end()))
                if cite not in state.citations_used:
                    state.citations_used.append(cite)

        if not cfg.silent_citation_checking:
            return AgentResponse(agent_name=self.name)

        # 2. Flag loose reporter mentions that did not match a valid format.
        flags = []
        for m in _LOOSE.finditer(query):
            inside_valid = any(s <= m.start() < e for s, e in valid_spans)
            if inside_valid:
                continue
            fragment = re.sub(r"\s+", " ", m.group(0)).strip()
            # Only flag if it actually carries digits — a bare "SCC" in prose
            # ("reported in the SCC") is not a citation attempt.
            if not re.search(r"\d", fragment):
                continue
            flags.append(fragment[:60])

        if not flags:
            return AgentResponse(agent_name=self.name)

        # Senior advocates only hear about material problems; a slightly odd
        # format is not worth the interruption.
        if cfg.experience_level == ExperienceLevel.SENIOR and len(flags) < 2:
            return AgentResponse(agent_name=self.name)

        msgs = []
        for f in flags[:2]:
            msg = f"Verify citation: \u201c{f}\u201d does not match a standard reporter format."
            if cfg.experience_level == ExperienceLevel.STUDENT:
                msg += " " + _FORMAT_HELP
            msgs.append(msg)
            state.citation_flags.append({"fragment": f, "message": msg})

        return AgentResponse(
            agent_name         = self.name,
            text               = " ".join(msgs),
            spoken_text        = "",   # silent — UI note only, never spoken
            score_contribution = {"precision": -0.4 * len(msgs)},
        )
