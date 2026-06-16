# ─────────────────────────────────────────────
# moot/models.py
# Pydantic v2 models for the Moot Chamber.
# SessionState is fully JSON-serialisable so it
# round-trips through Redis without surprises.
# ─────────────────────────────────────────────

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .config import SCORE_WEIGHTS


class JudgePersonality(str, Enum):
    VERMA        = "verma"          # The Constitutional Philosopher
    MEHTA        = "mehta"          # The Technocrat
    KRISHNASWAMY = "krishnaswamy"   # The Activist
    SINHA        = "sinha"          # The Skeptic
    KAUL         = "kaul"           # The Pragmatist


class CourtLevel(str, Enum):
    DISTRICT   = "district"
    HIGH_COURT = "high_court"
    SUPREME    = "supreme"


class ExperienceLevel(str, Enum):
    STUDENT = "student"
    JUNIOR  = "junior"
    SENIOR  = "senior"


class Intent(str, Enum):
    JUDGE            = "judge"
    PRECEDENT_SEARCH = "precedent_search"
    COUNTER_ARGUMENT = "counter_argument"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentResponse(BaseModel):
    agent_name:         str
    text:               str = ""
    spoken_text:        str = ""
    citations:          List[str] = Field(default_factory=list)
    score_contribution: dict = Field(default_factory=dict)   # {dimension: delta}
    metadata:           dict = Field(default_factory=dict)


class SessionScore(BaseModel):
    structure:      float = 5.0
    authority:      float = 5.0
    responsiveness: float = 5.0
    precision:      float = 5.0
    coherence:      float = 5.0

    def overall(self) -> float:
        return round(
            self.structure      * SCORE_WEIGHTS["structure"]
            + self.authority      * SCORE_WEIGHTS["authority"]
            + self.responsiveness * SCORE_WEIGHTS["responsiveness"]
            + self.precision      * SCORE_WEIGHTS["precision"]
            + self.coherence      * SCORE_WEIGHTS["coherence"],
            1,
        )


class SessionConfig(BaseModel):
    case_name:                str = ""
    side_arguing:             str = "petitioner"
    case_statement:           str = ""
    relevant_statutes:        List[str] = Field(default_factory=list)
    court_level:              CourtLevel = CourtLevel.HIGH_COURT
    judge_personality:        JudgePersonality = JudgePersonality.SINHA
    experience_level:         ExperienceLevel = ExperienceLevel.JUNIOR
    language:                 str = "en-IN"
    matter_id:                Optional[int] = None
    silent_citation_checking: bool = True
    weakness_alerts:          bool = True
    show_transcript:          bool = True
    # When true, opposing counsel proactively attacks every substantive
    # submission (not only when explicitly summoned). This makes the chamber
    # a genuine two-on-one: the bench questions AND the other side debates.
    opposing_counsel_active:  bool = True


class SessionState(BaseModel):
    session_id:            str = Field(default_factory=lambda: str(uuid.uuid4()))
    config:                SessionConfig = Field(default_factory=SessionConfig)
    # argument_history items: {role, text, ts, score_contribution?}
    argument_history:      List[dict] = Field(default_factory=list)
    citations_used:        List[str]  = Field(default_factory=list)
    cases_surfaced:        List[dict] = Field(default_factory=list)
    weaknesses_flagged:    List[str]  = Field(default_factory=list)
    citation_flags:        List[dict] = Field(default_factory=list)
    score:                 SessionScore = Field(default_factory=SessionScore)
    exchange_count:        int  = 0
    mooter_exchange_count: int  = 0
    is_active:             bool = True
    created_at:            str = Field(default_factory=_utcnow)
    ended_at:              Optional[str] = None

    # ── helpers ──────────────────────────────
    def add_history(self, role: str, text: str, score_contribution: dict | None = None):
        entry: dict[str, Any] = {"role": role, "text": text, "ts": _utcnow()}
        if score_contribution:
            entry["score_contribution"] = score_contribution
        self.argument_history.append(entry)
        self.exchange_count += 1
        if role == "mooter":
            self.mooter_exchange_count += 1

    def recent_history_text(self, n: int = 8, max_chars: int = 250) -> str:
        recent = self.argument_history[-n:]
        if not recent:
            return "No prior exchanges. This is the opening of arguments."
        lines = []
        for h in recent:
            who = "JUDGE" if h["role"] == "judge" else "COUNSEL"
            lines.append(f"{who}: {h['text'][:max_chars]}")
        return "\n".join(lines)

    def duration_seconds(self) -> int:
        try:
            start = datetime.fromisoformat(self.created_at)
            end   = datetime.fromisoformat(self.ended_at) if self.ended_at else datetime.now(timezone.utc)
            return max(0, int((end - start).total_seconds()))
        except Exception:
            return 0
