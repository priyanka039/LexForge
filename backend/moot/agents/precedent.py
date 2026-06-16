# ─────────────────────────────────────────────
# moot/agents/precedent.py
# On-demand case law researcher. Pulls from the
# project's ChromaDB corpus (real ingested
# judgments) — never invents authority.
# Output depth adapts to experience level.
# ─────────────────────────────────────────────

import asyncio

from .base import BaseAgent
from ..models import AgentResponse, ExperienceLevel, SessionState


class PrecedentAgent(BaseAgent):
    name = "Precedent"

    async def process(self, query: str, state: SessionState) -> AgentResponse:
        cfg = state.config
        search_query = f"{query} {cfg.case_statement}"[:400]

        try:
            from utils import search_chromadb
            # Real authorities only — exclude bare statutes/Constitution so the
            # bench is handed citeable case law, not provision text.
            chunks = await asyncio.to_thread(
                lambda: search_chromadb(
                    search_query, 4, exclude_doc_types=["statute"], min_score=0.30
                )
            )
        except Exception:
            chunks = []

        if not chunks:
            return AgentResponse(
                agent_name  = self.name,
                text        = "No matching authority found in the library for that point.",
                spoken_text = "No matching authority on record for that point.",
                metadata    = {"tts_voice": "neha"},
            )

        cases, lines = [], []
        seen = set()
        for c in chunks:
            meta = c.get("metadata", {})
            name = meta.get("case_name", "Unknown")
            if name in seen:
                continue
            seen.add(name)
            year    = meta.get("year", "")
            court   = meta.get("court", "")
            snippet = (c.get("text") or "")[:220].replace("\n", " ").strip()
            cases.append({
                "case_name": name,
                "year":      year,
                "court":     court,
                "holding":   snippet,
                "score":     c.get("score", 0),
            })
            if cfg.experience_level == ExperienceLevel.SENIOR:
                lines.append(f"{name} ({year})")                      # peer researcher: name only
            elif cfg.experience_level == ExperienceLevel.JUNIOR:
                lines.append(f"{name} ({year}, {court}) — {snippet[:90]}")
            else:
                lines.append(f"{name} ({year}, {court}) — {snippet}")  # student: full context

        text = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))

        # Spoken summary stays short regardless of level — reading holdings
        # aloud would stall the hearing.
        top = cases[0]
        spoken = f"On that point: {top['case_name']}"
        if top["year"]:
            spoken += f", {top['year']}"
        if len(cases) > 1:
            spoken += f". {len(cases) - 1} further authorities are on your research panel."
        else:
            spoken += ". It is on your research panel."

        state.cases_surfaced.extend(cases)

        return AgentResponse(
            agent_name         = self.name,
            text               = text,
            spoken_text        = spoken,
            citations          = [f"{c['case_name']} ({c['year']})" for c in cases],
            score_contribution = {"authority": +0.1},
            metadata           = {"cases": cases, "tts_voice": "neha"},
        )
