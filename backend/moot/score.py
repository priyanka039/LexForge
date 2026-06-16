# ─────────────────────────────────────────────
# moot/score.py
# Five-dimension session scoring + debrief text.
# Computed from agent metadata accumulated during
# the session — no ML, fully deterministic.
# ─────────────────────────────────────────────

from .models import SessionScore, SessionState


def compute_final_score(state: SessionState) -> SessionScore:
    """Final scores at session end. Blends live deltas with end-state heuristics."""
    score = SessionScore()
    total = state.mooter_exchange_count
    if total == 0:
        return score

    # STRUCTURE — penalised by weaknesses flagged
    weakness_count = len(state.weaknesses_flagged)
    score.structure = max(2.0, 8.0 - min(weakness_count * 0.5, 3.0))

    # AUTHORITY — did counsel actually use the cases that were available?
    cited    = len(state.citations_used)
    surfaced = len(state.cases_surfaced)
    if surfaced > 0:
        coverage = min(cited / surfaced, 1.0)
        score.authority = round(4.0 + coverage * 4.0, 1)
    else:
        score.authority = 6.0 if cited >= 2 else (5.0 if cited else 3.0)

    # PRECISION — penalised by citation flags
    score.precision = max(2.0, 8.5 - min(len(state.citation_flags) * 0.8, 4.0))

    # RESPONSIVENESS — walk the judge's reactions across the session
    responsiveness = 5.0
    for h in state.argument_history:
        delta = (h.get("score_contribution") or {}).get("responsiveness")
        if delta:
            responsiveness = max(2.0, min(9.0, responsiveness + delta))
    score.responsiveness = round(responsiveness, 1)

    # COHERENCE — more exchanges with fewer weaknesses = more coherent
    coherence_ratio = 1.0 - (weakness_count / max(total, 1))
    score.coherence = round(max(3.0, min(9.0, coherence_ratio * 8.0 + 1.0)), 1)

    score.structure = round(score.structure, 1)
    score.precision = round(score.precision, 1)
    return score


def generate_feedback(state: SessionState) -> dict:
    """Human-readable feedback for the debrief screen."""
    score   = state.score
    overall = score.overall()
    cited   = state.citations_used
    surfaced = state.cases_surfaced

    level_text = {
        "student": "for a competition mooter",
        "junior":  "for a junior advocate",
        "senior":  "for a senior advocate",
    }.get(state.config.experience_level.value, "")

    # Cases the system surfaced that counsel never picked up — homework.
    cited_lower = " ".join(cited).lower()
    cases_to_know = []
    for c in surfaced:
        name = c.get("case_name", "")
        if name and name.lower() not in cited_lower and name not in cases_to_know:
            cases_to_know.append(name)

    return {
        "overall_summary":     _overall_text(overall, level_text),
        "structure_note":      _structure_note(state.weaknesses_flagged),
        "authority_note":      _authority_note(score.authority, cited, surfaced),
        "precision_note":      _precision_note(state.citation_flags),
        "responsiveness_note": _responsiveness_note(score.responsiveness),
        "cases_to_know":       cases_to_know[:5],
        "weaknesses_summary":  state.weaknesses_flagged,
    }


def _overall_text(score: float, level_text: str) -> str:
    if score >= 7.5:
        return f"Strong session {level_text}. The argument was coherent, structured, and well-supported by authority."
    if score >= 6.0:
        return f"Solid session {level_text}, with clear room for improvement in specific areas below."
    if score >= 4.5:
        return f"Developing session {level_text}. Several areas need focused work before the next hearing."
    return f"The fundamentals need attention {level_text}. Review the flagged weaknesses and the surfaced cases before arguing again."


def _structure_note(weaknesses: list) -> str:
    if not weaknesses:
        return "Argument structure was complete. No structural gaps were flagged."
    return f"{len(weaknesses)} structural gap(s) flagged: " + "; ".join(w[:160] for w in weaknesses[:3])


def _authority_note(score: float, cited: list, surfaced: list) -> str:
    if score >= 7.0:
        return f"Strong use of authority — {len(cited)} citation(s) made, {len(surfaced)} case(s) surfaced."
    if cited:
        unused = max(0, len(surfaced) - len(cited))
        return f"{len(cited)} citation(s) made. {unused} additional relevant case(s) were surfaced but never used."
    return "No citations were made during the session. Ground every proposition in precedent."


def _precision_note(flags: list) -> str:
    if not flags:
        return "No citation errors detected."
    return f"{len(flags)} citation issue(s) flagged. Verify reporter formats before the next hearing."


def _responsiveness_note(score: float) -> str:
    if score >= 7.0:
        return "Bench queries were met directly, and the argument adapted to interventions."
    if score >= 5.0:
        return "Partial responsiveness — some bench queries were left incompletely answered."
    return "Bench queries were not adequately addressed. Practise answering the question actually asked, completely, before returning to your structure."
