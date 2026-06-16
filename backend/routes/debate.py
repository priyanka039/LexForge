# routes/debate.py — Debate Simulation
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import (
    search_chromadb, call_qwen, build_context, parse_json_robust,
    format_precedents, response_language_directive,
    verify_citations, corpus_is_empty,
)
from database import save_session

router = APIRouter()

JUDGE_PERSONAS = {
    "strict_proceduralist": {
        "name":   "The Strict Proceduralist",
        "desc":   "Stickler for procedure. Raises maintainability, limitation, and jurisdiction first. Demands citations for every proposition.",
        "prompt": "You are a strict proceduralist judge. You raise maintainability, limitation, and jurisdiction sua sponte. You demand a cited authority for every proposition advanced. Merits come only after procedure is satisfied.",
    },
    "rights_oriented": {
        "name":   "The Rights-Oriented Bench",
        "desc":   "Expansive interpreter of fundamental rights. Sympathetic to weaker parties. Willing to mould relief creatively.",
        "prompt": "You are a progressive, rights-oriented judge. You interpret constitutional provisions expansively. You are sympathetic to the disadvantaged party. You look to the spirit of law, not merely its letter. You may mould relief beyond what was pleaded if justice requires.",
    },
    "commercial_pragmatist": {
        "name":   "The Commercial Court",
        "desc":   "Efficiency-first. Respects party autonomy and contractual freedom. Upholds arbitration clauses and liquidated damages.",
        "prompt": "You are a commercial court judge who values efficiency and party autonomy above all. You enforce contracts as written. You respect arbitration clauses and do not rewrite commercial bargains. You focus on what parties actually agreed, not what they should have agreed.",
    },
    "criminal_bench": {
        "name":   "The Criminal Bench",
        "desc":   "Rigorous on evidentiary standards. Scrutinises mens rea carefully. Prosecution must prove beyond reasonable doubt.",
        "prompt": "You are a criminal court judge. You demand proof beyond reasonable doubt from the prosecution. You scrutinise identification evidence, confession admissibility, and custodial statements with care. You consider mens rea, actus reus, and mitigating factors at sentencing.",
    },
    "constitutional": {
        "name":   "The Constitutional Bench",
        "desc":   "Concerned with separation of powers, Basic Structure doctrine, and constitutional morality.",
        "prompt": "You are a constitutional bench judge. You examine whether impugned legislation violates the Basic Structure. You balance individual rights against public interest under the doctrine of proportionality. You are mindful of the separation of powers and the limits of judicial review.",
    },
}

DEFAULT_PERSONA = "strict_proceduralist"


class DebateRequest(BaseModel):
    case_summary:       str
    jurisdiction:       str           = "High Court of Delhi"
    proceeding_level:   str           = ""     # trial | high_court | supreme | appeal
    plaintiff_position: str           = ""
    defense_position:   str           = ""
    judge_persona:      str           = DEFAULT_PERSONA
    known_outcome:      str           = ""     # actual result, if a known/landmark case
    case_id:            Optional[int] = None
    response_language:  str           = "auto"  # auto | en | hi


# Role labels differ by proceeding level — in HC a challenger is the
# Petitioner; in an SC appeal the same party may be the Appellant or
# Respondent. We surface this so labels are never silently wrong.
PROCEEDING_ROLES = {
    "trial":      ("Plaintiff/Prosecution", "Defendant/Accused"),
    "high_court": ("Petitioner", "Respondent"),
    "supreme":    ("Appellant", "Respondent"),
    "appeal":     ("Appellant", "Respondent"),
}


def safe_parse(raw: str, key: str) -> list:
    try:
        result = parse_json_robust(raw)
        return result.get(key, [])
    except Exception:
        # Fallback: wrap raw text as a single point
        return [{"point": raw[:300], "citation": ""}]


def _flatten(points: list, limit: int = 4) -> str:
    """Render structured argument points (with citations) into prompt text."""
    out = []
    for i, a in enumerate(points[:limit], 1):
        pt   = (a.get("point") or "").strip()
        cite = (a.get("citation") or "").strip()
        out.append(f"{i}. {pt}" + (f"  [Authority: {cite}]" if cite else ""))
    return "\n".join(out) or "(no submissions recorded)"


@router.post("/api/debate")
def debate(request: DebateRequest):
    try:
        persona    = JUDGE_PERSONAS.get(request.judge_persona, JUDGE_PERSONAS[DEFAULT_PERSONA])
        judge_sys  = persona["prompt"]
        judge_name = persona["name"]
        lang       = response_language_directive(request.response_language)

        level            = (request.proceeding_level or "high_court").lower()
        p_role, d_role   = PROCEEDING_ROLES.get(level, PROCEEDING_ROLES["high_court"])
        forum            = request.jurisdiction

        # Case law and statute provisions retrieved separately so a bare act
        # cannot crowd out citeable precedent in the authorities panel.
        case_chunks    = search_chromadb(
            request.case_summary, top_k=5, exclude_doc_types=["statute"]
        )
        statute_chunks = search_chromadb(
            request.case_summary, top_k=2, doc_types=["statute"], min_score=0.40
        )
        chunks  = case_chunks + statute_chunks
        context = build_context(chunks)
        no_docs = corpus_is_empty() or not chunks

        # Lawyers need the WINNING ground, not generic points. Force counsel to
        # lead with the strongest procedural / special-act ground first.
        lead_directive = (
            "Lead with the single STRONGEST procedural, jurisdictional, or special-statute "
            "ground first (e.g. for the Prevention of Corruption Act check designated-officer "
            "sanction under S.5A; for NDPS check S.50; for POCSO check mandatory procedure), "
            "THEN substantive grounds. Do not bury the winning point among generic ones."
        )

        # ── Round 1: Opening Submissions ──────────────────────
        p_opening = safe_parse(call_qwen(
            f"""You are {p_role}'s Senior Counsel making OPENING submissions before {forum}.
CASE: {request.case_summary}
{f"YOUR POSITION: {request.plaintiff_position}" if request.plaintiff_position else ""}
JUDGE'S TEMPERAMENT: {judge_sys}
RELEVANT AUTHORITIES:
{context}

{lead_directive}
Draft 3-4 opening submission points, strongest first. Tailor to this judge's temperament.
Cite a specific case or statute for each point where one genuinely supports it; if none does, set citation to "" rather than inventing one.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"arguments":[{{"point":"Your submission point","citation":"Case name or statute, or empty"}}]}}{lang}""",
            max_tokens=900), "arguments")

        d_opening = safe_parse(call_qwen(
            f"""You are {d_role}'s Senior Counsel making OPENING submissions before {forum}, opposing the {p_role}.
CASE: {request.case_summary}
{f"YOUR POSITION: {request.defense_position}" if request.defense_position else ""}
JUDGE'S TEMPERAMENT: {judge_sys}
RELEVANT AUTHORITIES:
{context}

{lead_directive}
Draft 3-4 opening submission points opposing the {p_role}, strongest first.
Cite a specific case or statute only where one genuinely supports the point; otherwise citation "".
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"arguments":[{{"point":"Your submission point","citation":"Case name or statute, or empty"}}]}}{lang}""",
            max_tokens=900), "arguments")

        # ── Round 2: Rebuttals — now threaded with the FULL opponent round ──
        p_open_txt = _flatten(p_opening)
        d_open_txt = _flatten(d_opening)

        p_rebuttal = safe_parse(call_qwen(
            f"""You are {p_role}'s Counsel in REBUTTAL before {forum}.
YOUR OWN OPENING (do NOT merely repeat it):
{p_open_txt}

THE {d_role.upper()}'S FULL OPENING you must now answer:
{d_open_txt}

JUDGE: {judge_sys}
AUTHORITIES: {context}

Directly address at least TWO specific points the {d_role} made above — name the point and dismantle it.
Do NOT restate your opening. Escalate: be sharper and more specific than the opening round.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"rebuttals":[{{"point":"Your rebuttal to a specific opposing point","citation":"Case or statute, or empty"}}]}}{lang}""",
            max_tokens=800), "rebuttals")

        # Sur-rebuttal sees BOTH the petitioner's opening and the petitioner's rebuttal.
        p_reb_txt = _flatten(p_rebuttal)
        d_rebuttal = safe_parse(call_qwen(
            f"""You are {d_role}'s Counsel in SUR-REBUTTAL before {forum}.
YOUR OWN OPENING (do NOT merely repeat it):
{d_open_txt}

THE {p_role.upper()}'S OPENING:
{p_open_txt}
THE {p_role.upper()}'S REBUTTAL you must now answer:
{p_reb_txt}

JUDGE: {judge_sys}
AUTHORITIES: {context}

Directly answer at least TWO specific points from the {p_role}'s rebuttal above. Do NOT restate your opening.
Escalate: this is your last word — make it the sharpest exchange of the hearing.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"rebuttals":[{{"point":"Your sur-rebuttal to a specific point","citation":"Case or statute, or empty"}}]}}{lang}""",
            max_tokens=800), "rebuttals")

        # ── Round 3: Judicial Observations — judge now sees ALL FOUR exchanges ──
        d_reb_txt = _flatten(d_rebuttal)
        outcome_directive = ""
        if request.known_outcome.strip():
            outcome_directive = (
                f"\nThe ACTUAL recorded outcome of this matter is: {request.known_outcome.strip()}\n"
                "In 'outcome_comparison', compare your prediction to this actual outcome and state "
                "whether it is a MATCH, PARTIAL MATCH, or MISS, with one line of reasoning."
            )

        summary_raw = call_qwen(
            f"""You are {judge_name} presiding in {forum} ({p_role} v. {d_role}).
CASE: {request.case_summary}

{p_role.upper()} OPENING:
{p_open_txt}
{d_role.upper()} OPENING:
{d_open_txt}
{p_role.upper()} REBUTTAL:
{p_reb_txt}
{d_role.upper()} SUR-REBUTTAL:
{d_reb_txt}

YOUR JUDICIAL TEMPERAMENT: {judge_sys}
Weigh the REBUTTAL exchanges, not just the openings. Ask the hard question each side left unanswered.{outcome_directive}

Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{
  "overall_assessment": "2-3 sentence analysis weighing both openings AND rebuttals",
  "plaintiff_strength": "HIGH or MODERATE or LOW",
  "defense_strength": "HIGH or MODERATE or LOW",
  "likely_outcome": "1-2 sentence prediction",
  "outcome_comparison": "{'MATCH/PARTIAL/MISS vs actual outcome with reasoning' if request.known_outcome.strip() else 'N/A — no known outcome supplied'}",
  "risk_level": "HIGH or MODERATE or LOW",
  "risk_factors": ["2-3 specific factors driving the risk rating, one line each"],
  "strategic_recommendation": "Specific litigation advice for the weaker party",
  "bench_question": "The single hardest question you would put to counsel from the bench",
  "judicial_observation": "What you as this specific judge would say from the bench — first person, direct"
}}{lang}""",
            max_tokens=1000)

        try:
            summary = parse_json_robust(summary_raw)
        except Exception:
            summary = {
                "overall_assessment":       summary_raw[:300],
                "plaintiff_strength":       "MODERATE",
                "defense_strength":         "MODERATE",
                "likely_outcome":           "Outcome uncertain on present pleadings.",
                "outcome_comparison":       "N/A",
                "risk_level":               "MODERATE",
                "risk_factors":             ["Assessment could not be fully parsed — review the transcript."],
                "strategic_recommendation": "Obtain senior counsel's opinion before the next date.",
                "bench_question":           "",
                "judicial_observation":     "",
            }

        # ── Citation provenance across every generated point ──
        all_text = " ".join(
            (a.get("point", "") + " " + a.get("citation", ""))
            for a in (p_opening + d_opening + p_rebuttal + d_rebuttal)
        )
        provenance = verify_citations(all_text, grounded_chunks=case_chunks)
        unverified = [c["citation"] for c in provenance if not c["verified"]]

        warnings = []
        if no_docs:
            warnings.append(
                "No case judgments were retrieved for this matter. Arguments are based on model "
                "knowledge only and MAY BE INACCURATE. Use 'Fetch from Indian Kanoon' to import "
                "the judgment into your library before relying on this hearing."
            )
        if unverified:
            warnings.append(
                "These cited authorities are not in your corpus and may be from model memory — "
                "verify before use: " + "; ".join(unverified[:6])
            )

        output = {
            "case_summary":     request.case_summary,
            "jurisdiction":     request.jurisdiction,
            "proceeding_level": level,
            "roles":            {"plaintiff": p_role, "defense": d_role},
            "judge_persona":    persona,
            "round1":           {"plaintiff": p_opening,  "defense": d_opening},
            "round2":           {"plaintiff": p_rebuttal, "defense": d_rebuttal},
            "summary":          summary,
            "precedents":       format_precedents(case_chunks),
            "provisions":       format_precedents(statute_chunks),
            "citation_provenance": provenance,
            "unverified_citations": unverified,
            "corpus_empty":     no_docs,
            "warnings":         warnings,
        }

        session = save_session(
            session_type = "debate",
            title        = f"Debate: {request.case_summary[:100]}",
            input_data   = {
                "case_summary":       request.case_summary,
                "jurisdiction":       request.jurisdiction,
                "proceeding_level":   level,
                "plaintiff_position": request.plaintiff_position,
                "defense_position":   request.defense_position,
                "judge_persona":      request.judge_persona,
                "known_outcome":      request.known_outcome,
                "response_language":  request.response_language,
            },
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/debate/personas")
def get_personas():
    return {
        "personas": [
            {"id": k, "name": v["name"], "desc": v["desc"]}
            for k, v in JUDGE_PERSONAS.items()
        ]
    }