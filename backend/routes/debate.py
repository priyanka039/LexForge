# routes/debate.py — Debate Simulation
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, parse_json_robust, format_precedents, response_language_directive
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
    plaintiff_position: str           = ""
    defense_position:   str           = ""
    judge_persona:      str           = DEFAULT_PERSONA
    case_id:            Optional[int] = None
    response_language:  str           = "auto"  # auto | en | hi


def safe_parse(raw: str, key: str) -> list:
    try:
        result = parse_json_robust(raw)
        return result.get(key, [])
    except Exception:
        # Fallback: wrap raw text as a single point
        return [{"point": raw[:300], "citation": ""}]


@router.post("/api/debate")
def debate(request: DebateRequest):
    try:
        persona    = JUDGE_PERSONAS.get(request.judge_persona, JUDGE_PERSONAS[DEFAULT_PERSONA])
        judge_sys  = persona["prompt"]
        judge_name = persona["name"]

        chunks  = search_chromadb(request.case_summary, top_k=5)
        context = build_context(chunks)
        lang    = response_language_directive(request.response_language)

        # ── Round 1: Opening Submissions ──────────────────────
        p_opening = safe_parse(call_qwen(
            f"""You are Petitioner's Senior Counsel making opening submissions in {request.jurisdiction}.
CASE: {request.case_summary}
{f"YOUR POSITION: {request.plaintiff_position}" if request.plaintiff_position else ""}
JUDGE'S TEMPERAMENT: {judge_sys}
RELEVANT AUTHORITIES:
{context}

Draft 3 strong opening submission points. Tailor them to this judge's temperament.
Cite specific cases or statutes where possible.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"arguments":[{{"point":"Your submission point","citation":"Case name or statute"}}]}}{lang}""",
            max_tokens=800), "arguments")

        d_opening = safe_parse(call_qwen(
            f"""You are Respondent's Counsel making opening submissions in {request.jurisdiction}.
CASE: {request.case_summary}
{f"YOUR POSITION: {request.defense_position}" if request.defense_position else ""}
JUDGE'S TEMPERAMENT: {judge_sys}
RELEVANT AUTHORITIES:
{context}

Draft 3 strong opening submission points opposing the Petitioner. Tailor to this judge's temperament.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"arguments":[{{"point":"Your submission point","citation":"Case name or statute"}}]}}{lang}""",
            max_tokens=800), "arguments")

        # ── Round 2: Rebuttal round (P) + Sur-rebuttal (R) ─────
        # NOTE: p_sum = what PETITIONER argued (from p_opening)
        #       d_sum = what RESPONDENT argued (from d_opening)
        p_sum = " | ".join([a.get("point", "") for a in p_opening[:3]])
        d_sum = " | ".join([a.get("point", "") for a in d_opening[:3]])

        p_rebuttal = safe_parse(call_qwen(
            f"""You are Petitioner's Counsel in Rebuttal.
The Respondent argued in opening: {d_sum}
JUDGE: {judge_sys}
AUTHORITIES: {context}
Rebut each of Respondent's points directly and specifically.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"rebuttals":[{{"point":"Your rebuttal","citation":"Case or statute"}}]}}{lang}""",
            max_tokens=700), "rebuttals")

        d_rebuttal = safe_parse(call_qwen(
            f"""You are Respondent's Counsel in Sur-Rebuttal.
The Petitioner argued in opening: {p_sum}
JUDGE: {judge_sys}
AUTHORITIES: {context}
Rebut each of Petitioner's points directly and specifically.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{"rebuttals":[{{"point":"Your rebuttal","citation":"Case or statute"}}]}}{lang}""",
            max_tokens=700), "rebuttals")

        # ── Round 3: Judicial Observations (Bench) ───────────
        summary_raw = call_qwen(
            f"""You are {judge_name} presiding over this matter in {request.jurisdiction}.
CASE: {request.case_summary}
PETITIONER SUBMITTED: {p_sum}
RESPONDENT SUBMITTED: {d_sum}
YOUR JUDICIAL TEMPERAMENT: {judge_sys}

Give your judicial observations as this specific judge would — in your voice, from the bench.
Return ONLY valid JSON (all string values must follow OUTPUT LANGUAGE below):
{{
  "overall_assessment": "2-3 sentence balanced analysis of both sides",
  "plaintiff_strength": "HIGH or MODERATE or LOW",
  "defense_strength": "HIGH or MODERATE or LOW",
  "likely_outcome": "1-2 sentence prediction",
  "strategic_recommendation": "Specific litigation advice for the weaker party",
  "risk_level": "HIGH or MODERATE or LOW",
  "judicial_observation": "What you as this specific judge would say from the bench — in first person, direct"
}}{lang}""",
            max_tokens=900)

        try:
            summary = parse_json_robust(summary_raw)
        except Exception:
            summary = {
                "overall_assessment":       summary_raw[:300],
                "plaintiff_strength":       "MODERATE",
                "defense_strength":         "MODERATE",
                "likely_outcome":           "Outcome uncertain on present pleadings.",
                "strategic_recommendation": "Obtain senior counsel's opinion before the next date.",
                "risk_level":               "MODERATE",
                "judicial_observation":     ""
            }

        output = {
            "case_summary":  request.case_summary,
            "jurisdiction":  request.jurisdiction,
            "judge_persona": persona,
            "round1":        {"plaintiff": p_opening,  "defense": d_opening},
            "round2":        {"plaintiff": p_rebuttal, "defense": d_rebuttal},
            "summary":       summary,
            "precedents":    format_precedents(chunks),
        }

        session = save_session(
            session_type = "debate",
            title        = f"Debate: {request.case_summary[:100]}",
            input_data   = {
                "case_summary":       request.case_summary,
                "jurisdiction":       request.jurisdiction,
                "plaintiff_position": request.plaintiff_position,
                "defense_position":   request.defense_position,
                "judge_persona":      request.judge_persona,
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