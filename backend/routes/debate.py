# routes/debate.py — Simulate Court
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, parse_json_robust, format_precedents
from database import save_session

router = APIRouter()


class DebateRequest(BaseModel):
    case_summary:       str
    jurisdiction:       str           = "High Court of Delhi"
    plaintiff_position: str           = ""
    defense_position:   str           = ""
    case_id:            Optional[int] = None


def safe_parse(raw: str, key: str) -> list:
    try:
        return parse_json_robust(raw).get(key, [])
    except Exception:
        return [{"point": raw[:300], "citation": ""}]


@router.post("/api/debate")
def debate(request: DebateRequest):
    try:
        chunks  = search_chromadb(request.case_summary, top_k=5)
        context = build_context(chunks)

        p_opening = safe_parse(call_qwen(f"""You are petitioner's senior counsel.
Give 3 strong opening argument points.
CASE: {request.case_summary}
COURT: {request.jurisdiction}
{f"YOUR POSITION: {request.plaintiff_position}" if request.plaintiff_position else ""}
CASES: {context}
Return ONLY: {{"arguments":[{{"point":"...","citation":"..."}}]}}""", 600), "arguments")

        d_opening = safe_parse(call_qwen(f"""You are respondent's counsel.
Give 3 strong opening argument points against the petitioner.
CASE: {request.case_summary}
COURT: {request.jurisdiction}
{f"YOUR POSITION: {request.defense_position}" if request.defense_position else ""}
CASES: {context}
Return ONLY: {{"arguments":[{{"point":"...","citation":"..."}}]}}""", 600), "arguments")

        p_rebuttal = safe_parse(call_qwen(f"""You are petitioner's counsel.
Rebut: {" | ".join([a.get("point","") for a in d_opening[:3]])}
CASES: {context}
Return ONLY: {{"rebuttals":[{{"point":"...","citation":"..."}}]}}""", 500), "rebuttals")

        d_rebuttal = safe_parse(call_qwen(f"""You are respondent's counsel.
Rebut: {" | ".join([a.get("point","") for a in p_opening[:3]])}
CASES: {context}
Return ONLY: {{"rebuttals":[{{"point":"...","citation":"..."}}]}}""", 500), "rebuttals")

        summary_raw = call_qwen(f"""You are a neutral senior judge.
Assess this case: {request.case_summary}
Petitioner argued: {" | ".join([a.get("point","") for a in p_opening[:3]])}
Respondent argued: {" | ".join([a.get("point","") for a in d_opening[:3]])}
Return ONLY: {{
  "overall_assessment":"...",
  "plaintiff_strength":"HIGH/MODERATE/LOW",
  "defense_strength":"HIGH/MODERATE/LOW",
  "likely_outcome":"...",
  "strategic_recommendation":"...",
  "risk_level":"HIGH/MODERATE/LOW"
}}""", 700)

        try:
            summary = parse_json_robust(summary_raw)
        except Exception:
            summary = {"overall_assessment": summary_raw[:300],
                       "plaintiff_strength":"MODERATE","defense_strength":"MODERATE",
                       "likely_outcome":"Uncertain.","strategic_recommendation":"Seek expert counsel.",
                       "risk_level":"MODERATE"}

        output = {
            "case_summary": request.case_summary,
            "jurisdiction": request.jurisdiction,
            "round1":  {"plaintiff": p_opening,  "defense": d_opening},
            "round2":  {"plaintiff": p_rebuttal, "defense": d_rebuttal},
            "summary":    summary,
            "precedents": format_precedents(chunks)
        }

        session = save_session(
            session_type = "debate",
            title        = f"Court simulation: {request.case_summary[:100]}",
            input_data   = {"case_summary": request.case_summary, "jurisdiction": request.jurisdiction},
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))