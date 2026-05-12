# routes/opposition.py — Devil's Advocate
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, parse_json_robust, response_language_directive
from database import save_session
from routes.debate  import JUDGE_PERSONAS, DEFAULT_PERSONA
from routes.search_web import fetch_legal_sources

router = APIRouter()


class OppositionRequest(BaseModel):
    argument:           str
    judge_persona:      str           = DEFAULT_PERSONA
    case_id:            Optional[int] = None
    response_language:  str           = "auto"  # auto | en | hi


@router.post("/api/opposition")
def opposition(request: OppositionRequest):
    try:
        persona   = JUDGE_PERSONAS.get(request.judge_persona, JUDGE_PERSONAS[DEFAULT_PERSONA])
        judge_sys = persona["prompt"]

        # Local library
        chunks  = search_chromadb(request.argument, top_k=4)
        context = build_context(chunks)

        # Live adverse cases
        live = fetch_legal_sources(request.argument, max_results=3)
        if live:
            context += "\n\nADVERSE CASES FROM INTERNET:\n"
            for i, r in enumerate(live):
                cit = f" | {r['citation']}" if r.get("citation") else ""
                context += f"\nLIVE {i+1}: {r['title']} | {r['court']} | {r['year']}{cit}\n{r['snippet']}\n---"

        prompt = f"""You are both opposing senior counsel and an expert on how {persona['name']} approaches cases.
Goal: help the user stress-test their submissions before hearing — anticipate opposing counsel's lines and the Bench's questions ({persona['name']}'s style).

Your task: ruthlessly identify every vulnerability in this submission.

CRITICAL RULES:
- Every weakness you identify must cite a legal basis (statute, principle, or case)
- When you cite a case, give the full name, court, and year
- Flag if jurisdiction is wrong, if evidence would be inadmissible, if limitation has run
- Do NOT fabricate cases. If you do not have a specific case, say "No specific authority cited here — opposing counsel would research this"
- The bench questions must reflect how {persona['name']} specifically thinks, not generic questions
- Remove the confidence percentage — it is meaningless without knowing the evidence, bench composition, and procedure

SUBMISSION TO ATTACK:
{request.argument}

AVAILABLE ADVERSE AUTHORITIES:
{context}

JUDGE TEMPERAMENT: {judge_sys}

Return ONLY valid JSON:
{{
  "risk_level": "HIGH/MODERATE/LOW",
  "risk_explanation": "One sentence explaining why this risk level, grounded in specific legal issues",
  "weaknesses": [
    {{
      "id": "W1",
      "severity": "HIGH/MODERATE/LOW",
      "description": "Specific weakness with legal basis. Name the statute, principle, or case.",
      "how_to_address": "Concrete step the advocate should take before the hearing"
    }}
  ],
  "counter_arguments": [
    {{"point": "Specific counter submission opposing counsel will advance", "authority": "Case name, court, year or statute"}}
  ],
  "bench_questions": [
    {{
      "question": "The exact question this specific judge would ask, in their voice",
      "implication": "Why this question is dangerous to your client's case",
      "suggested_answer": "The best response available on current instructions"
    }}
  ],
  "priority_actions": [
    "Most urgent thing to do before the next date — specific and actionable",
    "Second priority",
    "Third priority"
  ],
  "verification_note": "Cases in the above analysis that must be independently verified on SCC Online or Manupatra before reliance"
}}{response_language_directive(request.response_language)}"""

        result_raw = call_qwen(prompt, max_tokens=1400)
        try:
            result = parse_json_robust(result_raw)
        except Exception:
            result = {
                "risk_level": "MODERATE",
                "risk_explanation": "Unable to fully parse analysis. Raw output below.",
                "weaknesses": [{"id": "W1", "severity": "MODERATE", "description": result_raw[:400], "how_to_address": "Review manually."}],
                "counter_arguments": [], "bench_questions": [], "priority_actions": [],
                "verification_note": "All citations must be independently verified.",
            }

        # Remove confidence percentage if the model included it
        result.pop("overall_confidence", None)
        result.pop("confidence_score", None)

        output = {
            "argument":      request.argument,
            "judge_persona": persona,
            "analysis":      result,
            "contrary_precedents": [
                {
                    "case_name": c["metadata"].get("case_name",""),
                    "court":     c["metadata"].get("court",""),
                    "year":      c["metadata"].get("year",""),
                    "snippet":   c["text"][:200],
                    "score":     c["score"],
                    "source":    "Local Library",
                }
                for c in chunks
            ] + [
                {
                    "case_name": r["title"],
                    "court":     r["court"],
                    "year":      r["year"],
                    "snippet":   r["snippet"][:200],
                    "score":     0.75,
                    "source":    r["source"],
                    "url":       r.get("url",""),
                    "citation":  r.get("citation",""),
                }
                for r in live
            ],
        }

        session = save_session(
            session_type = "opposition",
            title        = f"Devil's Advocate: {request.argument[:100]}",
            input_data   = {
                "argument":           request.argument,
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