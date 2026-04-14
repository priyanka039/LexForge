# routes/opposition.py — Test Your Case
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, parse_json_robust
from database import save_session

router = APIRouter()


class OppositionRequest(BaseModel):
    argument: str
    case_id:  Optional[int] = None


@router.post("/api/opposition")
def opposition(request: OppositionRequest):
    try:
        chunks  = search_chromadb(request.argument, top_k=4)
        context = build_context(chunks)

        prompt = f"""You are the opposing counsel in an Indian court.
Find every weakness in this argument. Be thorough and direct.
Use the case excerpts where they support the opposing position.

ARGUMENT TO CHALLENGE:
{request.argument}

AVAILABLE CASES:
{context}

Return ONLY this JSON:
{{
  "risk_level": "HIGH/MODERATE/LOW",
  "overall_confidence": 70,
  "weaknesses": [
    {{"id": "W1", "severity": "HIGH/MODERATE/LOW",
      "description": "Specific weakness with legal basis"}}
  ],
  "counter_arguments": [
    {{"point": "Specific counter argument", "source": "Case or statute or empty string"}}
  ],
  "strategy_recommendations": [
    {{"type": "DO",    "advice": "Specific action to strengthen your case"}},
    {{"type": "AVOID", "advice": "Specific thing that will hurt your case"}}
  ]
}}"""

        result_raw = call_qwen(prompt, max_tokens=1000)
        try:
            result = parse_json_robust(result_raw)
        except Exception:
            result = {"risk_level": "MODERATE", "overall_confidence": 65,
                      "weaknesses": [{"id":"W1","severity":"MODERATE","description":result_raw[:400]}],
                      "counter_arguments": [], "strategy_recommendations": []}

        output = {
            "argument":            request.argument,
            "analysis":            result,
            "contrary_precedents": [{"case_name": c['metadata'].get('case_name',''),
                                     "court":     c['metadata'].get('court',''),
                                     "year":      c['metadata'].get('year',''),
                                     "snippet":   c['text'][:200],
                                     "score":     c['score']} for c in chunks]
        }

        session = save_session(
            session_type = "opposition",
            title        = f"Weakness check: {request.argument[:100]}",
            input_data   = {"argument": request.argument},
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))