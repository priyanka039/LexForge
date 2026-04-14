# routes/argument.py — Build Your Argument
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, parse_json_robust, format_precedents
from database import save_session

router = APIRouter()


class ArgumentRequest(BaseModel):
    facts:           str
    jurisdiction:    str           = "High Court of Delhi"
    area_of_law:     str           = "General"
    client_position: str           = "Petitioner"
    case_id:         Optional[int] = None


@router.post("/api/argument")
def build_argument(request: ArgumentRequest):
    try:
        # Step 1: Extract legal issues
        issue_prompt = f"""You are a senior Indian lawyer.
Read these facts and identify every distinct legal issue.
There are usually 2–4 separate issues. List each one separately.

Facts: {request.facts}
Court: {request.jurisdiction}

Return ONLY this JSON — no explanation:
{{
  "issues": [
    {{"issue": "brief issue title", "area_of_law": "specific area", "priority": "high/medium/low"}},
    {{"issue": "brief issue title", "area_of_law": "specific area", "priority": "high/medium/low"}}
  ]
}}"""

        issues_data = parse_json_robust(call_qwen(issue_prompt, max_tokens=500))
        issues      = issues_data.get('issues', [])

        if not issues:
            raise ValueError("Could not identify legal issues from the facts provided.")

        # Step 2: IRAC for each issue
        irac_results   = []
        all_precedents = []

        for idx, issue in enumerate(issues):
            chunks  = search_chromadb(
                f"{issue['issue']} {issue.get('area_of_law','')} {request.jurisdiction} India",
                top_k=3
            )
            context = build_context(chunks)

            irac_prompt = f"""You are a senior Indian lawyer drafting a legal argument.
Write a complete legal analysis for this issue.
Use ONLY the provided case excerpts. Cite with [SOURCE N].
Write in clear legal language a judge would appreciate.

FACTS: {request.facts}
ISSUE: {issue['issue']}
YOUR CLIENT'S POSITION: {request.client_position}
COURT: {request.jurisdiction}

RELEVANT CASES:
{context}

Return ONLY this JSON:
{{
  "issue":       "The precise legal question",
  "rule":        "The applicable law and precedents [SOURCE N]",
  "application": "How the law applies to these specific facts [SOURCE N]",
  "conclusion":  "The legal outcome and remedy for {request.client_position}"
}}"""

            irac_raw = call_qwen(irac_prompt, max_tokens=1000)
            try:
                irac = parse_json_robust(irac_raw)
            except Exception:
                irac = {"issue": issue['issue'], "rule": irac_raw[:500],
                        "application": "See analysis above.", "conclusion": "Further review required."}

            for chunk in chunks:
                meta  = chunk['metadata']
                court = meta.get('court', 'Unknown')
                all_precedents.append({
                    "case_name": meta.get('case_name', 'Unknown'), "court": court,
                    "year": meta.get('year', 'Unknown'), "score": chunk['score'],
                    "issue_num": idx + 1, "snippet": chunk['text'][:200],
                    "binding": "Binding" if court == 'Supreme Court of India' else "Persuasive"
                })

            irac_results.append({
                "issue_title": issue['issue'],
                "area_of_law": issue.get('area_of_law', request.area_of_law),
                "priority":    issue.get('priority', 'medium'),
                "irac":        irac,
                "precedents":  [{"case_name": c['metadata'].get('case_name',''),
                                 "court": c['metadata'].get('court',''),
                                 "year":  c['metadata'].get('year',''),
                                 "score": c['score']} for c in chunks]
            })

        output = {
            "facts":           request.facts,
            "jurisdiction":    request.jurisdiction,
            "client_position": request.client_position,
            "total_issues":    len(issues),
            "arguments":       irac_results,
            "all_precedents":  all_precedents,
            "weak_result":     len(all_precedents) == 0,
            "suggestion":      "Upload cases related to this area of law for stronger arguments." if len(all_precedents) == 0 else None
        }

        session = save_session(
            session_type = "argument",
            title        = request.facts[:120],
            input_data   = {"facts": request.facts, "jurisdiction": request.jurisdiction,
                            "client_position": request.client_position},
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))