# routes/research.py — Find Legal Precedents
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, format_precedents
from database import save_session
from routes.search_web import scrape_indian_kanoon

router = APIRouter()


class ResearchRequest(BaseModel):
    query:      str
    top_k:      int            = 4
    case_id:    Optional[int]  = None   # optionally save under a matter
    use_internet: bool         = True   # also search Indian Kanoon


@router.post("/api/research")
def research(request: ResearchRequest):
    try:
        # 1. Search local case library
        chunks = search_chromadb(request.query, request.top_k)

        if not chunks:
            # Return helpful message rather than 404
            return {
                "query":          request.query,
                "answer":         "No relevant cases found in your local library for this query. Try uploading more case documents or use the Live Search feature to find cases from Indian Kanoon.",
                "precedents":     [],
                "live_results":   [],
                "total_sources":  0,
                "session_id":     None,
                "suggestion":     "Upload more cases related to this topic to get better results."
            }

        # 2. Search Indian Kanoon for live results
        live_results = []
        if request.use_internet:
            live_results = scrape_indian_kanoon(request.query, max_results=4)

        # 3. Build context (local + live)
        context = build_context(chunks)
        if live_results:
            context += "\n\nADDITIONAL RECENT JUDGMENTS FROM INDIAN KANOON:\n"
            for i, r in enumerate(live_results):
                context += f"\nLIVE SOURCE {i+1}: {r['title']} ({r['court']}, {r['year']})\n{r['snippet']}\n---"

        # 4. Generate answer
        prompt = f"""You are a senior Indian lawyer's research assistant.
Answer the legal question below using the provided case excerpts.
Cite every claim with [SOURCE N] or [LIVE SOURCE N].
Use clear, plain language that a lawyer can directly use.
Structure with numbered points. Bold key legal principles.
If sources don't cover something, say so honestly.

QUESTION: {request.query}

CASE EXCERPTS FROM LIBRARY:
{context}

ANSWER:"""

        answer = call_qwen(prompt, max_tokens=800)

        # 5. Check if answer is weak (no sources cited)
        source_count = answer.count('[SOURCE') + answer.count('[LIVE SOURCE')
        weak_result  = source_count == 0 or "Not found" in answer

        # 6. Format output
        output = {
            "query":          request.query,
            "answer":         answer,
            "precedents":     format_precedents(chunks),
            "live_results":   live_results,
            "total_sources":  len(chunks) + len(live_results),
            "weak_result":    weak_result,
            "suggestion":     "Upload more cases on this topic to get more specific answers." if weak_result else None
        }

        # 7. Auto-save to database
        session = save_session(
            session_type = "research",
            title        = request.query[:120],
            input_data   = {"query": request.query, "top_k": request.top_k},
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))