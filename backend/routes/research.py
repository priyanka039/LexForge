# routes/research.py
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from utils    import search_chromadb, call_qwen, build_context, format_precedents, response_language_directive
from database import save_session
from routes.search_web import fetch_legal_sources

router = APIRouter()


class ResearchRequest(BaseModel):
    query:              str
    top_k:              int           = 4
    case_id:            Optional[int] = None
    use_internet:       bool          = True
    response_language:  str           = "auto"  # auto | en | hi


@router.post("/api/research")
def research(request: ResearchRequest):
    try:
        # 1. Local library search
        chunks = search_chromadb(request.query, request.top_k)

        # 2. Live multi-source search
        live_results = []
        if request.use_internet:
            live_results = fetch_legal_sources(request.query, max_results=5)

        if not chunks and not live_results:
            return {
                "query":         request.query,
                "answer":        "No relevant cases found for this query. Upload more judgments or refine the search terms.",
                "precedents":    [],
                "live_results":  [],
                "total_sources": 0,
                "session_id":    None,
            }

        # 3. Build context
        context = build_context(chunks)
        if live_results:
            context += "\n\nSOURCES FROM INTERNET (Indian Kanoon, SCI, eCourts):\n"
            for i, r in enumerate(live_results):
                citation_str = f" | Citation: {r['citation']}" if r.get('citation') else ""
                context += (
                    f"\nLIVE SOURCE {i+1}: {r['title']}"
                    f"\nCourt: {r['court']} | Year: {r['year']}{citation_str}"
                    f"\nSource: {r['source']}"
                    f"\n{r['snippet']}\n---"
                )

        # 4. Generate answer with strict citation discipline
        prompt = f"""You are a senior Indian advocate conducting legal research for a colleague.

CRITICAL CITATION RULES:
- Cite EVERY legal proposition with [SOURCE N] or [LIVE SOURCE N]
- Use full case names when referencing cases, never abbreviate
- State the court and year for every case
- If a source only partially supports a point, say so clearly
- If the sources do not cover something, say explicitly: "No authority found in the provided sources for this proposition. Independent verification on SCC Online or Manupatra is strongly advised."
- Do NOT fabricate case names, citation numbers, or holdings

FORMAT:
- Use numbered points
- Bold key legal principles and case names
- Keep language precise and professional
- No vague statements like "courts have held" without citing which court

QUERY: {request.query}

CASE EXCERPTS AND SOURCES:
{context}

RESEARCH MEMORANDUM:{response_language_directive(request.response_language)}"""

        answer = call_qwen(prompt, max_tokens=1000)

        # 5. Citation quality check
        source_count = answer.count('[SOURCE') + answer.count('[LIVE SOURCE')

        # 6. Disclaimer when sources are thin
        disclaimer = None
        if source_count < 2:
            disclaimer = (
                "The library does not contain sufficient cases on this topic. "
                "The analysis above is based on limited sources. "
                "Verify all propositions on SCC Online, Manupatra, or Indian Kanoon independently."
            )

        output = {
            "query":         request.query,
            "answer":        answer,
            "precedents":    format_precedents(chunks),
            "live_results":  live_results,
            "total_sources": len(chunks) + len(live_results),
            "disclaimer":    disclaimer,
        }

        session = save_session(
            session_type = "research",
            title        = request.query[:120],
            input_data   = {
                "query":              request.query,
                "top_k":              request.top_k,
                "use_internet":       request.use_internet,
                "response_language":  request.response_language,
            },
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))