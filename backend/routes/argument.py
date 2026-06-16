# routes/argument.py — Argument Builder (IRAC)
import json

from fastapi            import APIRouter, HTTPException
from fastapi.responses  import StreamingResponse
from pydantic           import BaseModel
from typing             import Optional, Iterator

from utils    import (
    search_chromadb, call_qwen, build_context, parse_json_robust,
    format_precedents, response_language_directive, verify_citations,
)
from database import save_session
from routes.search_web import fetch_legal_sources

router = APIRouter()


class ArgumentRequest(BaseModel):
    facts:              str
    jurisdiction:       str           = "High Court of Delhi"
    area_of_law:        str           = "General"
    client_position:    str           = "Petitioner"
    extra_context:      str           = ""
    case_id:            Optional[int] = None
    response_language:  str           = "auto"  # auto | en | hi


# Indian litigation is frequently won on a procedural or special-statute
# ground, not the substantive merits. Force the model to hunt for these first
# rather than producing generic submissions.
_PROCEDURAL_DIRECTIVE = (
    "Before substantive merits, actively check for a winning PROCEDURAL, "
    "JURISDICTIONAL, or SPECIAL-STATUTE ground and lead with it if it exists: "
    "e.g. Prevention of Corruption Act S.5A/S.19 sanction & designated-officer "
    "requirements; NDPS Act S.50 search procedure; POCSO mandatory procedure; "
    "Cr.P.C. S.197/S.482 and limitation; CPC Order VII Rule 11 and res judicata; "
    "writ maintainability and alternative-remedy bars. If such a ground exists, "
    "it MUST appear as the first issue/submission."
)


def _issue_prompt(request: "ArgumentRequest") -> str:
    return f"""You are a senior Indian advocate reviewing instructions from a client.
Identify the distinct legal issues in these facts. There are usually 2 to 4 issues.
List each one separately. Be precise — this is a professional legal analysis.

{_PROCEDURAL_DIRECTIVE}

Facts: {request.facts}
Court: {request.jurisdiction}
Area of Law: {request.area_of_law}
{f"Additional context: {request.extra_context}" if request.extra_context else ""}

Order issues by winning strength — strongest procedural/technical ground FIRST.
Return ONLY valid JSON, no explanation:
{{
  "issues": [
    {{"issue": "precise legal question", "area_of_law": "specific area", "priority": "high/medium/low", "ground_type": "procedural/jurisdictional/substantive"}}
  ]
}}{response_language_directive(request.response_language)}"""


def _irac_prompt(request: "ArgumentRequest", issue: dict, context: str) -> str:
    return f"""You are a senior Indian advocate drafting a legal argument for a {request.client_position}.
Write a rigorous IRAC analysis for the issue below.

{_PROCEDURAL_DIRECTIVE}

CITATION RULES (strictly follow):
- Every legal proposition MUST be supported by a cited case or statute
- Use the full case name: e.g. "Maneka Gandhi v. Union of India (1978) 1 SCC 248"
- State court and year: e.g. "Supreme Court, 1978" or "Delhi High Court, 2019"
- Prefer authorities from AVAILABLE AUTHORITIES below; tag them [SOURCE N] / [LIVE SOURCE N]
- If no case supports a proposition directly, write: "No direct authority cited. Verify independently."
- Do NOT invent case names. If unsure of citation, flag it.
- Do NOT use vague references like "it has been held" without naming the court and case

FACTS: {request.facts}
ISSUE: {issue['issue']}
CLIENT POSITION: {request.client_position}
COURT: {request.jurisdiction}

AVAILABLE AUTHORITIES:
{context}

Return ONLY valid JSON:
{{
  "issue":       "The precise legal question raised",
  "rule":        "The applicable statute and leading precedents with full citations [SOURCE N]",
  "application": "How the rule applies specifically to these facts, distinguishing adverse cases [SOURCE N]",
  "conclusion":  "The legal outcome and specific relief available to {request.client_position}"
}}{response_language_directive(request.response_language)}"""


@router.post("/api/argument")
def build_argument(request: ArgumentRequest):
    try:
        # Step 1: Issue identification
        issues_data = parse_json_robust(call_qwen(_issue_prompt(request), max_tokens=500))
        issues      = issues_data.get("issues", [])

        if not issues:
            raise ValueError("Could not identify legal issues. Please provide more detailed facts.")

        # Step 2: IRAC for each issue with live sources
        irac_results   = []
        all_precedents = []

        for idx, issue in enumerate(issues):
            search_q = f"{issue['issue']} {issue.get('area_of_law','')} {request.jurisdiction}"

            # Local library — case law (real precedent) only; bare statutes
            # are retrieved separately so the Constitution does not crowd out
            # citeable authority.
            chunks  = search_chromadb(search_q, top_k=3, exclude_doc_types=["statute"])
            context = build_context(chunks)

            # Live sources for this specific issue
            live = fetch_legal_sources(search_q, max_results=3)
            if live:
                context += "\n\nLIVE SOURCES (Indian Kanoon / SCI / eCourts):\n"
                for i, r in enumerate(live):
                    cit = f" | {r['citation']}" if r.get("citation") else ""
                    context += (
                        f"\nLIVE SOURCE {i+1}: {r['title']}"
                        f"\nCourt: {r['court']} | Year: {r['year']}{cit}"
                        f"\n{r['snippet']}\n---"
                    )

            irac_raw = call_qwen(_irac_prompt(request, issue, context), max_tokens=1200)
            try:
                irac = parse_json_robust(irac_raw)
            except Exception:
                irac = {
                    "issue":       issue["issue"],
                    "rule":        irac_raw[:600],
                    "application": "See analysis above.",
                    "conclusion":  "Further review required.",
                }

            # Collect precedents
            for chunk in chunks:
                meta  = chunk["metadata"]
                court = meta.get("court", "Unknown")
                all_precedents.append({
                    "case_name": meta.get("case_name", "Unknown"), "court": court,
                    "year":      meta.get("year", "Unknown"), "score": chunk["score"],
                    "issue_num": idx + 1, "snippet": chunk["text"][:200],
                    "binding":   "Binding" if court == "Supreme Court of India" else "Persuasive",
                    "source":    "Local Library",
                })

            for r in live:
                all_precedents.append({
                    "case_name": r["title"], "court": r["court"],
                    "year":      r["year"],  "score": 0.8,
                    "issue_num": idx + 1,    "snippet": r["snippet"][:200],
                    "binding":   r["binding"],
                    "source":    r["source"],
                    "url":       r.get("url", ""),
                    "citation":  r.get("citation", ""),
                })

            irac_results.append({
                "issue_title": issue["issue"],
                "area_of_law": issue.get("area_of_law", request.area_of_law),
                "priority":    issue.get("priority", "medium"),
                "irac":        irac,
                "precedents":  [
                    {
                        "case_name": c["metadata"].get("case_name",""),
                        "court":     c["metadata"].get("court",""),
                        "year":      c["metadata"].get("year",""),
                        "score":     c["score"],
                        "source":    "Local Library",
                    }
                    for c in chunks
                ] + [
                    {
                        "case_name": r["title"],
                        "court":     r["court"],
                        "year":      r["year"],
                        "score":     0.8,
                        "source":    r["source"],
                        "url":       r.get("url",""),
                        "citation":  r.get("citation",""),
                    }
                    for r in live
                ],
            })

        # Citation quality disclaimer
        total_cites = sum(
            str(a["irac"]).count("[SOURCE") + str(a["irac"]).count("[LIVE SOURCE")
            for a in irac_results
        )
        disclaimer = None
        if total_cites < len(irac_results):
            disclaimer = (
                "Some arguments lack direct authority from the available sources. "
                "Verify all case citations on SCC Online, Manupatra, or Indian Kanoon "
                "before filing or advancing these submissions."
            )

        all_irac_text = " ".join(
            f"{a['irac'].get('rule','')} {a['irac'].get('application','')}"
            for a in irac_results
        )
        provenance = verify_citations(all_irac_text)
        unverified = [c["citation"] for c in provenance if not c["verified"]]

        output = {
            "facts":           request.facts,
            "jurisdiction":    request.jurisdiction,
            "client_position": request.client_position,
            "total_issues":    len(issues),
            "arguments":       irac_results,
            "all_precedents":  all_precedents,
            "citation_provenance": provenance,
            "unverified_citations": unverified,
            "disclaimer":      disclaimer,
        }

        session = save_session(
            session_type = "argument",
            title        = request.facts[:120],
            input_data   = {
                "facts":              request.facts,
                "jurisdiction":       request.jurisdiction,
                "area_of_law":       request.area_of_law,
                "client_position":    request.client_position,
                "extra_context":     request.extra_context or "",
                "response_language": request.response_language,
            },
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# STREAMING VARIANT
# Emits NDJSON events so the UI can show
# issues / per-issue IRAC progressively.
# Same final output as /api/argument; this
# endpoint also persists the session at the end.
# ─────────────────────────────────────────────
def _ndjson(evt: dict) -> str:
    """Serialise one event as a single NDJSON line."""
    return json.dumps(evt, ensure_ascii=False) + "\n"


def _argument_event_stream(request: "ArgumentRequest") -> Iterator[str]:
    try:
        yield _ndjson({"kind": "stage", "stage": "issues", "status": "active",
                       "message": "Reading instructions and identifying legal issues..."})

        issues_data = parse_json_robust(call_qwen(_issue_prompt(request), max_tokens=500))
        issues      = issues_data.get("issues", [])

        if not issues:
            yield _ndjson({"kind": "error",
                           "message": "Could not identify legal issues. Please provide more detailed facts."})
            return

        yield _ndjson({"kind": "issues", "issues": issues})
        yield _ndjson({"kind": "stage", "stage": "issues", "status": "done",
                       "message": f"{len(issues)} legal issue{'s' if len(issues) != 1 else ''} identified"})
        yield _ndjson({"kind": "stage", "stage": "search", "status": "active",
                       "message": "Searching authorities for each issue..."})

        irac_results   = []
        all_precedents = []

        for idx, issue in enumerate(issues):
            yield _ndjson({"kind": "issue_progress", "index": idx, "phase": "searching",
                           "message": f"Issue {idx + 1}: searching authorities..."})

            search_q = f"{issue['issue']} {issue.get('area_of_law', '')} {request.jurisdiction}"
            chunks   = search_chromadb(search_q, top_k=3, exclude_doc_types=["statute"])
            context  = build_context(chunks)

            live = fetch_legal_sources(search_q, max_results=3)
            if live:
                context += "\n\nLIVE SOURCES (Indian Kanoon / SCI / eCourts):\n"
                for i, r in enumerate(live):
                    cit = f" | {r['citation']}" if r.get("citation") else ""
                    context += (
                        f"\nLIVE SOURCE {i+1}: {r['title']}"
                        f"\nCourt: {r['court']} | Year: {r['year']}{cit}"
                        f"\n{r['snippet']}\n---"
                    )

            yield _ndjson({"kind": "issue_progress", "index": idx, "phase": "drafting",
                           "message": f"Issue {idx + 1}: drafting IRAC submissions..."})

            irac_raw = call_qwen(_irac_prompt(request, issue, context), max_tokens=1200)
            try:
                irac = parse_json_robust(irac_raw)
            except Exception:
                irac = {
                    "issue":       issue["issue"],
                    "rule":        irac_raw[:600],
                    "application": "See analysis above.",
                    "conclusion":  "Further review required.",
                }

            for chunk in chunks:
                meta  = chunk["metadata"]
                court = meta.get("court", "Unknown")
                all_precedents.append({
                    "case_name": meta.get("case_name", "Unknown"), "court": court,
                    "year":      meta.get("year", "Unknown"), "score": chunk["score"],
                    "issue_num": idx + 1, "snippet": chunk["text"][:200],
                    "binding":   "Binding" if court == "Supreme Court of India" else "Persuasive",
                    "source":    "Local Library",
                })
            for r in live:
                all_precedents.append({
                    "case_name": r["title"], "court": r["court"],
                    "year":      r["year"], "score": 0.8,
                    "issue_num": idx + 1,   "snippet": r["snippet"][:200],
                    "binding":   r["binding"],
                    "source":    r["source"],
                    "url":       r.get("url", ""),
                    "citation":  r.get("citation", ""),
                })

            issue_obj = {
                "issue_title": issue["issue"],
                "area_of_law": issue.get("area_of_law", request.area_of_law),
                "priority":    issue.get("priority", "medium"),
                "irac":        irac,
                "precedents":  [
                    {
                        "case_name": c["metadata"].get("case_name", ""),
                        "court":     c["metadata"].get("court", ""),
                        "year":      c["metadata"].get("year", ""),
                        "score":     c["score"],
                        "source":    "Local Library",
                    } for c in chunks
                ] + [
                    {
                        "case_name": r["title"],
                        "court":     r["court"],
                        "year":      r["year"],
                        "score":     0.8,
                        "source":    r["source"],
                        "url":       r.get("url", ""),
                        "citation":  r.get("citation", ""),
                    } for r in live
                ],
            }
            irac_results.append(issue_obj)
            yield _ndjson({"kind": "issue_done", "index": idx, "argument": issue_obj})

        yield _ndjson({"kind": "stage", "stage": "search", "status": "done",
                       "message": "Authorities indexed"})
        yield _ndjson({"kind": "stage", "stage": "irac",   "status": "done",
                       "message": "Submissions drafted"})

        total_cites = sum(
            str(a["irac"]).count("[SOURCE") + str(a["irac"]).count("[LIVE SOURCE")
            for a in irac_results
        )
        disclaimer = None
        if total_cites < len(irac_results):
            disclaimer = (
                "Some arguments lack direct authority from the available sources. "
                "Verify all case citations on SCC Online, Manupatra, or Indian Kanoon "
                "before filing or advancing these submissions."
            )

        all_irac_text = " ".join(
            f"{a['irac'].get('rule','')} {a['irac'].get('application','')}"
            for a in irac_results
        )
        provenance = verify_citations(all_irac_text)
        unverified = [c["citation"] for c in provenance if not c["verified"]]

        output = {
            "facts":           request.facts,
            "jurisdiction":    request.jurisdiction,
            "client_position": request.client_position,
            "total_issues":    len(issues),
            "arguments":       irac_results,
            "all_precedents":  all_precedents,
            "citation_provenance": provenance,
            "unverified_citations": unverified,
            "disclaimer":      disclaimer,
        }

        session = save_session(
            session_type = "argument",
            title        = request.facts[:120],
            input_data   = {
                "facts":             request.facts,
                "jurisdiction":      request.jurisdiction,
                "area_of_law":       request.area_of_law,
                "client_position":   request.client_position,
                "extra_context":     request.extra_context or "",
                "response_language": request.response_language,
            },
            output_data  = output,
            case_id      = request.case_id
        )
        output["session_id"] = session["id"]
        yield _ndjson({"kind": "stage", "stage": "done", "status": "done",
                       "message": "Filed to Instruction Log"})
        yield _ndjson({"kind": "complete",
                       "session_id":     session["id"],
                       "all_precedents": all_precedents,
                       "citation_provenance": provenance,
                       "unverified_citations": unverified,
                       "disclaimer":     disclaimer})

    except Exception as e:
        yield _ndjson({"kind": "error", "message": str(e)})


@router.post("/api/argument/stream")
def build_argument_stream(request: ArgumentRequest):
    return StreamingResponse(
        _argument_event_stream(request),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",   # disable proxy buffering when behind nginx
        },
    )