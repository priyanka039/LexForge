# ─────────────────────────────────────────────
# routes/sessions_router.py
# Saved Work Sessions
#
# Every time a lawyer runs research, builds an
# argument, tests weaknesses, or simulates court
# the output is automatically saved here so they
# can come back without re-running.
#
# Endpoints:
#   GET    /api/sessions              → list recent sessions
#   GET    /api/sessions/{id}         → get one full session
#   PATCH  /api/sessions/{id}/notes   → save notes on session
#   PATCH  /api/sessions/{id}/case    → move to a case
#   DELETE /api/sessions/{id}         → delete a session
# ─────────────────────────────────────────────

from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from database import (
    get_session, get_sessions,
    update_session_notes, update_session_case,
    delete_session
)

router = APIRouter()


class NotesUpdate(BaseModel):
    notes: str


class CaseAssign(BaseModel):
    case_id: Optional[int] = None   # None = unassign from any case


@router.get("/api/sessions")
def list_sessions(session_type: Optional[str] = None, limit: int = 30):
    """
    List all saved sessions. Optionally filter by type:
    research / argument / opposition / debate
    """
    sessions = get_sessions(session_type=session_type, limit=limit)
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/api/sessions/{session_id}")
def get_one_session(session_id: int):
    """
    Return a full saved session including all
    input and output data, so work can be revisited
    without re-running.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": session}


@router.patch("/api/sessions/{session_id}/notes")
def save_notes(session_id: int, req: NotesUpdate):
    """Save or update the lawyer's own notes on a session."""
    session = update_session_notes(session_id, req.notes)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": session}


@router.patch("/api/sessions/{session_id}/case")
def assign_to_case(session_id: int, req: CaseAssign):
    """Move a session into a case (project)."""
    session = update_session_case(session_id, req.case_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session": session}


@router.delete("/api/sessions/{session_id}")
def remove_session(session_id: int):
    """Permanently delete a saved session."""
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}