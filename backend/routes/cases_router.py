# ─────────────────────────────────────────────
# routes/cases_router.py
# Your Matters / Client Files
#
# A "case" is like a project — it groups all
# your research, arguments, and simulations
# for one client matter together.
#
# Endpoints:
#   GET    /api/cases              → list all cases
#   POST   /api/cases              → create new case
#   GET    /api/cases/{id}         → get one case
#   PUT    /api/cases/{id}         → update case
#   DELETE /api/cases/{id}         → delete case
#   GET    /api/cases/{id}/sessions→ all work in a case
# ─────────────────────────────────────────────

from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel
from typing   import Optional
from database import (
    create_case, get_all_cases, get_case,
    update_case, delete_case, get_sessions
)

router = APIRouter()


class CaseCreate(BaseModel):
    name:        str
    description: Optional[str] = ""
    area_of_law: Optional[str] = "General"
    client_name: Optional[str] = ""


class CaseUpdate(BaseModel):
    name:        Optional[str] = None
    description: Optional[str] = None
    area_of_law: Optional[str] = None
    client_name: Optional[str] = None
    status:      Optional[str] = None


@router.get("/api/cases")
def list_cases():
    """Return all cases ordered by most recently updated."""
    return {"cases": get_all_cases()}


@router.post("/api/cases")
def new_case(req: CaseCreate):
    """Create a new case / matter."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Case name cannot be empty.")
    case = create_case(
        name        = req.name.strip(),
        description = req.description or "",
        area_of_law = req.area_of_law or "General",
        client_name = req.client_name or ""
    )
    return {"case": case}


@router.get("/api/cases/{case_id}")
def get_one_case(case_id: int):
    """Return a single case with its details."""
    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    return {"case": case}


@router.put("/api/cases/{case_id}")
def edit_case(case_id: int, req: CaseUpdate):
    """Update a case's name, description, or status."""
    case = update_case(
        case_id,
        **{k: v for k, v in req.dict().items() if v is not None}
    )
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    return {"case": case}


@router.delete("/api/cases/{case_id}")
def remove_case(case_id: int):
    """
    Delete a case. Sessions linked to it will have
    their case_id set to NULL (not deleted).
    """
    if not get_case(case_id):
        raise HTTPException(status_code=404, detail="Case not found.")
    delete_case(case_id)
    return {"status": "deleted", "case_id": case_id}


@router.get("/api/cases/{case_id}/sessions")
def case_sessions(case_id: int):
    """
    Return all sessions (work) done under this case,
    ordered by most recent first.
    """
    if not get_case(case_id):
        raise HTTPException(status_code=404, detail="Case not found.")
    sessions = get_sessions(case_id=case_id, limit=100)
    return {
        "case_id":  case_id,
        "sessions": sessions,
        "total":    len(sessions)
    }