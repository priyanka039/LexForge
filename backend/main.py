# ─────────────────────────────────────────────
# main.py — LexForge API v3.0
#
# Routes:
#   GET    /                          health check
#   POST   /api/research              find precedents
#   POST   /api/argument              build argument
#   POST   /api/opposition            test weaknesses
#   POST   /api/debate                simulate court
#   POST   /api/corpus/upload         add to library
#   GET    /api/corpus/list           list library
#   DELETE /api/corpus/delete/{name}  remove from library
#   POST   /api/export/report         download PDF
#   GET    /api/cases                 list all matters
#   POST   /api/cases                 new matter
#   GET    /api/cases/{id}            get matter
#   PUT    /api/cases/{id}            update matter
#   DELETE /api/cases/{id}            delete matter
#   GET    /api/cases/{id}/sessions   work in matter
#   GET    /api/sessions              recent work
#   GET    /api/sessions/{id}         get saved work
#   PATCH  /api/sessions/{id}/notes   save notes
#   PATCH  /api/sessions/{id}/case    move to matter
#   DELETE /api/sessions/{id}         delete work
#   POST   /api/search/live           Indian Kanoon search
# ─────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes.research        import router as research_router
from routes.argument        import router as argument_router
from routes.opposition      import router as opposition_router
from routes.debate          import router as debate_router
from routes.corpus          import router as corpus_router
from routes.export          import router as export_router
from routes.cases_router    import router as cases_router
from routes.sessions_router import router as sessions_router
from routes.search_web      import router as search_web_router

from config   import collection, CHAT_MODEL, EMBED_MODEL
from database import init_db

app = FastAPI(title="LexForge", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(research_router)
app.include_router(argument_router)
app.include_router(opposition_router)
app.include_router(debate_router)
app.include_router(corpus_router)
app.include_router(export_router)
app.include_router(cases_router)
app.include_router(sessions_router)
app.include_router(search_web_router)


@app.on_event("startup")
def startup():
    """Initialise database tables on startup."""
    init_db()
    print("LexForge database ready.")


@app.get("/")
def health_check():
    return {
        "status":      "LexForge is running",
        "library_size": collection.count(),
        "model":        CHAT_MODEL,
        "version":      "3.0"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)