# ─────────────────────────────────────────────
# main.py — LexForge API v3.1
#
# IMPORTANT: Do NOT open index.html as a file.
# Always run this server and open:
#   http://localhost:8000/app
#
# Run: python main.py
# ─────────────────────────────────────────────

import os
from pathlib import Path


# ── tiny inline .env loader (no extra deps) ──
# .env is authoritative: it always overrides whatever was in the shell
# environment, otherwise stale values like `SARVAM_API_KEY=your_key_here`
# left over from `.env.example` will silently win and break voice features.
# Placeholder values are explicitly rejected.
_PLACEHOLDER_VALUES = {"", "your_key_here", "your-key-here", "changeme", "placeholder"}


def _load_env_file(path: Path) -> int:
    """Returns the number of keys actually applied from this file."""
    if not path.exists():
        return 0
    applied = 0
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            # Strip inline comments: KEY=value  # note
            if "#" in val:
                val = val.split("#", 1)[0].strip().strip('"').strip("'")
            if not key:
                continue
            if val.lower() in _PLACEHOLDER_VALUES:
                if key == "SARVAM_API_KEY":
                    print(
                        f"[env] {path.name}: SARVAM_API_KEY is still a placeholder "
                        "(e.g. your_key_here) — save a real key from Sarvam and restart the server.",
                        flush=True,
                    )
                continue
            os.environ[key] = val
            applied += 1
    except Exception as e:
        print(f"[env] failed to read {path}: {type(e).__name__}: {e}", flush=True)
    return applied


_PROJECT_ROOT = Path(__file__).parent.parent
_loaded_any = False
for _candidate in (_PROJECT_ROOT / ".env", Path(__file__).parent / ".env"):
    n = _load_env_file(_candidate)
    if n:
        print(f"[env] loaded {n} key(s) from {_candidate}", flush=True)
        _loaded_any = True
    elif _candidate.exists():
        print(f"[env] found {_candidate} but applied 0 keys (placeholders / empty)", flush=True)
if not _loaded_any:
    print(f"[env] no .env file found at {_PROJECT_ROOT / '.env'} or {Path(__file__).parent / '.env'}", flush=True)

# Final guard: if SARVAM_API_KEY is still a placeholder (came from the OS
# env), wipe it so the server logs a clear "key not set" instead of a
# misleading 403.
if os.environ.get("SARVAM_API_KEY", "").strip().lower() in _PLACEHOLDER_VALUES:
    os.environ.pop("SARVAM_API_KEY", None)


from fastapi                  import FastAPI
from fastapi.middleware.cors  import CORSMiddleware
from fastapi.staticfiles      import StaticFiles
from fastapi.responses        import FileResponse
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
from routes.voice           import router as voice_router
from moot.router            import router as moot_router
from moot.router            import router as moot_router

from config   import collection, CHAT_MODEL, EMBED_MODEL
from database import init_db

app = FastAPI(title="LexForge", version="3.1")

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
app.include_router(voice_router)
app.include_router(moot_router)
app.include_router(moot_router)


@app.on_event("startup")
def startup():
    """Initialise database tables on startup."""
    init_db()
    print("\n" + "="*52)
    print("  LexForge is running.")
    print("  Open http://localhost:8000/app in your browser.")
    print("  Do NOT open index.html as a file:// URL.")
    print("="*52 + "\n")


@app.get("/")
def health_check():
    return {
        "status":      "LexForge is running",
        "library_size": collection.count(),
        "model":        CHAT_MODEL,
        "version":      "3.1"
    }


# ── Serve the frontend from FastAPI ──────────
# This is what fixes the "buttons don't work" issue.
# When opened as file://, browsers block all fetch()
# calls to localhost. Serving from the same origin fixes it.
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    @app.get("/app", response_class=FileResponse)
    def serve_app():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/landing", response_class=FileResponse)
    def serve_landing():
        return FileResponse(FRONTEND_DIR / "landing.html")

    @app.get("/moot-chamber", response_class=FileResponse)
    def serve_moot_chamber():
        return FileResponse(FRONTEND_DIR / "moot_chamber.html")

    @app.get("/moot-chamber", response_class=FileResponse)
    def serve_moot_chamber():
        return FileResponse(FRONTEND_DIR / "moot_chamber.html")

    # Serve CSS, JS, images etc.
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    print(f"Frontend served from: {FRONTEND_DIR}")
else:
    print(f"WARNING: Place index.html in a 'frontend/' folder next to 'backend/'")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)