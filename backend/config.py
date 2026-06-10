# ─────────────────────────────────────────────
# config.py
# Shared configuration and ChromaDB connection.
# Every route file imports from here — do NOT
# create separate DB connections elsewhere.
# ─────────────────────────────────────────────

import chromadb
from pathlib import Path

# ── AI Model names ────────────────────────────
EMBED_MODEL = "nomic-embed-text"   # for vector search
CHAT_MODEL  = "qwen3:8b"           # for text generation

# ── Data paths (resolve from this file so cwd does not matter) ──
_BACKEND_DIR    = Path(__file__).resolve().parent
_DATA_DIR       = _BACKEND_DIR.parent / "data"
CHROMA_DB_PATH  = str(_DATA_DIR / "chroma_db")
RAW_PDFS_FOLDER = str(_DATA_DIR / "raw_pdfs")
COLLECTION_NAME = "legal_cases"

# Single client created once at import time.
# All routes share this same connection.
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection      = _chroma_client.get_collection(COLLECTION_NAME)