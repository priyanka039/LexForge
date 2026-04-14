# ─────────────────────────────────────────────
# config.py
# Shared configuration and ChromaDB connection.
# Every route file imports from here — do NOT
# create separate DB connections elsewhere.
# ─────────────────────────────────────────────

import chromadb

# ── AI Model names ────────────────────────────
EMBED_MODEL = "nomic-embed-text"   # for vector search
CHAT_MODEL  = "qwen3:8b"           # for text generation

# ── ChromaDB ──────────────────────────────────
CHROMA_DB_PATH  = "../data/chroma_db"
COLLECTION_NAME = "legal_cases"

# Single client created once at import time.
# All routes share this same connection.
_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection      = _chroma_client.get_collection(COLLECTION_NAME)