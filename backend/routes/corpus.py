# ─────────────────────────────────────────────
# routes/corpus.py
# Feature: Corpus Upload
#
# Flow:
#   User uploads PDF from browser →
#   FastAPI receives the file → save copy under data/raw_pdfs →
#   Extract text → chunk it →
#   Embed each chunk with nomic-embed-text →
#   Store in ChromaDB → return summary
#
# Endpoints:
#   POST /api/corpus/upload  → upload a PDF
#   GET  /api/corpus/list    → list all cases
#   DELETE /api/corpus/delete/{case_file} → remove a case
# ─────────────────────────────────────────────

import os
import re
import tempfile
import fitz                     # pymupdf — reads PDFs
import ollama

from fastapi    import APIRouter, HTTPException, UploadFile, File
from config     import collection, EMBED_MODEL, RAW_PDFS_FOLDER
from llama_index.core             import Document
from llama_index.core.node_parser import SentenceSplitter

router = APIRouter()


def _library_pdf_path(filename: str) -> str:
    """Absolute path for a PDF under raw_pdfs; filename must be basename only."""
    safe = os.path.basename(filename)
    if not safe.lower().endswith(".pdf"):
        safe = f"{safe}.pdf"
    return os.path.join(RAW_PDFS_FOLDER, safe)


def _sliding_window_chunks(text: str, window: int = 2000, overlap: int = 200) -> list:
    """Last-resort splitting when section/sentence parsers yield nothing."""
    text = text.strip()
    out, step = [], max(1, window - overlap)
    i = 0
    while i < len(text):
        piece = text[i : i + window].strip()
        if piece:
            out.append({"text": piece, "section": "GENERAL"})
        i += step
    return out


# ═════════════════════════════════════════════
# HELPER — EXTRACT TEXT FROM PDF BYTES
# ═════════════════════════════════════════════
def extract_text(pdf_bytes: bytes) -> str:
    """Extract and clean text from raw PDF bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        doc       = fitz.open(tmp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    finally:
        os.unlink(tmp_path)

    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    return full_text.strip()


# ═════════════════════════════════════════════
# HELPER — AUTO DETECT COURT + YEAR + CASE NAME
# ═════════════════════════════════════════════
def detect_metadata(text: str, filename: str) -> dict:
    """Auto-detect court, year, and case name from text."""
    sample = text[:2500]

    # Year (2000–2099 plus 1950–1999)
    years = re.findall(r'\b(19[5-9]\d|20\d{2})\b', sample)
    year  = years[0] if years else "Unknown"

    # Court
    court = "Unknown"
    court_patterns = [
        (r'Supreme Court of India|SUPREME COURT',            'Supreme Court of India'),
        (r'High Court of Delhi|Delhi High Court|HC-DEL',     'Delhi High Court'),
        (r'High Court of Bombay|Bombay High Court|HC-BOM',   'Bombay High Court'),
        (r'High Court of Madras|Madras High Court|HC-MAD',   'Madras High Court'),
        (r'High Court of Calcutta|Calcutta High Court',      'Calcutta High Court'),
        (r'High Court of Karnataka|Karnataka High Court|HC-KAR', 'Karnataka High Court'),
        (r'High Court of Kerala|Kerala High Court',          'Kerala High Court'),
        (r'High Court of Allahabad|Allahabad High Court',    'Allahabad High Court'),
        (r'High Court of Gujarat|Gujarat High Court',        'Gujarat High Court'),
        (r'High Court of Jharkhand|Jharkhand High Court',    'Jharkhand High Court'),
        (r'High Court of Punjab|Punjab.*High Court',         'Punjab & Haryana High Court'),
        (r'High Court of Himachal|Himachal Pradesh High Court|HP High Court',
                                                              'HP High Court'),
        (r'National Company Law Appellate|NCLAT',            'NCLAT'),
        (r'National Company Law Tribunal|NCLT',              'NCLT'),
        (r'National Consumer|NCDRC',                         'NCDRC'),
    ]
    for pattern, name in court_patterns:
        if re.search(pattern, sample, re.IGNORECASE):
            court = name
            break

    # Case name — try to find "X v. Y" pattern
    case_name  = os.path.splitext(filename)[0].replace('_', ' ')
    case_match = re.search(
        r'([A-Z][A-Za-z\s\.&]+?)\s+[vV][sS]?\.?\s+([A-Z][A-Za-z\s\.&]+)',
        text[:600]
    )
    if case_match:
        extracted = case_match.group(0).strip()
        if 10 < len(extracted) < 120:
            case_name = extracted

    return {
        "case_name":   case_name,
        "court":       court,
        "year":        year,
        "case_file":   os.path.splitext(filename)[0],
        "area_of_law": "General"
    }


# ═════════════════════════════════════════════
# HELPER — CHUNK LEGAL TEXT
# ═════════════════════════════════════════════
def chunk_legal_text(text: str, case_file: str) -> list:
    """Split legal text into chunks using section-aware splitting."""
    section_pattern = re.compile(
        r'\n(?=FACTS|HELD|JUDGMENT|CONCLUSION|LEGAL ISSUES|BACKGROUND|ORDER|'
        r'RATIO|HEADNOTES|ISSUE|RELIEF|SUBMISSIONS|ANALYSIS|REASONING|'
        r'FINDINGS|AWARD|DECISION|DISCUSSION|OBSERVATIONS)',
        re.IGNORECASE
    )
    sections   = section_pattern.split(text)
    all_chunks = []

    for section in sections:
        section = section.strip()
        if len(section) < 80:
            continue
        lines   = section.split('\n')
        heading = lines[0].strip() if lines else "GENERAL"

        if len(section) <= 800:
            all_chunks.append({"text": section, "section": heading})
        else:
            sub_doc   = Document(text=section)
            splitter  = SentenceSplitter(chunk_size=150, chunk_overlap=25)
            sub_nodes = splitter.get_nodes_from_documents([sub_doc])
            for node in sub_nodes:
                all_chunks.append({"text": node.text, "section": heading})

    stripped = text.strip()
    if not all_chunks and len(stripped) >= 80:
        sub_doc   = Document(text=stripped)
        splitter  = SentenceSplitter(chunk_size=512, chunk_overlap=72)
        sub_nodes = splitter.get_nodes_from_documents([sub_doc])
        for node in sub_nodes:
            nt = (node.text or "").strip()
            if len(nt) >= 50:
                all_chunks.append({"text": nt, "section": "GENERAL"})
    if not all_chunks and len(stripped) >= 50:
        all_chunks = _sliding_window_chunks(stripped)
    return all_chunks


# ═════════════════════════════════════════════
# HELPER — EMBED AND STORE IN CHROMADB
# ═════════════════════════════════════════════
def store_chunks(chunks: list, meta: dict) -> int:
    """Embed each chunk and store in ChromaDB. Returns chunk count."""
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id  = f"{meta['case_file']}_chunk_{i}"
        try:
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk['text'])['embedding']
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Embedding service failed — start Ollama and ensure model "
                    f"'{EMBED_MODEL}' is available (`ollama pull {EMBED_MODEL}`). Error: {e}"
                ),
            ) from e
        metadata  = {
            **meta,
            "section":     chunk['section'],
            "chunk_index": i,
            "chunk_total": len(chunks)
        }
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk['text'])
        metadatas.append(metadata)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    return len(chunks)


# ═════════════════════════════════════════════
# ROUTE 1 — UPLOAD PDF
# POST /api/corpus/upload
# ═════════════════════════════════════════════
@router.post("/api/corpus/upload")
async def upload_corpus(file: UploadFile = File(...)):
    """
    Upload a PDF and add it to the ChromaDB corpus.

    Steps:
    1. Validate file is a PDF
    2. Check if already indexed (prevent duplicates); rewrite raw file if missing on disk
    3. Save a copy under data/raw_pdfs
    4. Extract text with PyMuPDF
    5. Auto-detect court, year, case name
    6. Chunk the text (section-aware)
    7. Embed each chunk with nomic-embed-text
    8. Store everything in ChromaDB (or remove the saved PDF if indexing fails)
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    filename  = os.path.basename(file.filename)
    case_file = os.path.splitext(filename)[0]
    pdf_path  = _library_pdf_path(filename)

    # Read file bytes (needed for disk + processing)
    pdf_bytes = await file.read()

    if len(pdf_bytes) < 64 or not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="File is missing a PDF header (%PDF) or is too small to be valid."
        )

    # Check for duplicate in Chroma — keep raw_pdfs in sync (restore file if missing)
    try:
        existing = collection.get(
            where={"case_file": case_file},
            include=["metadatas"]
        )
        if existing and existing.get('ids') and len(existing['ids']) > 0:
            os.makedirs(RAW_PDFS_FOLDER, exist_ok=True)
            if not os.path.isfile(pdf_path):
                with open(pdf_path, "wb") as out:
                    out.write(pdf_bytes)
            return {
                "status":      "already_exists",
                "message":     f"'{filename}' is already in your corpus.",
                "case_file":   case_file,
                "chunk_count": len(existing['ids']),
                "corpus_size": collection.count()
            }
    except Exception:
        pass   # collection.get with where-filter may fail on empty DB

    os.makedirs(RAW_PDFS_FOLDER, exist_ok=True)
    try:
        with open(pdf_path, "wb") as out:
            out.write(pdf_bytes)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save PDF to library: {e}"
        )

    try:
        # Extract text
        text = extract_text(pdf_bytes)

        if len(text) < 100:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Too little selectable text was extracted. This PDF may be image-only "
                    "(scan) or copy-protected: use OCR or another version."
                )
            )

        # Detect metadata
        meta   = detect_metadata(text, filename)

        # Chunk
        chunks = chunk_legal_text(text, case_file)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not create chunks from the extracted text."
            )

        # Embed and store
        count = store_chunks(chunks, meta)

        return {
            "status":      "success",
            "message":     f"'{filename}' successfully added to your corpus.",
            "case_name":   meta['case_name'],
            "court":       meta['court'],
            "year":        meta['year'],
            "chunk_count": count,
            "corpus_size": collection.count()
        }

    except HTTPException:
        if os.path.isfile(pdf_path):
            try:
                os.unlink(pdf_path)
            except OSError:
                pass
        raise
    except Exception as e:
        if os.path.isfile(pdf_path):
            try:
                os.unlink(pdf_path)
            except OSError:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


# ═════════════════════════════════════════════
# ROUTE 2 — LIST CORPUS
# GET /api/corpus/list
# ═════════════════════════════════════════════
@router.get("/api/corpus/list")
def list_corpus():
    """
    Returns a deduplicated list of all cases currently
    in ChromaDB, with their metadata.
    """
    try:
        results = collection.get(include=['metadatas'])
        metadatas = results.get('metadatas', [])

        # Deduplicate by case_file
        seen  = {}
        cases = []
        for meta in metadatas:
            cf = meta.get('case_file', 'unknown')
            if cf not in seen:
                seen[cf] = True
                cases.append({
                    "case_file":  cf,
                    "case_name":  meta.get('case_name', cf),
                    "court":      meta.get('court', 'Unknown'),
                    "year":       meta.get('year', 'Unknown'),
                    "area_of_law": meta.get('area_of_law', 'General'),
                })

        return {
            "total_cases":  len(cases),
            "total_chunks": collection.count(),
            "cases":        sorted(cases, key=lambda x: x['case_name'])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════
# ROUTE 3 — DELETE CASE
# DELETE /api/corpus/delete/{case_file}
# ═════════════════════════════════════════════
@router.delete("/api/corpus/delete/{case_file}")
def delete_case(case_file: str):
    """
    Removes all chunks for a given case from ChromaDB and deletes the PDF from raw_pdfs.
    """
    try:
        safe_stem = os.path.basename(case_file)
        pdf_path  = os.path.join(RAW_PDFS_FOLDER, f"{safe_stem}.pdf")

        # Find all chunk IDs for this case
        results = collection.get(
            where={"case_file": safe_stem},
            include=["metadatas"]
        )
        ids_to_delete = results.get('ids', [])

        # Allow removing the file from disk when present even if vectors were cleared manually
        if not ids_to_delete and not os.path.isfile(pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"Case '{safe_stem}' not found in corpus."
            )

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

        file_removed = False
        if os.path.isfile(pdf_path):
            try:
                os.unlink(pdf_path)
                file_removed = True
            except OSError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Removed from index but could not delete PDF file: {e}"
                )

        return {
            "status":         "deleted",
            "case_file":      safe_stem,
            "chunks_removed": len(ids_to_delete),
            "file_removed":   file_removed,
            "corpus_size":    collection.count()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))