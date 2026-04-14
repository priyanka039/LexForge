# ─────────────────────────────────────────────
# routes/corpus.py
# Feature: Corpus Upload
#
# Flow:
#   User uploads PDF from browser →
#   FastAPI receives the file →
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
from config     import collection, EMBED_MODEL
from llama_index.core             import Document
from llama_index.core.node_parser import SentenceSplitter

router = APIRouter()


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

    # Year
    years = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', sample)
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

    return all_chunks


# ═════════════════════════════════════════════
# HELPER — EMBED AND STORE IN CHROMADB
# ═════════════════════════════════════════════
def store_chunks(chunks: list, meta: dict) -> int:
    """Embed each chunk and store in ChromaDB. Returns chunk count."""
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id  = f"{meta['case_file']}_chunk_{i}"
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk['text'])['embedding']
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
    2. Check if already indexed (prevent duplicates)
    3. Extract text with PyMuPDF
    4. Auto-detect court, year, case name
    5. Chunk the text (section-aware)
    6. Embed each chunk with nomic-embed-text
    7. Store everything in ChromaDB
    8. Return summary
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    filename  = file.filename
    case_file = os.path.splitext(filename)[0]

    # Check for duplicate
    try:
        existing = collection.get(
            where={"case_file": case_file},
            include=["metadatas"]
        )
        if existing and existing.get('ids') and len(existing['ids']) > 0:
            return {
                "status":      "already_exists",
                "message":     f"'{filename}' is already in your corpus.",
                "case_file":   case_file,
                "chunk_count": len(existing['ids']),
                "corpus_size": collection.count()
            }
    except Exception:
        pass   # collection.get with where-filter may fail on empty DB

    # Read file bytes
    pdf_bytes = await file.read()

    if len(pdf_bytes) < 1000:
        raise HTTPException(
            status_code=400,
            detail="File appears to be empty or corrupted."
        )

    try:
        # Extract text
        text = extract_text(pdf_bytes)

        if len(text) < 200:
            raise HTTPException(
                status_code=422,
                detail="Could not extract readable text from this PDF. It may be scanned/image-based."
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
        raise
    except Exception as e:
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
    Removes all chunks for a given case from ChromaDB.
    """
    try:
        # Find all chunk IDs for this case
        results = collection.get(
            where={"case_file": case_file},
            include=["metadatas"]
        )
        ids_to_delete = results.get('ids', [])

        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"Case '{case_file}' not found in corpus."
            )

        collection.delete(ids=ids_to_delete)

        return {
            "status":         "deleted",
            "case_file":      case_file,
            "chunks_removed": len(ids_to_delete),
            "corpus_size":    collection.count()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))