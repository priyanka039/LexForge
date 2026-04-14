import fitz
import os
import re
import sys
import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import ollama

RAW_PDFS_FOLDER = "../data/raw_pdfs"
CHROMA_DB_PATH  = "../data/chroma_db"
COLLECTION_NAME = "legal_cases"
EMBED_MODEL     = "nomic-embed-text"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    return full_text.strip()


def detect_metadata(text, filename):
    sample = text[:2500]

    years = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', sample)
    year  = years[0] if years else "Unknown"

    court = "Unknown"
    court_patterns = [
        (r'Supreme Court of India|SUPREME COURT OF INDIA',         'Supreme Court of India'),
        (r'High Court of Delhi|Delhi High Court|HC-DEL',           'Delhi High Court'),
        (r'High Court of Bombay|Bombay High Court|HC-BOM',         'Bombay High Court'),
        (r'High Court of Madras|Madras High Court|HC-MAD',         'Madras High Court'),
        (r'High Court of Calcutta|Calcutta High Court|HC-CAL',     'Calcutta High Court'),
        (r'High Court of Karnataka|Karnataka High Court|HC-KAR',   'Karnataka High Court'),
        (r'High Court of Kerala|Kerala High Court',                 'Kerala High Court'),
        (r'High Court of Allahabad|Allahabad High Court',          'Allahabad High Court'),
        (r'High Court of Gujarat|Gujarat High Court',               'Gujarat High Court'),
        (r'High Court of Rajasthan|Rajasthan High Court',          'Rajasthan High Court'),
        (r'High Court of Punjab|Punjab.*High Court',                'Punjab & Haryana High Court'),
        (r'High Court of Jharkhand|Jharkhand High Court|HC-JHA',   'Jharkhand High Court'),
        (r'High Court of Himachal|Himachal Pradesh High Court',    'HP High Court'),
        (r'High Court of Andhra|Andhra Pradesh High Court',        'AP High Court'),
        (r'High Court of Telangana|Telangana High Court',          'Telangana High Court'),
        (r'National Company Law Appellate|NCLAT',                  'NCLAT'),
        (r'National Company Law Tribunal|NCLT',                    'NCLT'),
        (r'National Consumer|NCDRC',                               'NCDRC'),
        (r'Income Tax Appellate|ITAT',                             'ITAT'),
        (r'Armed Forces Tribunal',                                  'AFT'),
    ]
    for pattern, name in court_patterns:
        if re.search(pattern, sample, re.IGNORECASE):
            court = name
            break

    if court == "Unknown":
        fname = filename.lower()
        if any(x in fname for x in ['sc', 'supreme']):
            court = 'Supreme Court of India'
        elif any(x in fname for x in ['delhi', 'del']):
            court = 'Delhi High Court'
        elif any(x in fname for x in ['bombay', 'bom']):
            court = 'Bombay High Court'
        elif 'jharkhand' in fname:
            court = 'Jharkhand High Court'
        elif any(x in fname for x in ['karnataka', 'kar']):
            court = 'Karnataka High Court'

    case_name = os.path.splitext(filename)[0].replace('_', ' ')
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


def chunk_legal_text(text, filename):
    base_name = os.path.splitext(filename)[0]
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
            all_chunks.append({"text": section, "section": heading, "case_file": base_name})
        else:
            sub_doc   = Document(text=section)
            splitter  = SentenceSplitter(chunk_size=150, chunk_overlap=25)
            sub_nodes = splitter.get_nodes_from_documents([sub_doc])
            for node in sub_nodes:
                all_chunks.append({"text": node.text, "section": heading, "case_file": base_name})

    return all_chunks


def get_embedding(text):
    result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return result['embedding']


def store_in_chromadb(chunks, case_metadata, collection):
    print(f"  Embedding {len(chunks)} chunks...")
    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id  = f"{case_metadata['case_file']}_chunk_{i}"
        embedding = get_embedding(chunk['text'])
        metadata  = {**case_metadata, "section": chunk['section'],
                     "chunk_index": i, "chunk_total": len(chunks)}
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk['text'])
        metadatas.append(metadata)
        print(f"  ✓ {i+1}/{len(chunks)} [{chunk['section'][:35]}]")

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"  ✅ Stored {len(chunks)} chunks")


def ingest_all_pdfs(force_reingest=False):
    pdf_folder = os.path.abspath(RAW_PDFS_FOLDER)
    pdf_files  = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDFs found in raw_pdfs folder!")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if force_reingest:
        print("Force mode: deleting existing collection...")
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Old collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    existing = set()
    if not force_reingest:
        try:
            res = collection.get(include=['metadatas'])
            for m in res['metadatas']:
                existing.add(m.get('case_file', ''))
        except Exception:
            pass

    print(f"\n LEXFORGE INGESTION PIPELINE v2")
    print(f"PDFs found: {len(pdf_files)} | Already indexed: {len(existing)}\n")

    for pdf_file in pdf_files:
        base = os.path.splitext(pdf_file)[0]
        if base in existing and not force_reingest:
            print(f"Skipping (already indexed): {pdf_file}")
            continue

        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"{'─'*52}")
        print(f"Processing: {pdf_file}")

        text = extract_text_from_pdf(pdf_path)
        print(f"  {len(text.split())} words extracted")

        meta = detect_metadata(text, pdf_file)
        print(f"  Court: {meta['court']} | Year: {meta['year']}")
        print(f"  Case:  {meta['case_name']}")

        chunks = chunk_legal_text(text, pdf_file)
        print(f"  Chunks: {len(chunks)}")

        store_in_chromadb(chunks, meta, collection)
        print(f"Done: {pdf_file}\n")

    total = collection.count()
    print(f"{'='*52}")
    print(f"INGESTION COMPLETE — Total chunks: {total}")
    print(f"{'='*52}")


if __name__ == "__main__":
    force = '--force' in sys.argv
    if force:
        print("Force reingest mode — re-indexing all PDFs with correct metadata")
    ingest_all_pdfs(force_reingest=force)