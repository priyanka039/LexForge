import fitz                    # pymupdf - reads PDFs
import os
import re
import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
import ollama

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_PDFS_FOLDER = "../data/raw_pdfs"
CHROMA_DB_PATH  = "../data/chroma_db"
COLLECTION_NAME = "legal_cases"
EMBED_MODEL     = "nomic-embed-text"

# ─────────────────────────────────────────────
# STEP 1 — EXTRACT TEXT FROM PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file"""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += text + "\n"
    
    doc.close()
    
    # Basic cleanup
    # Remove excessive whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    
    return full_text.strip()


# ─────────────────────────────────────────────
# STEP 2 — SMART LEGAL CHUNKING
# Method 3 + Method 1 combined (our decision)
# ─────────────────────────────────────────────
def chunk_legal_text(text, filename):
    """
    Split legal text into meaningful chunks
    First try legal section splitting
    Then sentence split if section too large
    """
    
    # Extract case metadata from filename
    # e.g. "dtc_v_mazdoor_1991.pdf" → case_name, year
    base_name = os.path.splitext(filename)[0]
    
    # Legal section patterns for Indian judgments
    section_pattern = re.compile(
        r'\n(?=FACTS|HELD|JUDGMENT|CONCLUSION|LEGAL ISSUES|'
        r'BACKGROUND|ORDER|RATIO|HEADNOTES|ISSUE|RELIEF|'
        r'SUBMISSIONS|ANALYSIS|REASONING|FINDINGS)',
        re.IGNORECASE
    )
    
    sections = section_pattern.split(text)
    
    all_chunks = []
    
    for section in sections:
        section = section.strip()
        if len(section) < 80:      # skip tiny fragments
            continue
        
        # Get section heading (first line)
        lines = section.split('\n')
        heading = lines[0].strip() if lines else "GENERAL"
        
        # If section fits in one chunk → keep it whole
        if len(section) <= 800:
            all_chunks.append({
                "text": section,
                "section": heading,
                "case_file": base_name
            })
        
        # If section too large → split by sentences
        else:
            sub_doc = Document(text=section)
            splitter = SentenceSplitter(
                chunk_size=150,
                chunk_overlap=25
            )
            sub_nodes = splitter.get_nodes_from_documents([sub_doc])
            
            for node in sub_nodes:
                all_chunks.append({
                    "text": node.text,
                    "section": heading,    # keep section label!
                    "case_file": base_name
                })
    
    return all_chunks


# ─────────────────────────────────────────────
# STEP 3 — GET EMBEDDING FOR A TEXT
# ─────────────────────────────────────────────
def get_embedding(text):
    """Convert text to vector using nomic-embed-text"""
    result = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return result['embedding']


# ─────────────────────────────────────────────
# STEP 4 — STORE CHUNKS IN CHROMADB
# ─────────────────────────────────────────────
def store_in_chromadb(chunks, case_metadata):
    """Store all chunks with embeddings in ChromaDB"""
    
    # Connect to ChromaDB (creates folder if not exists)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for search
    )
    
    print(f"  Generating embeddings for {len(chunks)} chunks...")
    
    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []
    
    for i, chunk in enumerate(chunks):
        
        # Create unique ID for this chunk
        chunk_id = f"{case_metadata['case_file']}_chunk_{i}"
        
        # Get embedding vector
        embedding = get_embedding(chunk['text'])
        
        # Build metadata
        metadata = {
            **case_metadata,           # case_name, court, year, area_of_law
            "section": chunk['section'],
            "chunk_index": i,
            "chunk_total": len(chunks)
        }
        
        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk['text'])
        metadatas.append(metadata)
        
        print(f"  ✓ Chunk {i+1}/{len(chunks)} embedded — [{chunk['section'][:30]}]")
    
    # Store everything in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"  ✅ Stored {len(chunks)} chunks in ChromaDB")
    return collection


# ─────────────────────────────────────────────
# STEP 5 — PROCESS ALL PDFs IN FOLDER
# ─────────────────────────────────────────────
def ingest_all_pdfs():
    """Process every PDF in the raw_pdfs folder"""
    
    pdf_folder = os.path.abspath(RAW_PDFS_FOLDER)
    pdf_files  = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDFs found in raw_pdfs folder!")
        return
    
    print(f"\n⚖  LEXFORGE INGESTION PIPELINE")
    print(f"Found {len(pdf_files)} PDFs to process\n")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        print(f"{'─'*50}")
        print(f"📄 Processing: {pdf_file}")
        
        # Extract text
        print("  Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(text)} characters, {len(text.split())} words")
        
        # Chunk it
        print("  Chunking text...")
        chunks = chunk_legal_text(text, pdf_file)
        print(f"  Created {len(chunks)} chunks")
        
        # Metadata for this case
        # You can improve this later with auto-extraction
        case_metadata = {
            "case_file":    os.path.splitext(pdf_file)[0],
            "case_name":    os.path.splitext(pdf_file)[0].replace('_', ' '),
            "court":        "Unknown",    # we'll auto-detect later
            "year":         "Unknown",
            "area_of_law":  "General"
        }
        
        # Store in ChromaDB
        store_in_chromadb(chunks, case_metadata)
        print(f"✅ Done: {pdf_file}\n")
    
    # Final summary
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"{'═'*50}")
    print(f"🎉 INGESTION COMPLETE")
    print(f"   Total chunks in ChromaDB: {total}")
    print(f"   Ready for search!")
    print(f"{'═'*50}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ingest_all_pdfs()