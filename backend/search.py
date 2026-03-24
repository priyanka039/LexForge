import chromadb
import ollama
import re

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHROMA_DB_PATH  = "../data/chroma_db"
COLLECTION_NAME = "legal_cases"
EMBED_MODEL     = "nomic-embed-text"
CHAT_MODEL      = "qwen3:8b"

# ─────────────────────────────────────────────
# CONNECT TO CHROMADB
# ─────────────────────────────────────────────
client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

# ─────────────────────────────────────────────
# STEP 1 — SEARCH CHROMADB
# Convert query to vector → find similar chunks
# ─────────────────────────────────────────────
def search_cases(query, top_k=4):
    """
    Convert query to embedding
    Find top_k most similar chunks in ChromaDB
    """
    
    # Convert query to vector
    result = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )
    query_vector = result['embedding']
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    
    # Format results nicely
    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            "text":      results['documents'][0][i],
            "metadata":  results['metadatas'][0][i],
            "score":     round(1 - results['distances'][0][i], 3)
            # distance → similarity score (1.0 = perfect match)
        })
    
    return retrieved


# ─────────────────────────────────────────────
# STEP 2 — GENERATE ANSWER WITH QWEN
# Send retrieved chunks + query to Qwen
# Qwen answers ONLY using those chunks
# ─────────────────────────────────────────────
def generate_answer(query, retrieved_chunks):
    """
    Build prompt with retrieved context
    Ask Qwen to answer using only that context
    """
    
    # Build context string from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk['metadata']
        context += f"""
SOURCE {i+1}:
Case: {meta.get('case_name', 'Unknown')}
Section: {meta.get('section', 'Unknown')}
Relevance Score: {chunk['score']}
Text: {chunk['text']}
---
"""
    
    # The RAG prompt
    prompt = f"""You are LexForge, an expert Indian legal AI assistant.

Answer the legal question below using ONLY the provided case excerpts.
For every claim you make, cite the source number like [SOURCE 1] or [SOURCE 2].
If the answer is not in the sources, say "Not found in current corpus."
Be precise and concise.

QUESTION: {query}

CASE EXCERPTS:
{context}

ANSWER:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={
            "temperature": 0.1,
            "num_predict": 600,
            "num_ctx": 3000,
        }
    )
    
    raw = response['message']['content']
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return clean


# ─────────────────────────────────────────────
# STEP 3 — FULL RAG PIPELINE
# Search + Generate combined
# ─────────────────────────────────────────────
def rag_search(query, top_k=4):
    """
    Full RAG pipeline:
    query → search ChromaDB → generate cited answer
    """
    
    print(f"\n{'═'*55}")
    print(f"  QUERY: {query}")
    print(f"{'═'*55}")
    
    # Search
    print("\n🔍 Searching ChromaDB...")
    retrieved = search_cases(query, top_k)
    
    # Show what was retrieved
    print(f"\n📚 Retrieved {len(retrieved)} chunks:")
    for i, chunk in enumerate(retrieved):
        meta = chunk['metadata']
        print(f"\n  [{i+1}] Score: {chunk['score']}")
        print(f"       Case: {meta.get('case_name', 'Unknown')}")
        print(f"       Section: {meta.get('section', 'Unknown')[:40]}")
        print(f"       Preview: {chunk['text'][:100]}...")
    
    # Generate answer
    print("\n🤖 Generating answer with Qwen...")
    answer = generate_answer(query, retrieved)
    
    print(f"\n{'─'*55}")
    print("  ANSWER:")
    print(f"{'─'*55}")
    print(answer)
    print(f"{'═'*55}\n")
    
    return {
        "query":     query,
        "answer":    answer,
        "sources":   retrieved
    }


# ─────────────────────────────────────────────
# TEST WITH 3 LEGAL QUERIES
# ─────────────────────────────────────────────
if __name__ == "__main__":
    
    print("\n⚖  LEXFORGE — RAG SEARCH TEST")
    print("Testing retrieval + generation pipeline\n")
    
    # Test Query 1 — Employment law
    rag_search("What are the rules for wrongful termination of employment in India?")
    
    # Test Query 2 — Contract law  
    rag_search("What remedies are available for breach of contract?")
    
    # Test Query 3 — Specific to your cases
    rag_search("What did the court hold in the Spicejet case?")