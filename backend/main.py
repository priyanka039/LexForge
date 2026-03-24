from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import ollama
import re
import json

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="LexForge API", version="1.0")

# CORS — allows your HTML frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production, change to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONNECT TO CHROMADB ONCE AT STARTUP
# ─────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="../data/chroma_db")
collection     = chroma_client.get_collection("legal_cases")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "qwen3:8b"

# ─────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# These define the shape of JSON in and out
# ─────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    top_k: int = 4             # how many chunks to retrieve

class ArgumentRequest(BaseModel):
    facts: str
    jurisdiction: str = "High Court of Delhi"
    area_of_law: str  = "General"
    client_position: str = "Plaintiff"

class OppositionRequest(BaseModel):
    argument: str

# ─────────────────────────────────────────────
# HELPER — SEARCH CHROMADB
# ─────────────────────────────────────────────
def search_chromadb(query: str, top_k: int = 4):
    # Convert query to vector
    result = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    query_vector = result['embedding']
    
    # Search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    
    retrieved = []
    for i in range(len(results['documents'][0])):
        retrieved.append({
            "text":     results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "score":    round(1 - results['distances'][0][i], 3)
        })
    
    return retrieved

# ─────────────────────────────────────────────
# HELPER — CALL QWEN
# ─────────────────────────────────────────────
def call_qwen(prompt: str, max_tokens: int = 600):
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={
            "temperature": 0.1,
            "num_predict": max_tokens,
            "num_ctx": 3000,
        }
    )
    raw = response['message']['content']
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return clean

# ─────────────────────────────────────────────
# HELPER — BUILD CONTEXT FROM CHUNKS
# ─────────────────────────────────────────────
def build_context(chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        meta = chunk['metadata']
        context += f"""
SOURCE {i+1} (Score: {chunk['score']}):
Case: {meta.get('case_name', 'Unknown')}
Section: {meta.get('section', 'Unknown')}
Text: {chunk['text']}
---"""
    return context


# ═════════════════════════════════════════════
# ROUTE 1 — HEALTH CHECK
# Test if API is running
# ═════════════════════════════════════════════
@app.get("/")
def health_check():
    total_chunks = collection.count()
    return {
        "status":       "LexForge API is running",
        "corpus_size":  total_chunks,
        "model":        CHAT_MODEL,
        "embed_model":  EMBED_MODEL
    }


# ═════════════════════════════════════════════
# ROUTE 2 — RESEARCH MODE
# POST /api/research
# Takes a query → RAG search → cited answer
# ═════════════════════════════════════════════
@app.post("/api/research")
def research(request: ResearchRequest):
    try:
        # Step 1: Retrieve relevant chunks
        chunks = search_chromadb(request.query, request.top_k)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant cases found")
        
        # Step 2: Build prompt
        context = build_context(chunks)
        prompt = f"""You are LexForge, an expert Indian legal AI assistant.

Answer the legal question using ONLY the provided case excerpts.
Cite sources like [SOURCE 1], [SOURCE 2] for every claim.
If not found in sources, say "Not found in current corpus."
Be precise and structured.

QUESTION: {request.query}

CASE EXCERPTS:
{context}

ANSWER:"""

        # Step 3: Generate answer
        answer = call_qwen(prompt, max_tokens=600)
        
        # Step 4: Format precedents for frontend
        precedents = []
        for i, chunk in enumerate(chunks):
            meta = chunk['metadata']
            precedents.append({
                "rank":      i + 1,
                "case_name": meta.get('case_name', 'Unknown'),
                "court":     meta.get('court', 'Unknown'),
                "year":      meta.get('year', 'Unknown'),
                "section":   meta.get('section', '')[:50],
                "snippet":   chunk['text'][:200],
                "score":     chunk['score'],
                "binding":   "Binding" if meta.get('court') == 'Supreme Court of India' else "Persuasive"
            })
        
        return {
            "query":      request.query,
            "answer":     answer,
            "precedents": precedents,
            "total_sources": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════
# ROUTE 3 — ARGUMENT BUILDER
# POST /api/argument
# Takes case facts → extracts issues → IRAC
# ═════════════════════════════════════════════
@app.post("/api/argument")
def build_argument(request: ArgumentRequest):
    try:
        # Step 1: Extract legal issues from facts
        issue_prompt = f"""You are a legal AI. Extract distinct legal issues from these facts.
Respond with valid JSON only. No explanation, no markdown.

Facts: {request.facts}
Jurisdiction: {request.jurisdiction}
Area of Law: {request.area_of_law}

Respond in this exact format:
{{
  "issues": [
    {{"issue": "issue title", "area_of_law": "area", "priority": "high/medium/low"}},
    {{"issue": "issue title", "area_of_law": "area", "priority": "high/medium/low"}}
  ]
}}"""

        issues_raw = call_qwen(issue_prompt, max_tokens=400)
        
        # Parse JSON — clean any markdown if present
        issues_raw = re.sub(r'```json|```', '', issues_raw).strip()
        issues_data = json.loads(issues_raw)
        issues = issues_data.get('issues', [])
        
        # Step 2: For each issue, RAG search + generate IRAC
        irac_results = []
        
        for issue in issues:
            # Search for relevant precedents for this issue
            search_query = f"{issue['issue']} {request.area_of_law} India"
            chunks = search_chromadb(search_query, top_k=3)
            context = build_context(chunks)
            
            # Generate IRAC
            irac_prompt = f"""You are LexForge, an Indian legal AI.
Generate a structured IRAC argument for this legal issue.
Use ONLY the provided case excerpts as your legal authority.
Cite sources like [SOURCE 1] for every rule and application point.

CASE FACTS: {request.facts}
LEGAL ISSUE: {issue['issue']}
CLIENT POSITION: {request.client_position}
JURISDICTION: {request.jurisdiction}

RELEVANT CASES:
{context}

Respond in this exact JSON format:
{{
  "issue": "State the precise legal question",
  "rule": "State the applicable law and cite precedents",
  "application": "Apply the law to the facts with citations",
  "conclusion": "State the conclusion and remedy"
}}"""

            irac_raw = call_qwen(irac_prompt, max_tokens=800)
            irac_raw = re.sub(r'```json|```', '', irac_raw).strip()
            
            try:
                irac = json.loads(irac_raw)
            except:
                # If JSON fails, return raw text
                irac = {
                    "issue":       issue['issue'],
                    "rule":        irac_raw,
                    "application": "",
                    "conclusion":  ""
                }
            
            irac_results.append({
                "issue_title": issue['issue'],
                "area_of_law": issue['area_of_law'],
                "priority":    issue['priority'],
                "irac":        irac,
                "precedents":  [{"case_name": c['metadata'].get('case_name', ''), 
                                 "score": c['score']} for c in chunks]
            })
        
        return {
            "facts":        request.facts,
            "jurisdiction": request.jurisdiction,
            "total_issues": len(issues),
            "arguments":    irac_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════
# ROUTE 4 — OPPOSITION ENGINE
# POST /api/opposition
# Takes argument → finds weaknesses
# ═════════════════════════════════════════════
@app.post("/api/opposition")
def opposition(request: OppositionRequest):
    try:
        # Search for contrary precedents
        chunks = search_chromadb(request.argument, top_k=4)
        context = build_context(chunks)
        
        prompt = f"""You are opposing counsel in an Indian court.
Analyze this argument and identify every weakness and counter-argument.
Use the provided case excerpts where relevant.
Respond in valid JSON only.

ARGUMENT TO CHALLENGE:
{request.argument}

AVAILABLE CASES:
{context}

Respond in this exact format:
{{
  "risk_level": "HIGH/MODERATE/LOW",
  "overall_confidence": 75,
  "weaknesses": [
    {{"id": "W1", "severity": "HIGH/MODERATE/LOW", "description": "weakness description"}}
  ],
  "counter_arguments": [
    {{"point": "counter argument", "source": "case or law cited"}}
  ],
  "strategy_recommendations": [
    {{"type": "DO/AVOID", "advice": "specific advice"}}
  ]
}}"""

        result_raw = call_qwen(prompt, max_tokens=800)
        result_raw = re.sub(r'```json|```', '', result_raw).strip()
        
        try:
            result = json.loads(result_raw)
        except:
            result = {"error": "Could not parse response", "raw": result_raw}
        
        return {
            "argument":  request.argument,
            "analysis":  result,
            "contrary_precedents": [
                {"case_name": c['metadata'].get('case_name', ''),
                 "snippet":   c['text'][:150],
                 "score":     c['score']} for c in chunks
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════
# RUN SERVER
# ═════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)