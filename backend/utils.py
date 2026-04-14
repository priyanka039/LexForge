# ─────────────────────────────────────────────
# utils.py
# Shared helper functions used by all 4 routes.
# Import these into any route file you need.
# ─────────────────────────────────────────────

import re
import json
import ollama
from config import collection, EMBED_MODEL, CHAT_MODEL


# ═════════════════════════════════════════════
# CHROMADB SEARCH
# Convert a text query → vector → find similar
# chunks stored in ChromaDB
# ═════════════════════════════════════════════
def search_chromadb(query: str, top_k: int = 4) -> list:
    """
    Embeds the query with nomic-embed-text,
    queries ChromaDB for the top_k closest chunks,
    returns a list of dicts: {text, metadata, score}
    """
    result       = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    query_vector = result['embedding']

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


# ═════════════════════════════════════════════
# QWEN INFERENCE
# Send a prompt to Qwen3:8b and get clean text
# ═════════════════════════════════════════════
def call_qwen(prompt: str, max_tokens: int = 600, system: str = None) -> str:
    """
    Calls Qwen3:8b via Ollama.
    - think=False disables the <think> reasoning block
    - Strips any leftover <think> tags from the response
    - Optional system prompt for persona/role setting
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        think=False,
        options={
            "temperature": 0.1,    # low = more deterministic, better for legal
            "num_predict": max_tokens,
            "num_ctx":     3000,
        }
    )
    raw   = response['message']['content']
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return clean


# ═════════════════════════════════════════════
# ROBUST JSON PARSER
# Handles markdown fences, extra text, think tags
# ═════════════════════════════════════════════
def parse_json_robust(raw: str) -> dict:
    """
    Tries multiple strategies to extract valid JSON
    from Qwen output. Raises ValueError if all fail.

    Strategy order:
    1. Strip think tags + markdown fences, direct parse
    2. Find first {...} block in the text
    3. Find first [...] block in the text
    """
    # 1. Clean
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*',     '', raw)
    raw = raw.strip()

    # 2. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 3. Find first JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Find first JSON array
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return {"data": json.loads(match.group(0))}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON. Raw response: {raw[:300]}")


# ═════════════════════════════════════════════
# BUILD CONTEXT STRING
# Formats retrieved chunks into a prompt-ready
# block that Qwen can reference
# ═════════════════════════════════════════════
def build_context(chunks: list) -> str:
    """
    Converts a list of retrieved chunks into a
    formatted SOURCE 1 / SOURCE 2 / ... block
    for use in prompts.
    """
    context = ""
    for i, chunk in enumerate(chunks):
        meta     = chunk['metadata']
        case     = meta.get('case_name', 'Unknown')
        court    = meta.get('court',     'Unknown')
        year     = meta.get('year',      'Unknown')
        section  = meta.get('section',   'Unknown')
        context += f"""
SOURCE {i+1} (Score: {chunk['score']}):
Case: {case} | Court: {court} | Year: {year}
Section: {section}
Text: {chunk['text']}
---"""
    return context


# ═════════════════════════════════════════════
# FORMAT PRECEDENTS FOR FRONTEND
# Converts raw ChromaDB chunks into the shape
# the frontend renderPrecedents() expects
# ═════════════════════════════════════════════
def format_precedents(chunks: list) -> list:
    """
    Returns a list of dicts ready for the
    frontend's renderPrecedents() function.
    Binding = Supreme Court, else Persuasive.
    """
    result = []
    for i, chunk in enumerate(chunks):
        meta  = chunk['metadata']
        court = meta.get('court', 'Unknown')
        result.append({
            "rank":      i + 1,
            "case_name": meta.get('case_name', 'Unknown'),
            "court":     court,
            "year":      meta.get('year', 'Unknown'),
            "section":   meta.get('section', '')[:50],
            "snippet":   chunk['text'][:250],
            "score":     chunk['score'],
            "binding":   "Binding" if court == 'Supreme Court of India' else "Persuasive"
        })
    return result