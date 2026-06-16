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
# DOCUMENT-TYPE CLASSIFICATION
# Distinguishes case judgments (real precedent)
# from statutes / bare acts / the Constitution
# (reference material). A judgment can be CITED as
# authority; a bare statute cannot, and a large
# reference document like the Constitution will
# otherwise flood every precedent search.
# ═════════════════════════════════════════════
_STATUTE_HINTS = re.compile(
    r"\b(constitution of india|bare act|the .*? act,?\s*\d{4}|"
    r"\bact\s+no\.?\s*\d+|rules?,?\s*\d{4}|code of (civil|criminal) procedure|"
    r"penal code|amendment act|gazette of india|ministry of law)\b",
    re.IGNORECASE,
)
_JUDGMENT_HINTS = re.compile(
    r"\b(versus|vs\.?|\bv\.\b|petitioner|respondent|appellant|honourable|"
    r"hon'?ble|bench|coram|judgment|held that|writ petition|civil appeal|"
    r"criminal appeal|special leave)\b",
    re.IGNORECASE,
)


def classify_doc_type(text: str, filename: str = "") -> str:
    """
    Returns 'statute' for bare acts / Constitution / rules,
    else 'judgment'. Conservative: only calls something a
    statute when statute markers clearly outweigh judgment
    markers, so real case law is never misfiled.
    """
    sample = (text or "")[:4000]
    fname  = (filename or "").lower()

    if any(k in fname for k in ("constitution", "bare_act", "bare-act", "_act_", "act_", "_rules", "penal_code")):
        return "statute"

    statute_markers  = len(_STATUTE_HINTS.findall(sample))
    judgment_markers = len(_JUDGMENT_HINTS.findall(sample))

    if "constitution of india" in sample.lower():
        return "statute"
    if statute_markers >= 2 and statute_markers > judgment_markers:
        return "statute"
    return "judgment"


# Cache: rebuilt automatically whenever the corpus size changes.
_DOC_TYPE_CACHE: dict = {"count": -1, "map": {}}


def corpus_doc_type_map() -> dict:
    """
    Maps each case_file -> 'statute' | 'judgment', decided at the DOCUMENT
    level (filename hint first, then a majority vote across the document's
    chunks). This overrides unreliable per-chunk labels so a large reference
    document like the Constitution is excluded from precedent search as a
    whole, not chunk-by-chunk.
    """
    try:
        cnt = collection.count()
    except Exception:
        return {}

    if cnt == _DOC_TYPE_CACHE["count"] and _DOC_TYPE_CACHE["map"]:
        return _DOC_TYPE_CACHE["map"]

    mapping: dict = {}
    try:
        res   = collection.get(include=["documents", "metadatas"])
        docs  = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []

        by_case: dict = {}
        for doc, meta in zip(docs, metas):
            cf = (meta or {}).get("case_file", "unknown")
            by_case.setdefault(cf, []).append(doc or "")

        for cf, chunk_texts in by_case.items():
            # Strong signal: filename keywords (constitution, bare_act, etc.).
            if classify_doc_type("", cf) == "statute":
                mapping[cf] = "statute"
                continue
            # Otherwise sample several chunks and take a majority vote.
            sample = chunk_texts[:10]
            votes  = [classify_doc_type(t, cf) for t in sample]
            statute_votes = votes.count("statute")
            mapping[cf] = (
                "statute"
                if statute_votes >= max(2, len(votes) // 3)
                else "judgment"
            )
    except Exception:
        pass

    _DOC_TYPE_CACHE["count"] = cnt
    _DOC_TYPE_CACHE["map"]   = mapping
    return mapping


# ═════════════════════════════════════════════
# CHROMADB SEARCH
# Convert a text query → vector → find similar
# chunks stored in ChromaDB.
#
# Relevance is enforced three ways so reference
# material (e.g. an uploaded Constitution) cannot
# crowd out the cases a lawyer actually needs:
#   1. min_score  — drop chunks below a similarity
#                   floor (irrelevant matches gone)
#   2. dedupe     — keep only the single best chunk
#                   per case, so results cover N
#                   DISTINCT authorities, not N
#                   fragments of one document
#   3. doc_type   — optionally restrict to judgments
#                   (real precedent) or statutes
# ═════════════════════════════════════════════
def search_chromadb(
    query: str,
    top_k: int = 4,
    *,
    min_score: float = 0.30,
    doc_types: list | None = None,
    exclude_doc_types: list | None = None,
    dedupe_by_case: bool = True,
    oversample: int = 6,
) -> list:
    """
    Embeds the query with nomic-embed-text, queries ChromaDB, then
    filters for relevance and de-duplicates by case.

    Returns a list of dicts: {text, metadata, score, doc_type}
    sorted best-first. May return fewer than top_k (or none) when
    nothing clears the relevance floor — that is correct behaviour:
    a precise empty result beats a confident irrelevant one.
    """
    result       = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    query_vector = result['embedding']

    # Over-fetch: we will filter/dedupe down to top_k afterwards.
    fetch_n = max(top_k * oversample, top_k)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_n,
        include=['documents', 'metadatas', 'distances'],
    )

    docs  = results['documents'][0] if results.get('documents') else []
    metas = results['metadatas'][0] if results.get('metadatas') else []
    dists = results['distances'][0] if results.get('distances') else []

    want    = {t.lower() for t in doc_types} if doc_types else None
    exclude = {t.lower() for t in exclude_doc_types} if exclude_doc_types else set()

    # Per-DOCUMENT type map takes precedence: a single Constitution/bare-act
    # chunk (e.g. a bare list of territories) carries no statute markers, so
    # classifying it in isolation mislabels it 'judgment' and it leaks into
    # precedent results. Classifying by case_file (filename + majority vote)
    # fixes that even when the stored per-chunk doc_type is stale or wrong.
    type_map = corpus_doc_type_map()

    candidates = []
    for i in range(len(docs)):
        meta  = metas[i] or {}
        score = round(1 - dists[i], 3)
        if score < min_score:
            continue
        cf    = meta.get("case_file", "")
        dtype = (
            type_map.get(cf)
            or meta.get("doc_type")
            or classify_doc_type(docs[i], cf)
        ).lower()
        if want is not None and dtype not in want:
            continue
        if dtype in exclude:
            continue
        candidates.append({
            "text":     docs[i],
            "metadata": meta,
            "score":    score,
            "doc_type": dtype,
        })

    if dedupe_by_case:
        best_by_case = {}
        for c in candidates:
            key = c["metadata"].get("case_file") or c["metadata"].get("case_name") or id(c)
            if key not in best_by_case or c["score"] > best_by_case[key]["score"]:
                best_by_case[key] = c
        candidates = list(best_by_case.values())

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


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
            # Legal prompts pack several full case excerpts + statutes + live
            # sources into the context. At num_ctx=3000 the prompt alone could
            # fill the window, leaving no room to finish the answer — which made
            # memos cut off mid-"Conclusion". 8192 leaves ample headroom.
            "num_ctx":     8192,
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
        dtype    = chunk.get('doc_type', meta.get('doc_type', 'judgment'))
        label    = "STATUTE/PROVISION" if dtype == "statute" else "AUTHORITY"
        context += f"""
SOURCE {i+1} [{label}] (Relevance: {chunk['score']}):
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
        dtype = chunk.get('doc_type', meta.get('doc_type', 'judgment'))
        if dtype == "statute":
            binding = "Statute"
        elif court == 'Supreme Court of India':
            binding = "Binding"
        else:
            binding = "Persuasive"
        result.append({
            "rank":      i + 1,
            "case_name": meta.get('case_name', 'Unknown'),
            "court":     court,
            "year":      meta.get('year', 'Unknown'),
            "section":   meta.get('section', '')[:50],
            "snippet":   chunk['text'][:250],
            "score":     chunk['score'],
            "doc_type":  dtype,
            "binding":   binding,
        })
    return result


# ═════════════════════════════════════════════
# CITATION PROVENANCE
# A citation a lawyer can rely on must be traceable
# to a real document. These helpers tell the UI
# which citations were grounded in the uploaded
# corpus and which are the model speaking from
# memory (and therefore must be verified).
# ═════════════════════════════════════════════
_CASE_NAME_RX = re.compile(
    r"[A-Z][A-Za-z.&'\- ]{1,60}?\s+(?:v\.?|vs\.?|versus)\s+[A-Z][A-Za-z.&'\- ]{1,60}",
)


def _norm_case(s: str) -> str:
    """Lowercase, collapse spaces, drop punctuation — for fuzzy case matching."""
    s = re.sub(r"\b(v\.?|vs\.?|versus)\b", "v", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def corpus_case_names() -> set:
    """All distinct case names currently in the corpus (normalized)."""
    try:
        res = collection.get(include=['metadatas'])
        names = set()
        for m in res.get('metadatas', []) or []:
            n = (m or {}).get('case_name')
            if n:
                names.add(_norm_case(n))
        return names
    except Exception:
        return set()


def verify_citations(text: str, grounded_chunks: list | None = None) -> list:
    """
    Extract 'X v. Y' citations from generated text and mark each as
    verified (present in the retrieved/grounded sources or the corpus)
    or unverified (model memory — verify before use in court).

    Returns: [{"citation": str, "verified": bool}]
    """
    if not text:
        return []

    grounded = set()
    for c in (grounded_chunks or []):
        meta = c.get('metadata', {}) if isinstance(c, dict) else {}
        name = meta.get('case_name')
        if name:
            grounded.add(_norm_case(name))
    if not grounded:
        grounded = corpus_case_names()

    out, seen = [], set()
    for m in _CASE_NAME_RX.finditer(text):
        raw = re.sub(r"\s+", " ", m.group(0)).strip().strip(".,;:")
        key = _norm_case(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        verified = any(key in g or g in key for g in grounded)
        out.append({"citation": raw, "verified": verified})
    return out


def corpus_is_empty() -> bool:
    """True when nothing has been ingested — callers should warn the user."""
    try:
        return collection.count() == 0
    except Exception:
        return True


def response_language_directive(lang: str | None) -> str:
    """
    Appended to model prompts. Values: auto | en | hi
    - en: standard legal English for filings
    - hi: Hindi (Devanagari) for client-facing explanations and simulations when chosen
    - auto: match the user's primary language (English vs Hindi/Devanagari)
    """
    key = (lang or "auto").strip().lower()
    if key == "hi":
        return (
            "\n\nOUTPUT LANGUAGE: Write the entire reply in Hindi (Devanagari script). "
            "Keep conventional English legal labels where standard in India (e.g. Petitioner, Respondent, IRAC section names in English if clearer), "
            "but all substantive analysis, arguments, and judicial observations must be clear Hindi."
        )
    if key == "en":
        return (
            "\n\nOUTPUT LANGUAGE: Write the entire reply in standard legal English suitable for written filings in Indian courts."
        )
    return (
        "\n\nOUTPUT LANGUAGE: Detect the user's primary language from their instructions. "
        "If they wrote mainly in Hindi (Devanagari) or asked for Hindi, respond fully in Hindi as above. "
        "If mainly in English, respond in standard legal English. Avoid unnecessary mixing within the same sentence."
    )