# legal_fetch.py — Fetch, score, and ingest Indian legal judgments
# from Indian Kanoon and other public sources.
import re
import time
import urllib.parse
from typing import Optional

import requests
from bs4 import BeautifulSoup

IK_BASE = "https://indiankanoon.org"
IK_API  = "https://api.indiankanoon.org"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

TIMEOUT_SHORT = 10
TIMEOUT_DOC   = 20

_CASE_RX = re.compile(
    r"([A-Z][A-Za-z.&'\- ]{2,50}?)\s+(?:v\.?|vs\.?|versus)\s+([A-Z][A-Za-z.&'\- ]{2,50})",
    re.IGNORECASE,
)

_CONSTITUTIONAL_RX = re.compile(
    r"\b(article\s+\d+[a-z]?|constitution|fundamental right|writ petition under art\.?)\b",
    re.IGNORECASE,
)

_STATUTE_SOURCE_RX = re.compile(
    r"\b(constitution of india|bare act|statute|/act/|legislative\.gov)\b",
    re.IGNORECASE,
)

# Tokens so common in Indian judgments they must not drive relevance.
_LEGAL_STOPWORDS = frozenset({
    "state", "india", "court", "high", "supreme", "the", "and", "ors", "ors.",
    "petitioner", "respondent", "appellant", "versus", "case", "order",
    "judgment", "bench", "honourable", "honble", "union", "government",
})


def _log(msg: str) -> None:
    try:
        print(f"[legal_fetch] {msg}", flush=True)
    except Exception:
        pass


def extract_case_names(text: str, limit: int = 3) -> list[str]:
    """Pull 'X v. Y' style case names from free text."""
    if not text:
        return []
    seen, out = set(), []
    for m in _CASE_RX.finditer(text):
        name = re.sub(r"\s+", " ", m.group(0)).strip(" .,")
        key  = name.lower()
        if len(name) < 12 or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= limit:
            break
    return out


def is_constitutional_query(query: str) -> bool:
    return bool(_CONSTITUTIONAL_RX.search(query or ""))


def _norm_words(s: str, *, drop_stopwords: bool = True) -> set[str]:
    words = {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}
    if drop_stopwords:
        words -= _LEGAL_STOPWORDS
    return words


def score_result_relevance(query: str, result: dict) -> float:
    """Higher = better match for a lawyer looking for citeable case law."""
    q     = (query or "").lower()
    title = (result.get("title") or "").lower()
    snip  = (result.get("snippet") or "").lower()
    score = 0.0

    # Case-name style query — reward distinctive party-name overlap.
    parties = [
        p.strip() for p in re.split(r"\s+v\.?\s+|\s+vs\.?\s+|\s+versus\s+", q, flags=re.IGNORECASE)
        if len(p.strip()) >= 4
    ]
    parties_matched = 0
    for part in parties:
        # Strip generic "State of X" prefix — "Haryana" is the signal.
        core = re.sub(r"^state\s+of\s+", "", part, flags=re.IGNORECASE).strip()
        if len(core) >= 4 and core in title:
            score += 5.0
            parties_matched += 1
        elif len(part) >= 6 and part in title:
            score += 4.0
            parties_matched += 1
    if len(parties) >= 2 and parties_matched >= 2:
        score += 3.0

    # Distinctive token overlap (stopwords removed).
    q_words = _norm_words(q)
    t_words = _norm_words(title)
    s_words = _norm_words(snip)
    if q_words:
        score += len(q_words & t_words) * 2.0
        score += len(q_words & s_words) * 0.5

    # Prefer judgments over reference material.
    source = (result.get("source") or "").lower()
    if _STATUTE_SOURCE_RX.search(title + " " + source):
        score -= 6.0
    if "constitution" in title and not is_constitutional_query(query):
        score -= 5.0

    # Court preference — SC slightly higher for national queries.
    court = result.get("court") or ""
    if court == "Supreme Court of India":
        score += 1.0
    elif court == "Unknown Court":
        score -= 0.5

    # Must have a doc link for import.
    if result.get("doc_id"):
        score += 0.5

    return round(score, 2)


def rank_results(query: str, results: list) -> list:
    """Sort by relevance score descending; drop obvious non-judgment noise."""
    scored = []
    for r in results:
        if not r.get("title"):
            continue
        rel = score_result_relevance(query, r)
        if rel < -2:
            continue
        scored.append({**r, "relevance": rel})
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored


def _classify_court(text: str) -> str:
    if not text:
        return "Unknown Court"
    t = text.lower()
    if "supreme court" in t:
        return "Supreme Court of India"
    court_map = [
        ("punjab", "Punjab and Haryana High Court"),
        ("delhi", "Delhi High Court"),
        ("bombay", "Bombay High Court"),
        ("madras", "Madras High Court"),
        ("calcutta", "Calcutta High Court"),
        ("karnataka", "Karnataka High Court"),
        ("allahabad", "Allahabad High Court"),
        ("gujarat", "Gujarat High Court"),
        ("kerala", "Kerala High Court"),
        ("high court", "High Court"),
    ]
    for kw, name in court_map:
        if kw in t:
            return name
    return "Unknown Court"


def _year(text: str) -> str:
    m = re.search(r"\b(19[5-9]\d|20[0-3]\d)\b", text or "")
    return m.group(1) if m else "Unknown"


def _ik_html_search(query: str, seen: set, raw: list, limit: int) -> None:
    """Append unique IK HTML search hits into raw."""
    html_url = f"{IK_BASE}/search/?formInput={urllib.parse.quote_plus(query.strip())}&pagenum=0"
    try:
        r = SESSION.get(html_url, timeout=TIMEOUT_SHORT)
        if r.status_code != 200:
            return
        soup = BeautifulSoup(r.text, "html.parser")
        for item in soup.find_all(class_="result")[:limit]:
            title_a = (
                item.find("h4", class_="result_title")
                and item.find("h4", class_="result_title").find("a")
            ) or item.find("a", class_="result_title") or item.find("a", href=re.compile(r"/doc/"))
            if not title_a:
                continue
            title = title_a.get_text(" ", strip=True)
            if not title:
                continue
            doc_a = item.find("a", href=re.compile(r"^/doc/\d+/?$"))
            href  = (doc_a or title_a).get("href", "")
            m     = re.search(r"/doc(?:fragment)?/(\d+)", href)
            tid   = m.group(1) if m else ""
            key   = title[:50].lower()
            if key in seen:
                continue
            seen.add(key)
            snip_el = item.find("div", class_="headline") or item.find("p")
            snippet = snip_el.get_text(" ", strip=True)[:400] if snip_el else ""
            src_el  = item.find("span", class_="docsource")
            court   = _classify_court((src_el.get_text() if src_el else "") or title)
            raw.append({
                "title":    title,
                "court":    court,
                "year":     _year(title + " " + snippet),
                "snippet":  snippet,
                "url":      f"{IK_BASE}/doc/{tid}/" if tid else (IK_BASE + href),
                "citation": "",
                "source":   "Indian Kanoon",
                "binding":  "Binding" if court == "Supreme Court of India" else "Persuasive",
                "doc_id":   tid,
            })
    except Exception as e:
        _log(f"IK HTML search failed: {e}")


def search_judgments(query: str, max_results: int = 8) -> list:
    """
    Search Indian Kanoon for judgments only.
    Returns ranked list with doc_id, url, court, year, snippet, relevance.
    """
    if not (query or "").strip():
        return []

    seen, raw = set(), []
    queries   = [query.strip()]

    # For "X v Y" queries, also search distinctive party fragments —
    # IK's search engine matches short name queries far better.
    names = extract_case_names(query)
    for name in names:
        queries.append(name)
        parts = re.split(r"\s+v\.?\s+|\s+vs\.?\s+|\s+versus\s+", name, flags=re.IGNORECASE)
        for p in parts:
            p = re.sub(r"^state\s+of\s+", "", p.strip(), flags=re.IGNORECASE)
            if len(p) >= 5:
                queries.append(p)

    # Official API first — best structured results.
    api_url = (
        f"{IK_API}/search/?formInput={urllib.parse.quote_plus(query.strip())}"
        f"&pagenum=0"
    )
    try:
        r = SESSION.get(api_url, headers={**HEADERS, "Accept": "application/json"},
                        timeout=TIMEOUT_SHORT)
        if r.status_code == 200:
            for doc in r.json().get("docs", [])[:max_results * 2]:
                title = (doc.get("title") or doc.get("headline") or "").strip()
                tid   = str(doc.get("tid") or "").strip()
                if not title or not tid:
                    continue
                key = title[:50].lower()
                if key in seen:
                    continue
                seen.add(key)
                snippet = BeautifulSoup(doc.get("headline", ""), "html.parser").get_text()[:400]
                court   = _classify_court(doc.get("docsource", "") or title)
                raw.append({
                    "title":    title,
                    "court":    court,
                    "year":     (doc.get("publishdate") or "")[:4] or _year(title),
                    "snippet":  snippet.strip(),
                    "url":      f"{IK_BASE}/doc/{tid}/",
                    "citation": doc.get("citation", ""),
                    "source":   "Indian Kanoon",
                    "binding":  "Binding" if court == "Supreme Court of India" else "Persuasive",
                    "doc_id":   tid,
                })
    except Exception as e:
        _log(f"IK API search failed: {e}")

    # HTML search — run multiple query variants, merge, then rank.
    for q in dict.fromkeys(queries):   # preserve order, dedupe
        if len(raw) >= max_results * 3:
            break
        _ik_html_search(q, seen, raw, max_results * 2)
        time.sleep(0.15)

    return rank_results(query, raw)[:max_results]


def fetch_ik_document_text(doc_id: str) -> Optional[dict]:
    """
    Download full judgment text from Indian Kanoon by document id.
    Returns {text, title, court, year, doc_id, url} or None.
    """
    doc_id = str(doc_id or "").strip()
    if not doc_id.isdigit():
        return None

    url = f"{IK_BASE}/doc/{doc_id}/"

    # Try JSON API first.
    try:
        r = SESSION.get(
            f"{IK_API}/doc/{doc_id}/",
            headers={**HEADERS, "Accept": "application/json"},
            timeout=TIMEOUT_DOC,
        )
        if r.status_code == 200:
            data  = r.json()
            title = (data.get("title") or "").strip()
            # API may return HTML in 'doc' or plain text in 'headline'+'doc'.
            raw_doc = data.get("doc") or data.get("headline") or ""
            text    = BeautifulSoup(raw_doc, "html.parser").get_text("\n", strip=True)
            if len(text) >= 500:
                court = _classify_court(data.get("docsource", "") or title)
                return {
                    "text":   text,
                    "title":  title or f"Indian Kanoon Document {doc_id}",
                    "court":  court,
                    "year":   (data.get("publishdate") or "")[:4] or _year(title),
                    "doc_id": doc_id,
                    "url":    url,
                    "source": "Indian Kanoon",
                }
    except Exception as e:
        _log(f"IK API doc fetch failed for {doc_id}: {e}")

    # HTML page fallback.
    try:
        r = SESSION.get(url, timeout=TIMEOUT_DOC)
        if r.status_code != 200:
            return None
        soup  = BeautifulSoup(r.text, "html.parser")
        title = ""
        h1    = soup.find("h1") or soup.find("h2", class_="doc_title")
        if h1:
            title = h1.get_text(strip=True)

        # Main judgment body — IK uses several possible containers.
        body_el = (
            soup.find("div", class_="judgments")
            or soup.find("div", class_="doc")
            or soup.find("div", id="doc")
            or soup.find("div", class_="docsource_main")
        )
        if not body_el:
            # Last resort: all paragraph text on page.
            paras = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
            text  = "\n\n".join(paras)
        else:
            text = body_el.get_text("\n", strip=True)

        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) < 500:
            _log(f"IK HTML doc {doc_id} too short ({len(text)} chars)")
            return None

        meta_el = soup.find("div", class_="docsource_main") or soup.find("div", class_="docsource")
        court   = _classify_court((meta_el.get_text() if meta_el else "") or title)
        return {
            "text":   text,
            "title":  title or f"Indian Kanoon Document {doc_id}",
            "court":  court,
            "year":   _year((meta_el.get_text() if meta_el else "") + " " + title),
            "doc_id": doc_id,
            "url":    url,
            "source": "Indian Kanoon",
        }
    except Exception as e:
        _log(f"IK HTML doc fetch failed for {doc_id}: {e}")
        return None


def pick_best_match(query: str, results: list) -> Optional[dict]:
    """Return the highest-relevance result, or None."""
    ranked = rank_results(query, results)
    return ranked[0] if ranked else None


def extract_citeable_excerpt(text: str, query: str = "", max_chars: int = 600) -> str:
    """
    Pull the most court-ready excerpt: prefer HOLDING / RATIO / CONCLUSION
    sections over background facts.
    """
    if not text:
        return ""
    sections = re.split(
        r"\n(?=(?:HELD|HOLDING|RATIO|CONCLUSION|ORDER|RELIEF|DECISION|"
        r"JUDGMENT|OBSERVATIONS|FINDINGS)\b)",
        text,
        flags=re.IGNORECASE,
    )
    priority = []
    for sec in sections:
        head = sec[:80].upper()
        weight = 0
        for kw, w in [("HELD", 5), ("HOLDING", 5), ("RATIO", 4), ("CONCLUSION", 4),
                      ("ORDER", 3), ("RELIEF", 3), ("OBSERVATIONS", 2)]:
            if kw in head:
                weight = w
                break
        if weight:
            priority.append((weight, sec.strip()))

    if priority:
        priority.sort(key=lambda x: x[0], reverse=True)
        best = priority[0][1]
        if query:
            q_words = _norm_words(query)
            for w, sec in priority:
                if len(_norm_words(sec) & q_words) >= 2:
                    best = sec
                    break
        return best[:max_chars]

    # Fallback: middle of document (skip boilerplate header).
    start = min(len(text) // 4, 2000)
    return text[start:start + max_chars].strip()


def safe_case_file(doc_id: str, title: str) -> str:
    """Filesystem-safe case_file stem for ChromaDB."""
    slug = re.sub(r"[^a-z0-9]+", "_", (title or "").lower())[:60].strip("_")
    return f"ik_{doc_id}_{slug}" if slug else f"ik_{doc_id}"
