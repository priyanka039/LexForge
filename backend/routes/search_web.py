# routes/search_web.py - Multi-source Indian Legal Search
# FIX: Wrapped all network calls in try/except with short timeouts
# so a failed live search never crashes the parent route with a 500.
import re, time, urllib.parse, requests
from bs4      import BeautifulSoup
from fastapi  import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

IK_BASE      = "https://indiankanoon.org"
SCI_BASE     = "https://www.sci.gov.in"
ECOURTS_BASE = "https://judgments.ecourts.gov.in"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",
}

# ── FIX 1: Use a session with a short global timeout ──────────────────────────
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ── FIX 2: Short timeouts everywhere so a blocked request fails fast ──────────
TIMEOUT_SHORT = 8    # seconds for API calls
TIMEOUT_HTML  = 10   # seconds for HTML scrapes


class LiveSearchRequest(BaseModel):
    query:       str
    max_results: int = 6


# ─── Court Classifier ────────────────────────
def _classify_court(text):
    if not text: return "Unknown Court"
    t = text.lower()
    if "supreme court" in t: return "Supreme Court of India"
    court_map = [
        ("allahabad",      "Allahabad High Court"),
        ("bombay",         "Bombay High Court"),
        ("calcutta",       "Calcutta High Court"),
        ("delhi",          "Delhi High Court"),
        ("madras",         "Madras High Court"),
        ("gujarat",        "Gujarat High Court"),
        ("kerala",         "Kerala High Court"),
        ("karnataka",      "Karnataka High Court"),
        ("punjab",         "Punjab and Haryana High Court"),
        ("rajasthan",      "Rajasthan High Court"),
        ("madhya pradesh", "Madhya Pradesh High Court"),
        ("andhra",         "Andhra Pradesh High Court"),
        ("telangana",      "Telangana High Court"),
        ("hyderabad",      "Telangana High Court"),
        ("patna",          "Patna High Court"),
        ("gauhati",        "Gauhati High Court"),
        ("jharkhand",      "Jharkhand High Court"),
        ("chhattisgarh",   "Chhattisgarh High Court"),
        ("uttarakhand",    "Uttarakhand High Court"),
        ("himachal",       "Himachal Pradesh High Court"),
        ("orissa",         "Orissa High Court"),
        ("nclat",          "NCLAT"),
        ("nclt",           "NCLT"),
        ("ncdrc",          "NCDRC"),
        ("itat",           "Income Tax Appellate Tribunal"),
        ("sat",            "Securities Appellate Tribunal"),
        ("high court",     "High Court"),
        ("tribunal",       "Tribunal"),
    ]
    for kw, name in court_map:
        if kw in t: return name
    return "Unknown Court"


def _year(text):
    m = re.search(r'\b(19[5-9]\d|20[0-3]\d)\b', text or "")
    return m.group(1) if m else "Unknown"


# ─── Source 1: Indian Kanoon API ─────────────
def _ik_api(query, max_results):
    url = (
        f"https://api.indiankanoon.org/search/"
        f"?formInput={urllib.parse.quote_plus(query)}&pagenum=0"
    )
    try:
        r = SESSION.get(
            url,
            headers={**HEADERS, "Accept": "application/json"},
            timeout=TIMEOUT_SHORT   # FIX: was missing timeout
        )
        if r.status_code != 200:
            return []
        docs = r.json().get("docs", [])
        out  = []
        for doc in docs[:max_results]:
            title = doc.get("title", doc.get("headline", "")).strip()
            tid   = doc.get("tid", "")
            if not title or not tid:
                continue
            snippet = BeautifulSoup(doc.get("headline", ""), "html.parser").get_text()[:300]
            year    = (doc.get("publishdate", "") or "")[:4] or _year(title)
            court   = _classify_court(doc.get("docsource", "") or title)
            out.append({
                "title":   title,
                "court":   court,
                "year":    year,
                "snippet": snippet.strip(),
                "url":     f"{IK_BASE}/doc/{tid}/",
                "citation": doc.get("citation", ""),
                "source":  "Indian Kanoon (Official API)",
                "binding": "Binding" if court == "Supreme Court of India" else "Persuasive",
                "doc_id":  tid,
            })
        return out
    except Exception:
        return []   # FIX: was bare 'except: return []' — explicit is better


# ─── Source 2: Indian Kanoon HTML ────────────
def _ik_html(query, max_results):
    url = f"{IK_BASE}/search/?formInput={urllib.parse.quote_plus(query)}&pagenum=0"
    try:
        r = SESSION.get(url, timeout=TIMEOUT_HTML)   # FIX: was timeout=14 but no keyword
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for div in soup.find_all("div", class_="result")[:max_results]:
            a = (
                div.find("a", class_="result_title")
                or div.find("a", href=re.compile(r'^/doc/'))
            )
            if not a:
                continue
            title = a.get_text(strip=True)
            href  = a.get("href", "")
            link  = (IK_BASE + href) if href.startswith("/") else href
            if not link.startswith("http"):
                continue
            snip_t  = div.find("p") or div.find("div", class_="snippet")
            snippet = re.sub(
                r"\[[\w\s,\.]+\]", "",
                snip_t.get_text(strip=True)[:320] if snip_t else ""
            )
            meta_t = (
                div.find("div", class_="docsource_main")
                or div.find("div", class_="docsource")
            )
            court = _classify_court(
                (meta_t.get_text(" ", strip=True) if meta_t else "") or title
            )
            year = _year((meta_t.get_text() if meta_t else "") + " " + title)
            out.append({
                "title":   title,
                "court":   court,
                "year":    year,
                "snippet": snippet.strip(),
                "url":     link,
                "citation": "",
                "source":  "Indian Kanoon",
                "binding": "Binding" if court == "Supreme Court of India" else "Persuasive",
            })
        return out
    except Exception:
        return []


# ─── Source 3: SCI Portal ────────────────────
def _sci_portal(query, max_results):
    try:
        r = SESSION.get(
            f"{SCI_BASE}/judgments/?title={urllib.parse.quote_plus(query)}",
            timeout=TIMEOUT_SHORT   # FIX: was timeout=10 without keyword
        )
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for a in soup.find_all("a", href=re.compile(r'\.pdf|/judgment', re.I))[:max_results]:
            title = a.get_text(strip=True)
            if len(title) < 10:
                continue
            href  = a.get("href", "")
            link  = href if href.startswith("http") else SCI_BASE + href
            court = "Supreme Court of India"
            year  = _year(title)
            out.append({
                "title":   title,
                "court":   court,
                "year":    year,
                "snippet": f"Supreme Court judgment. Source: sci.gov.in",
                "url":     link,
                "citation": "",
                "source":  "Supreme Court of India (sci.gov.in)",
                "binding": "Binding",
            })
        return out
    except Exception:
        return []


# ─── Source 4: eCourts Portal ────────────────
def _ecourts_portal(query, max_results):
    try:
        r = SESSION.get(
            f"{ECOURTS_BASE}/judgments?search={urllib.parse.quote_plus(query)}",
            timeout=TIMEOUT_SHORT
        )
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for a in soup.find_all("a", href=re.compile(r'judgment|case', re.I))[:max_results]:
            title = a.get_text(strip=True)
            if len(title) < 10:
                continue
            href  = a.get("href", "")
            link  = href if href.startswith("http") else ECOURTS_BASE + href
            court = _classify_court(title)
            year  = _year(title)
            out.append({
                "title":   title,
                "court":   court,
                "year":    year,
                "snippet": f"eCourts judgment portal result.",
                "url":     link,
                "citation": "",
                "source":  "eCourts Portal",
                "binding": "Binding" if court == "Supreme Court of India" else "Persuasive",
            })
        return out
    except Exception:
        return []


# ─── Source 5: DuckDuckGo fallback ───────────
def _ddg(query, max_results):
    try:
        search_q = urllib.parse.quote_plus(f"{query} site:indiankanoon.org OR site:sci.gov.in")
        r = SESSION.get(
            f"https://html.duckduckgo.com/html/?q={search_q}",
            timeout=TIMEOUT_HTML
        )
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for result in soup.find_all("div", class_="result")[:max_results]:
            a = result.find("a", class_="result__a")
            if not a:
                continue
            title   = a.get_text(strip=True)
            url     = a.get("href", "")
            snip_el = result.find("a", class_="result__snippet")
            snippet = snip_el.get_text(strip=True)[:300] if snip_el else ""
            court   = _classify_court(title + " " + snippet)
            year    = _year(title + " " + snippet)
            out.append({
                "title":   title,
                "court":   court,
                "year":    year,
                "snippet": snippet,
                "url":     url,
                "citation": "",
                "source":  "Web Search (DuckDuckGo)",
                "binding": "Binding" if court == "Supreme Court of India" else "Persuasive",
            })
        return out
    except Exception:
        return []


# ─── Constitutional Provisions Database ──────
CONSTITUTION_INDEX = {
    "article 14": (
        "Article 14 — Equality Before Law",
        "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.",
        "Reasonable classification test. Arbitrariness test from E.P. Royappa v. State of Tamil Nadu (1974). Maneka Gandhi expanded the scope."
    ),
    "article 19": (
        "Article 19 — Freedom of Speech and Expression",
        "All citizens have the right to freedom of speech and expression, to assemble peaceably, to form associations, to move freely, to reside, and to practise any profession.",
        "Subject to reasonable restrictions under Art. 19(2) to (6). Shreya Singhal v. Union of India (2015) on online speech."
    ),
    "article 21": (
        "Article 21 — Protection of Life and Personal Liberty",
        "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "Expanded by Maneka Gandhi v. Union of India (1978). Includes dignity, health, livelihood, fair trial. Puttaswamy v. Union of India on privacy."
    ),
    "article 22": (
        "Article 22 — Protection Against Arrest and Detention",
        "No person arrested shall be detained without being informed of the grounds of arrest, nor denied the right to consult an advocate.",
        "Must be produced before magistrate within 24 hours. D.K. Basu v. State of West Bengal on arrest guidelines."
    ),
    "article 32": (
        "Article 32 — Right to Constitutional Remedies",
        "The right to move the Supreme Court for enforcement of Fundamental Rights is itself a Fundamental Right.",
        "Five writs: habeas corpus, mandamus, prohibition, quo warranto, certiorari. Only SC under Art. 32; High Courts under Art. 226."
    ),
    "article 226": (
        "Article 226 — High Court Writ Jurisdiction",
        "Every High Court has the power to issue writs to any person or authority within its territorial jurisdiction.",
        "Wider than Art. 32 as it extends to any legal right, not just Fundamental Rights."
    ),
    "natural justice": (
        "Principles of Natural Justice",
        "Two cardinal principles: audi alteram partem (hear the other side) and nemo judex in causa sua (no person shall be a judge in their own cause).",
        "Apply to all quasi-judicial and administrative bodies. A.K. Kraipak v. Union of India (1969)."
    ),
    "wrongful termination": (
        "Wrongful Termination — Employment Law",
        "Termination without due process (natural justice, domestic enquiry, proper notice) violates the Industrial Disputes Act, 1947.",
        "For government employees: Art. 311 protection. L. Robert D'Souza v. Executive Engineer."
    ),
    "bail": (
        "Bail Jurisprudence — Section 437/438/439 CrPC (now 480/482/483 BNSS)",
        "Three types: regular bail (post-arrest), anticipatory bail (pre-arrest), and interim bail.",
        "Triple test: flight risk, tampering with evidence, repeat offence. Satender Kumar Antil v. CBI (2022)."
    ),
    "contract": (
        "Contract Law — Indian Contract Act, 1872",
        "A valid contract requires offer, acceptance, consideration, free consent, lawful object, and capacity.",
        "Breach remedies: specific performance, damages, injunction, rescission."
    ),
    "habeas corpus": (
        "Habeas Corpus — Writ for Unlawful Detention",
        "A writ commanding the person detaining another to bring the body before the court and show lawful cause for the detention.",
        "Available under Art. 32 (Supreme Court) and Art. 226 (High Courts)."
    ),
    "mandamus": (
        "Mandamus — Writ to Compel Performance of Public Duty",
        "A writ commanding a public authority to perform a public or statutory duty which it has failed or refused to perform.",
        "Will not lie against a private person or purely discretionary powers."
    ),
    "fundamental rights": (
        "Fundamental Rights — Part III, Articles 12 to 35",
        "Part III guarantees Fundamental Rights against state action: equality, freedom, against exploitation, religion, culture, and constitutional remedies.",
        "Basic Structure doctrine (Kesavananda Bharati 1973) prevents Parliament from abrogating Fundamental Rights."
    ),
    "pocso": (
        "POCSO Act, 2012",
        "Criminalises sexual assault, harassment, and use of children for pornography. Special Courts mandated.",
        "Mandatory reporting under s.19."
    ),
    "dowry": (
        "Dowry Prohibition Act, 1961 and Section 498A IPC",
        "Section 498A IPC: husband or relatives subjecting a woman to cruelty in connection with demands for dowry.",
        "Arnesh Kumar v. State of Bihar (2014) on arrest guidelines."
    ),
}


def _constitution_lookup(query):
    q   = query.lower()
    out = []
    for key, (title, text, note) in CONSTITUTION_INDEX.items():
        if any(part in q for part in key.split()):
            out.append({
                "title":    title,
                "court":    "Constitution of India / Statute",
                "year":     "1950",
                "snippet":  text + "  " + note,
                "url":      "https://legislative.gov.in/constitution-of-india",
                "citation": title,
                "source":   "Indian Constitution / Statute",
                "binding":  "Binding",
            })
            if len(out) >= 2:
                break
    return out


# ─── Main Public Function ─────────────────────
def fetch_legal_sources(query: str, max_results: int = 6) -> list:
    """
    FIX: Each individual scraper already returns [] on failure.
    This wrapper adds an outer try/except so even an unexpected
    error in the orchestration logic cannot propagate to the caller.
    """
    try:
        seen, all_r = set(), []

        # Always check constitutional provisions first (no network)
        for r in _constitution_lookup(query):
            k = r["title"][:40].lower()
            if k not in seen:
                seen.add(k)
                all_r.append(r)

        sources = [_ik_api, _ik_html, _sci_portal, _ecourts_portal, _ddg]
        for fn in sources:
            if len(all_r) >= max_results:
                break
            try:
                for r in fn(query, max_results):
                    k = r["title"][:40].lower()
                    if k not in seen and r["title"]:
                        seen.add(k)
                        all_r.append(r)
                        if len(all_r) >= max_results:
                            break
            except Exception:
                pass   # this source failed; try next one
            time.sleep(0.2)

        return all_r[:max_results]

    except Exception:
        # FIX: top-level safety net — live search failure must NEVER crash
        # the argument/debate/opposition routes that call this function.
        return []


# Backward-compatible alias
def scrape_indian_kanoon(query: str, max_results: int = 6) -> list:
    return fetch_legal_sources(query, max_results)


# ─── API Route ────────────────────────────────
@router.post("/api/search/live")
def live_precedent_search(req: LiveSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = fetch_legal_sources(req.query.strip(), req.max_results)
    return {"query": req.query, "results": results, "count": len(results)}