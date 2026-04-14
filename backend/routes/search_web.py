# ─────────────────────────────────────────────
# routes/search_web.py
# Live Precedent Search from Indian Kanoon
#
# Searches indiankanoon.org for the latest
# judgments matching the lawyer's query.
# Results are returned alongside the local
# case library results.
#
# Endpoint:
#   POST /api/search/live
# ─────────────────────────────────────────────

import re
import time
import urllib.parse
import requests
from bs4        import BeautifulSoup
from fastapi    import APIRouter, HTTPException
from pydantic   import BaseModel

router = APIRouter()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

IK_SEARCH  = "https://indiankanoon.org/search/?formInput={query}&pagenum=0"
IK_BASE    = "https://indiankanoon.org"


class LiveSearchRequest(BaseModel):
    query:       str
    max_results: int = 5


def scrape_indian_kanoon(query: str, max_results: int = 5) -> list:
    """
    Search Indian Kanoon and return top results.
    Each result has: title, court, year, snippet, url, citation
    Returns empty list on any error (graceful fallback).
    """
    try:
        url      = IK_SEARCH.format(query=urllib.parse.quote(query))
        response = requests.get(url, headers=HEADERS, timeout=12)

        if response.status_code != 200:
            return []

        soup    = BeautifulSoup(response.text, 'html.parser')
        results = []

        for result in soup.find_all('div', class_='result')[:max_results]:
            try:
                # Title + link
                title_tag = result.find('a', class_='result_title')
                if not title_tag:
                    title_tag = result.find('a')
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                href  = title_tag.get('href', '')
                url_  = IK_BASE + href if href.startswith('/') else href

                # Snippet
                snippet_tag = result.find('p')
                snippet     = snippet_tag.get_text(strip=True) if snippet_tag else ''
                snippet     = snippet[:300]

                # Court and year from metadata line
                court = "Unknown"
                year  = "Unknown"
                meta_tag = result.find('div', class_='docsource_main')
                if meta_tag:
                    meta_text = meta_tag.get_text(strip=True)
                    # Try to extract year
                    year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', meta_text)
                    if year_match:
                        year = year_match.group(1)
                    # Court hints
                    if 'Supreme Court' in meta_text:
                        court = 'Supreme Court of India'
                    elif 'High Court' in meta_text:
                        # Try to get which HC
                        hc_match = re.search(r'(\w+)\s+High Court', meta_text)
                        court = f"{hc_match.group(1)} High Court" if hc_match else 'High Court'
                    elif 'NCLAT' in meta_text:
                        court = 'NCLAT'
                    elif 'NCLT' in meta_text:
                        court = 'NCLT'

                results.append({
                    "title":   title,
                    "court":   court,
                    "year":    year,
                    "snippet": snippet,
                    "url":     url_,
                    "source":  "Indian Kanoon",
                    "binding": "Binding" if court == "Supreme Court of India" else "Persuasive"
                })

            except Exception:
                continue

        return results

    except Exception:
        return []


@router.post("/api/search/live")
def live_precedent_search(req: LiveSearchRequest):
    """
    Search Indian Kanoon for the latest judgments.
    Used to supplement local case library with
    up-to-date precedents from the internet.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = scrape_indian_kanoon(req.query.strip(), req.max_results)

    return {
        "query":   req.query,
        "source":  "Indian Kanoon",
        "results": results,
        "count":   len(results),
        "note":    "Results from indiankanoon.org — verify before citing in court."
    }