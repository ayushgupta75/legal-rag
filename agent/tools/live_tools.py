"""
Live legal API tools called by the LangGraph agent when vector DB is insufficient.
All functions return a list of result dicts with the same schema as vector chunks.
"""
import httpx
import logging
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4"
CONGRESS_BASE = "https://api.congress.gov/v3"
ECFR_BASE = "https://www.ecfr.gov/api/versioner/v1"


def _safe_get(url: str, params: dict, headers: dict = {}) -> dict:
    try:
        r = httpx.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"API call failed: {url} — {e}")
        return {}


def search_courtlistener(query: str, max_results: int = 4) -> list[dict]:
    """Search CourtListener for court opinions."""
    headers = {}
    if settings.courtlistener_api_key:
        headers["Authorization"] = f"Token {settings.courtlistener_api_key}"

    data = _safe_get(
        f"{COURTLISTENER_BASE}/search/",
        params={"q": query, "type": "o", "order_by": "score desc"},
        headers=headers,
    )
    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "id": f"cl_{r.get('id', '')}",
            "source": "caselaw",
            "citation": r["citation"][0] if isinstance(r.get("citation"), list) and r["citation"] else r.get("caseName", "Unknown"),
            "title": r.get("caseName", ""),
            "section": r.get("court", ""),
            "text": r.get("snippet", r.get("text", ""))[:2000],
            "url": f"https://www.courtlistener.com{r.get('absolute_url', '')}",
        })
    return results


def search_congress(query: str, max_results: int = 4) -> list[dict]:
    """Search Congress.gov for bills and statutes."""
    data = _safe_get(
        f"{CONGRESS_BASE}/bill",
        params={
            "query": query,
            "limit": max_results,
            "api_key": settings.congress_api_key or "DEMO_KEY",
        },
    )
    results = []
    for bill in data.get("bills", [])[:max_results]:
        results.append({
            "id": f"congress_{bill.get('number', '')}",
            "source": "legislation",
            "citation": f"{bill.get('type', 'Bill')} {bill.get('number', '')}, {bill.get('congress', '')}th Congress",
            "title": bill.get("title", ""),
            "section": bill.get("originChamber", ""),
            "text": bill.get("title", "") + ". " + bill.get("latestAction", {}).get("text", ""),
            "url": bill.get("url", ""),
        })
    return results


def search_ecfr(query: str, max_results: int = 4) -> list[dict]:
    """Search eCFR (Electronic Code of Federal Regulations) full text."""
    data = _safe_get(
        f"{ECFR_BASE}/search/",
        params={"query": query, "per_page": max_results},
    )
    results = []
    for r in data.get("results", {}).get("hits", {}).get("hits", [])[:max_results]:
        src = r.get("_source", {})
        results.append({
            "id": f"ecfr_{r.get('_id', '')}",
            "source": "cfr",
            "citation": f"{src.get('cfr_references', [{}])[0].get('cfr_reference', 'CFR')}",
            "title": src.get("hierarchy_headings", {}).get("title", ""),
            "section": src.get("hierarchy_headings", {}).get("section", ""),
            "text": src.get("full_text_excerpt", "")[:2000],
            "url": "",
        })
    return results
