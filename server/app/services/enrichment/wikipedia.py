"""Wikipedia lookups: search for the best article title, then fetch its REST summary."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"

#: Max characters of article summary kept on a card.
MAX_EXTRACT_CHARS = 1200

# Wikipedia's first search hit for one-word brand names often lands on the wrong sense
# (e.g. the fruit instead of Apple Inc.), so company/tech mentions are searched by full name.
_SEARCH_QUERY_BIAS: dict[str, str] = {
    "apple": "Apple Inc.",
    "amazon": "Amazon (company)",
    "meta": "Meta Platforms",
    "alphabet": "Alphabet Inc.",
    "oracle": "Oracle Corporation",
}


def search_query_for_entity(text: str, entity_type: str) -> str:
    """Return the Wikipedia search query for an entity, applying brand-name bias where needed."""
    low = text.strip().lower()
    if entity_type in ("COMPANY", "TECHNOLOGY") and low in _SEARCH_QUERY_BIAS:
        return _SEARCH_QUERY_BIAS[low]
    return text


async def search_title(client: httpx.AsyncClient, query: str) -> str | None:
    """Return the title of the top Wikipedia search hit for `query`, or None."""
    q = query.strip()
    if not q:
        return None
    try:
        response = await client.get(
            WIKI_API,
            params={"action": "query", "list": "search", "srsearch": q, "srlimit": 1, "format": "json"},
            timeout=20.0,
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search") or []
        return hits[0].get("title") if hits else None
    except Exception as e:  # network/HTTP/JSON errors: a missing card section is fine
        logger.debug("wikipedia search failed for %r: %s", query, e)
        return None


async def fetch_summary(client: httpx.AsyncClient, title: str) -> dict[str, Any] | None:
    """Fetch `{title, extract, url, thumbnail}` for an article; None for 404 or disambiguation pages."""
    if not title:
        return None
    path_title = quote(title.replace(" ", "_"), safe="")
    try:
        response = await client.get(f"{WIKI_REST_SUMMARY}/{path_title}", timeout=20.0)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        if data.get("type") in ("disambiguation", "https://en.wikipedia.org/wiki/Help:Disambiguation"):
            return None
        thumbnail = data.get("thumbnail") if isinstance(data.get("thumbnail"), dict) else {}
        return {
            "title": data.get("title") or title,
            "extract": (data.get("extract") or "")[:MAX_EXTRACT_CHARS],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
            or f"https://en.wikipedia.org/wiki/{path_title}",
            "thumbnail": thumbnail.get("source"),
        }
    except Exception as e:
        logger.debug("wikipedia summary failed for %r: %s", title, e)
        return None
