"""Stock photos from the Unsplash search API (optional; needs `UNSPLASH_ACCESS_KEY`)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

UNSPLASH_SEARCH = "https://api.unsplash.com/search/photos"


def _as_dict(value: Any) -> dict[str, Any]:
    """Return `value` if it is a dict, else an empty dict (Unsplash fields are sometimes null)."""
    return value if isinstance(value, dict) else {}


async def search_photo(client: httpx.AsyncClient, query: str, access_key: str) -> dict[str, Any] | None:
    """Return the top landscape photo for `query` with attribution fields, or None.

    Result keys: `image_url, thumb_url, alt, photographer_name, photographer_url, unsplash_url`.
    Unsplash's API terms require showing the photographer credit, which the card does.
    """
    q = query.strip()
    if not q or not access_key:
        return None
    try:
        response = await client.get(
            UNSPLASH_SEARCH,
            params={"query": q, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=20.0,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            return None
        photo = results[0]
        urls, user, links = _as_dict(photo.get("urls")), _as_dict(photo.get("user")), _as_dict(photo.get("links"))
        image_url = urls.get("regular") or urls.get("small") or urls.get("full")
        if not image_url:
            return None
        alt = photo.get("alt_description") or photo.get("description") or ""
        return {
            "image_url": image_url,
            "thumb_url": urls.get("small") or urls.get("thumb") or image_url,
            "alt": str(alt)[:400] if alt else "",
            "photographer_name": str(user.get("name") or "").strip(),
            "photographer_url": str(_as_dict(user.get("links")).get("html") or "").strip(),
            "unsplash_url": str(links.get("html") or "").strip(),
        }
    except Exception as e:
        logger.debug("unsplash failed for %r: %s", query, e)
        return None
