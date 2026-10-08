"""Place lookups via OpenStreetMap Nominatim, with an embeddable map URL.

Nominatim's usage policy allows roughly one request per second, so every call goes through a
process-wide throttle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

#: Half-width of the embedded map's bounding box, in degrees.
_MAP_BBOX_DEG = 0.06
#: Minimum gap between Nominatim calls (policy is 1 req/s; a little margin added).
_MIN_INTERVAL_SEC = 1.1

_throttle_lock = asyncio.Lock()
_last_call = 0.0

# --- Which OSM results count as a real "place" worth pinning on a map ------------------------
# Search often fuzzy-matches a shop or office with the same name; these lists keep only
# geographic, administrative and well-known civic features.
_LOCATION_CLASSES = frozenset(
    {"place", "boundary", "natural", "waterway", "aeroway", "historic", "mountain_pass", "geological"}
)
_TOURISM_EXCLUDE = frozenset(  # lodging POIs are rarely the intended sense of a mention
    {"hotel", "motel", "guest_house", "hostel", "chalet", "camp_site", "caravan_site", "apartment"}
)
_MAN_MADE_TYPES = frozenset({"bridge", "pier", "lighthouse", "tower"})
_RAILWAY_TYPES = frozenset({"station", "halt", "tram_stop", "subway_entrance"})
_AMENITY_TYPES = frozenset(
    {"townhall", "embassy", "university", "college", "place_of_worship", "library", "courthouse"}
)
_LEISURE_TYPES = frozenset({"park", "nature_reserve"})
_LANDUSE_TYPES = frozenset({"forest", "reservoir", "cemetery"})


async def _throttle() -> None:
    """Wait until at least `_MIN_INTERVAL_SEC` has passed since the previous Nominatim call."""
    global _last_call
    async with _throttle_lock:
        wait = _MIN_INTERVAL_SEC - (time.monotonic() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()


def is_direct_location(row: dict[str, Any]) -> bool:
    """True if a Nominatim result is a geographic/civic feature rather than a business."""
    cls = str(row.get("class") or "").strip().lower()
    typ = str(row.get("type") or "").strip().lower()
    if not cls:
        return False
    if cls in _LOCATION_CLASSES:
        return True
    rules = {
        "tourism": bool(typ) and typ not in _TOURISM_EXCLUDE,
        "man_made": typ in _MAN_MADE_TYPES,
        "railway": typ in _RAILWAY_TYPES,
        "amenity": typ in _AMENITY_TYPES,
        "leisure": typ in _LEISURE_TYPES,
        "landuse": typ in _LANDUSE_TYPES,
    }
    return rules.get(cls, False)


def _map_urls(lat: float, lon: float) -> tuple[str, str]:
    """Return `(embed_url, openstreetmap_url)` centred on a coordinate."""
    b = _MAP_BBOX_DEG
    embed = (
        "https://www.openstreetmap.org/export/embed.html?"
        f"bbox={lon - b},{lat - b},{lon + b},{lat + b}&layer=mapnik&marker={lat},{lon}"
    )
    osm = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=14/{lat}/{lon}"
    return embed, osm


async def lookup_place(client: httpx.AsyncClient, query: str) -> dict[str, Any] | None:
    """Geocode a place name.

    Returns:
        `{lat, lon, display_name, map_embed_url, openstreetmap_url}`, or None when nothing
        suitable is found (or the request fails).
    """
    q = query.strip()
    if not q:
        return None
    await _throttle()
    try:
        response = await client.get(
            NOMINATIM_SEARCH,
            params={"q": q, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": settings.nominatim_user_agent, "Accept-Language": "en"},
            timeout=20.0,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        if not isinstance(row, dict) or not is_direct_location(row):
            return None
        if row.get("lat") is None or row.get("lon") is None:
            return None
        lat, lon = float(row["lat"]), float(row["lon"])
        embed, osm = _map_urls(lat, lon)
        return {
            "lat": lat,
            "lon": lon,
            "display_name": row.get("display_name") or q,
            "map_embed_url": embed,
            "openstreetmap_url": osm,
        }
    except Exception as e:
        logger.debug("nominatim failed for %r: %s", query, e)
        return None
