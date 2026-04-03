"""Fetch Wikipedia summaries, Nominatim coordinates, and Unsplash photos for entities."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Wikipedia's first search hit for one-word brand names often lands on the wrong sense (e.g. fruit vs Apple Inc.).
_WIKI_SEARCH_QUERY_BIAS: dict[str, str] = {
    "apple": "Apple Inc.",
    "amazon": "Amazon (company)",
    "meta": "Meta Platforms",
    "alphabet": "Alphabet Inc.",
    "oracle": "Oracle Corporation",
}


def _wiki_search_query_for_entity(text: str, typ: str) -> str:
    low = text.strip().lower()
    if typ in ("COMPANY", "TECHNOLOGY") and low in _WIKI_SEARCH_QUERY_BIAS:
        return _WIKI_SEARCH_QUERY_BIAS[low]
    return text

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UNSPLASH_SEARCH = "https://api.unsplash.com/search/photos"

USER_AGENT = os.environ.get(
    "NOMINATIM_USER_AGENT",
    "PodcastTranscriptEnrich/1.0 (https://github.com/local; educational use)",
)

_nominatim_lock = asyncio.Lock()
_nominatim_last = 0.0


async def _nominatim_throttle() -> None:
    """Nominatim usage policy: max ~1 request per second."""
    global _nominatim_last
    async with _nominatim_lock:
        now = time.monotonic()
        wait = 1.1 - (now - _nominatim_last)
        if wait > 0:
            await asyncio.sleep(wait)
        _nominatim_last = time.monotonic()


def _dedupe_entities(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for e in raw:
        typ = str(e.get("type", "")).strip().upper()
        text = str(e.get("text", "")).strip()
        if not text:
            continue
        key = (typ, text.lower())
        if key not in seen:
            seen[key] = {
                "type": typ,
                "text": text,
                "start_sec": float(e.get("start_sec", 0)),
                "end_sec": float(e.get("end_sec", 0)),
                "chunk_id": int(e.get("chunk_id", 0)),
            }
        else:
            cur = seen[key]
            cur["start_sec"] = min(cur["start_sec"], float(e.get("start_sec", 0)))
            cur["end_sec"] = max(cur["end_sec"], float(e.get("end_sec", 0)))
    return list(seen.values())


async def _wiki_search_title(client: httpx.AsyncClient, query: str) -> str | None:
    q = query.strip()
    if not q:
        return None
    try:
        r = await client.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": q,
                "srlimit": 1,
                "format": "json",
            },
            timeout=20.0,
        )
        r.raise_for_status()
        data = r.json()
        hits = data.get("query", {}).get("search") or []
        if not hits:
            return None
        return hits[0].get("title")
    except Exception as e:
        logger.debug("wikipedia search failed for %r: %s", query, e)
        return None


async def _wiki_summary(client: httpx.AsyncClient, title: str) -> dict[str, Any] | None:
    if not title:
        return None
    path_title = quote(title.replace(" ", "_"), safe="")
    try:
        r = await client.get(f"{WIKI_REST}/{path_title}", timeout=20.0)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if data.get("type") in ("disambiguation", "https://en.wikipedia.org/wiki/Help:Disambiguation"):
            return None
        thumb = (data.get("thumbnail") or {}) if isinstance(data.get("thumbnail"), dict) else {}
        src = thumb.get("source")
        return {
            "title": data.get("title") or title,
            "extract": (data.get("extract") or "")[:1200],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
            or f"https://en.wikipedia.org/wiki/{path_title}",
            "thumbnail": src,
        }
    except Exception as e:
        logger.debug("wikipedia summary failed for %r: %s", title, e)
        return None


# OSM classes that represent real-world places worth showing on a map.
_NOMINATIM_LOCATION_CLASSES = frozenset(
    {
        "place",
        "boundary",
        "natural",
        "waterway",
        "aeroway",
        "historic",
        "mountain_pass",
        "geological",
    }
)
# Lodging POIs: usually not the geographic sense of a transcript mention.
_NOMINATIM_TOURISM_EXCLUDE = frozenset(
    {
        "hotel",
        "motel",
        "guest_house",
        "hostel",
        "chalet",
        "camp_site",
        "caravan_site",
        "apartment",
    }
)
_NOMINATIM_MAN_MADE_LOCATION_TYPES = frozenset({"bridge", "pier", "lighthouse", "tower"})
_NOMINATIM_RAILWAY_STOP_TYPES = frozenset({"station", "halt", "tram_stop", "subway_entrance"})
# Campuses / civic sites Nominatim often tags as amenities (not shops).
_NOMINATIM_AMENITY_LOCATION_TYPES = frozenset(
    {
        "townhall",
        "embassy",
        "university",
        "college",
        "place_of_worship",
        "library",
        "courthouse",
    }
)
_NOMINATIM_LEISURE_LOCATION_TYPES = frozenset({"park", "nature_reserve"})
_NOMINATIM_LANDUSE_LOCATION_TYPES = frozenset({"forest", "reservoir", "cemetery"})


def _nominatim_result_is_direct_location(row: dict[str, Any]) -> bool:
    """
    Keep map pins only for geographic / administrative features, not e.g. shops
    or offices that happen to fuzzy-match a tagged span.
    """
    cls = str(row.get("class") or "").strip().lower()
    typ = str(row.get("type") or "").strip().lower()
    if not cls:
        return False
    if cls in _NOMINATIM_LOCATION_CLASSES:
        return True
    if cls == "tourism" and typ and typ not in _NOMINATIM_TOURISM_EXCLUDE:
        return True
    if cls == "man_made" and typ in _NOMINATIM_MAN_MADE_LOCATION_TYPES:
        return True
    if cls == "railway" and typ in _NOMINATIM_RAILWAY_STOP_TYPES:
        return True
    if cls == "amenity" and typ in _NOMINATIM_AMENITY_LOCATION_TYPES:
        return True
    if cls == "leisure" and typ in _NOMINATIM_LEISURE_LOCATION_TYPES:
        return True
    if cls == "landuse" and typ in _NOMINATIM_LANDUSE_LOCATION_TYPES:
        return True
    return False


async def _nominatim_lookup(client: httpx.AsyncClient, query: str) -> dict[str, Any] | None:
    q = query.strip()
    if not q:
        return None
    await _nominatim_throttle()
    try:
        r = await client.get(
            NOMINATIM,
            params={"q": q, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
            timeout=20.0,
        )
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            return None
        row = rows[0]
        if not isinstance(row, dict) or not _nominatim_result_is_direct_location(row):
            return None
        lat = row.get("lat")
        lon = row.get("lon")
        if lat is None or lon is None:
            return None
        lat_f = float(lat)
        lon_f = float(lon)
        name = row.get("display_name") or q
        bbox = 0.06
        embed = (
            f"https://www.openstreetmap.org/export/embed.html?"
            f"bbox={lon_f - bbox},{lat_f - bbox},{lon_f + bbox},{lat_f + bbox}"
            f"&layer=mapnik&marker={lat_f},{lon_f}"
        )
        osm = f"https://www.openstreetmap.org/?mlat={lat_f}&mlon={lon_f}#map=14/{lat_f}/{lon_f}"
        return {
            "lat": lat_f,
            "lon": lon_f,
            "display_name": name,
            "map_embed_url": embed,
            "openstreetmap_url": osm,
        }
    except Exception as e:
        logger.debug("nominatim failed for %r: %s", query, e)
        return None


async def _unsplash_photo(
    client: httpx.AsyncClient, query: str, access_key: str
) -> dict[str, Any] | None:
    """One image from Unsplash search (requires Client ID access key)."""
    q = query.strip()
    if not q or not access_key:
        return None
    try:
        r = await client.get(
            UNSPLASH_SEARCH,
            params={"query": q, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=20.0,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results or not isinstance(results, list):
            return None
        ph = results[0]
        if not isinstance(ph, dict):
            return None
        urls = ph.get("urls") if isinstance(ph.get("urls"), dict) else {}
        user = ph.get("user") if isinstance(ph.get("user"), dict) else {}
        links = ph.get("links") if isinstance(ph.get("links"), dict) else {}
        user_links = user.get("links") if isinstance(user.get("links"), dict) else {}
        image_url = urls.get("regular") or urls.get("small") or urls.get("full")
        thumb_url = urls.get("small") or urls.get("thumb") or image_url
        if not image_url:
            return None
        alt = ph.get("alt_description") or ph.get("description") or ""
        return {
            "image_url": image_url,
            "thumb_url": thumb_url,
            "alt": str(alt)[:400] if alt else "",
            "photographer_name": str(user.get("name") or "").strip(),
            "photographer_url": str(user_links.get("html") or "").strip(),
            "unsplash_url": str(links.get("html") or "").strip(),
        }
    except Exception as e:
        logger.debug("unsplash failed for %r: %s", query, e)
        return None


async def enrich_entity_row(
    client: httpx.AsyncClient,
    entity: dict[str, Any],
    *,
    unsplash_key: str | None,
) -> dict[str, Any]:
    text = entity["text"]
    typ = entity["type"]
    card_id = hashlib.sha256(f"{typ}\n{text}".encode("utf-8")).hexdigest()[:16]
    card: dict[str, Any] = {
        "id": card_id,
        "type": typ,
        "text": text,
        "start_sec": entity["start_sec"],
        "end_sec": entity["end_sec"],
        "chunk_id": entity["chunk_id"],
        "wikipedia": None,
        "location": None,
        "unsplash": None,
    }

    wiki_task = asyncio.create_task(
        _wiki_search_title(client, _wiki_search_query_for_entity(text, typ))
    )
    nom_place_task = asyncio.create_task(_nominatim_lookup(client, text)) if typ == "PLACE" else None

    title = await wiki_task
    if title:
        card["wikipedia"] = await _wiki_summary(client, title)

    if nom_place_task:
        card["location"] = await nom_place_task

    if typ != "PLACE":
        card["location"] = None

    if unsplash_key:
        photo_query = text
        wiki = card.get("wikipedia")
        if isinstance(wiki, dict) and wiki.get("title"):
            photo_query = str(wiki["title"])
        elif typ == "PLACE" and isinstance(card.get("location"), dict):
            loc = card["location"]
            if isinstance(loc, dict) and loc.get("display_name"):
                photo_query = str(loc["display_name"])[:200]
        card["unsplash"] = await _unsplash_photo(client, photo_query, unsplash_key)

    return card


async def enrich_entities_payload(entities: list[dict[str, Any]]) -> dict[str, Any]:
    unique = _dedupe_entities(entities)
    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip() or None

    headers = {"User-Agent": USER_AGENT}
    cards: list[dict[str, Any]] = []
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        # Sequential: Nominatim allows ~1 req/s; interleaved per-entity parallel calls break that.
        for e in unique:
            cards.append(await enrich_entity_row(client, e, unsplash_key=unsplash_key))

    return {
        "cards": cards,
        "count": len(cards),
        "unsplash_enabled": bool(unsplash_key),
    }
