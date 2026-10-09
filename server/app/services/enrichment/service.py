"""Build "source cards" for entities: Wikipedia summary, map location and a photo."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import httpx

from app.core.config import settings
from app.services.enrichment import geocoding, unsplash, wikipedia


def dedupe_entities(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated mentions to one row per `(type, lower-cased text)`.

    The kept row spans from the earliest start to the latest end of all its mentions, and keeps
    the first non-empty `search_query` hint.
    """
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
                "search_query": str(e.get("search_query") or "").strip() or None,
            }
        else:
            current = seen[key]
            current["start_sec"] = min(current["start_sec"], float(e.get("start_sec", 0)))
            current["end_sec"] = max(current["end_sec"], float(e.get("end_sec", 0)))
            current["search_query"] = current["search_query"] or (str(e.get("search_query") or "").strip() or None)
    return list(seen.values())


def card_id(entity_type: str, text: str) -> str:
    """Stable 16-char id for a card, derived from its type and text."""
    return hashlib.sha256(f"{entity_type}\n{text}".encode("utf-8")).hexdigest()[:16]


async def enrich_entity(
    client: httpx.AsyncClient, entity: dict[str, Any], *, unsplash_key: str | None
) -> dict[str, Any]:
    """Build one source card.

    - Wikipedia search runs for every entity; the geocode runs in parallel for PLACE only.
    - `entity["search_query"]` (set by the LLM entity review) replaces the default Wikipedia query,
      so ambiguous names land on the intended article ("Yerba Buena" the town, not the plant).
    - The photo query prefers the Wikipedia title (more specific), then the review's search query,
      then the place name.
    """
    text, typ = entity["text"], entity["type"]
    card: dict[str, Any] = {
        "id": card_id(typ, text),
        "type": typ,
        "text": text,
        "start_sec": entity["start_sec"],
        "end_sec": entity["end_sec"],
        "chunk_id": entity["chunk_id"],
        "wikipedia": None,
        "location": None,
        "unsplash": None,
    }

    hint = str(entity.get("search_query") or "").strip()
    wiki_query = hint or wikipedia.search_query_for_entity(text, typ)
    wiki_task = asyncio.create_task(wikipedia.search_title(client, wiki_query))
    place_task = asyncio.create_task(geocoding.lookup_place(client, text)) if typ == "PLACE" else None

    title = await wiki_task
    if title:
        card["wikipedia"] = await wikipedia.fetch_summary(client, title)
    if place_task:
        card["location"] = await place_task

    if unsplash_key:
        photo_query = text
        wiki = card.get("wikipedia")
        location = card.get("location")
        if isinstance(wiki, dict) and wiki.get("title"):
            photo_query = str(wiki["title"])
        elif hint:
            photo_query = hint
        elif typ == "PLACE" and isinstance(location, dict) and location.get("display_name"):
            photo_query = str(location["display_name"])[:200]
        card["unsplash"] = await unsplash.search_photo(client, photo_query, unsplash_key)

    return card


async def enrich_entities_payload(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Enrich every unique entity and return `{cards, count, unsplash_enabled}`.

    Entities are processed one after another on purpose: Nominatim allows ~1 request/second,
    and running entities in parallel would break that limit.
    """
    unique = dedupe_entities(entities)
    unsplash_key = settings.unsplash_access_key or None
    cards: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        headers={"User-Agent": settings.nominatim_user_agent}, follow_redirects=True
    ) as client:
        for entity in unique:
            cards.append(await enrich_entity(client, entity, unsplash_key=unsplash_key))
    return {"cards": cards, "count": len(cards), "unsplash_enabled": bool(unsplash_key)}
