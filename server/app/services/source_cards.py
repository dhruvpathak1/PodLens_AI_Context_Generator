"""Source cards with optional LLM review: review entities -> look them up -> check the cards.

Used by `POST /api/enrich-entities` and the demo builder, so both produce the same result.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services import entity_review
from app.services.enrichment import enrich_entities_payload

logger = logging.getLogger(__name__)


async def build_source_cards(
    entities: list[dict[str, Any]],
    *,
    segments: list[dict[str, Any]] | None = None,
    source_label: str | None = None,
    review: bool | None = None,
    review_chain: Any = None,
    verify_chain: Any = None,
) -> dict[str, Any]:
    """Build source cards, reviewing the entities with an LLM first when enabled.

    Args:
        entities: Entity mentions from NER.
        segments: Transcript segments; required for the review (it needs context lines).
        review: True/False to force; None = on when `ENTITY_REVIEW` is set and OpenAI is configured.
        review_chain / verify_chain: injected LangChain runnables (tests).

    Returns:
        `{cards, count, unsplash_enabled}` plus, when the review ran, `entities` (the reviewed
        mentions the client should display) and `review` (what was fixed, merged and dropped).
        If the review fails, cards are built from the original entities and `review.error` is set.
    """
    run_review = (entity_review.review_enabled() if review is None else review) and bool(segments)
    if not run_review:
        return await enrich_entities_payload(entities)

    try:
        reviewed, report = await entity_review.review_entities(
            entities, segments or [], source_label, chain=review_chain
        )
    except Exception as e:  # fail open: never lose the cards because the reviewer broke
        logger.warning("entity review failed; using unreviewed entities: %s", e)
        result = await enrich_entities_payload(entities)
        return {**result, "review": {"error": str(e)}}

    result = await enrich_entities_payload(reviewed) if reviewed else {"cards": [], "count": 0, "unsplash_enabled": False}
    try:
        cards, removed = await entity_review.verify_cards(
            result["cards"], reviewed, segments or [], source_label, chain=verify_chain
        )
    except Exception as e:
        logger.warning("card check failed; keeping cards as built: %s", e)
        cards, removed = result["cards"], 0
    report["mismatches_removed"] = removed
    return {**result, "cards": cards, "count": len(cards), "entities": reviewed, "review": report}
