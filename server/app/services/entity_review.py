"""LLM review of extracted entities, in the context of the episode (LangChain + OpenAI).

Speech recognition and NER make predictable mistakes on podcasts: misheard names
("Traderus 8" for the Traitorous Eight), fragments ("San"), show names and filler tagged as
entities, the same place under three names ("SF", "San Francisco", "San Francisco, California"),
and ambiguous names that Wikipedia resolves to the wrong sense ("Yerba Buena" the plant).

Two passes, both with strict structured output:

1. `review_entities` (before lookups): the model sees each unique entity with the transcript
   lines it appears in and returns a verdict: keep, fix (corrected name/type), merge into another
   entity, or drop. It also writes a disambiguated Wikipedia `search_query` for each kept entity.
2. `verify_cards` (after lookups): the model checks each card's Wikipedia summary and photo
   description against how the episode uses the name, and mismatches are removed from the card.

Both passes fail open: if the model errors, the original entities/cards are used unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

REVIEW_MODEL = settings.review_model
#: Unique entities per review request (keeps prompts small and lets requests run in parallel).
REVIEW_BATCH = 60
REVIEW_MAX_CONCURRENCY = 4
#: Context lines per entity, and characters per line.
CONTEXT_LINES = 2
CONTEXT_CHARS = 220
#: Opening of the transcript shown to the model so it knows what the episode is about.
TOPIC_CHARS = 1500

EntityType = Literal["PLACE", "PERSON", "TECHNOLOGY", "EVENT", "COMPANY", "MISC"]


# ---------- Structured output schemas ----------
# Every field is required (nullable where optional) so the schemas are valid in OpenAI strict mode.


class EntityVerdict(BaseModel):
    """The model's decision for one unique entity."""

    id: str = Field(description="The entity id from the list, e.g. 'e3'. Copy it exactly.")
    action: Literal["keep", "fix", "merge", "drop"] = Field(
        description="keep: correct as is. fix: real entity but misspelled, misheard or mistyped. "
        "merge: same real-world thing as another listed entity. drop: not a specific, identifiable entity."
    )
    name: str = Field(description="Canonical display name (corrected spelling for 'fix'; unchanged otherwise).")
    type: EntityType = Field(description="Correct entity type.")
    merge_into: str | None = Field(description="For 'merge': id of the entity to merge into. Otherwise null.")
    search_query: str | None = Field(
        description="Precise Wikipedia search string for the sense used in this episode, "
        "e.g. 'Yerba Buena San Francisco history'. Null for 'drop'."
    )
    reason: str = Field(description="At most 12 words explaining the decision.")


class EntityReview(BaseModel):
    """Verdicts for one batch of entities."""

    verdicts: list[EntityVerdict]


class CardCheck(BaseModel):
    """Whether a card's looked-up content matches what the episode means."""

    id: str = Field(description="The card id from the list. Copy it exactly.")
    wikipedia_matches: bool | None = Field(
        description="True if the Wikipedia summary describes the thing the episode talks about; "
        "false if it is a different sense or unrelated; null if there is no summary."
    )
    photo_matches: bool | None = Field(
        description="True if the photo description plausibly shows it; false if clearly unrelated; "
        "null if there is no photo or no description."
    )


class CardReview(BaseModel):
    """Checks for one batch of cards."""

    checks: list[CardCheck]


REVIEW_SYSTEM = """You review named entities that speech recognition and an entity tagger extracted
from a podcast transcript. Decide, for every entity, what a careful editor would show as a source card.

- keep: a real, specific person, place, organisation, product, event or named concept, spelled correctly.
- fix: a real entity that was misheard or misspelled, or has the wrong type. Give the corrected
  canonical name, e.g. "Traderus 8" -> "Traitorous Eight", "Apollo Alto" -> "Palo Alto".
- merge: the same real-world thing as another entity in the list (e.g. "SF" and "San Francisco").
  Set merge_into to the id of the most complete name. Never merge different things.
- drop: not worth a card: generic words, fragments ("San"), the show's own name or segment titles,
  filler, mis-tagged common nouns, or anything you cannot identify from the context lines.

Use the context lines to decide what the speakers mean. Never add entities that are not listed.
Types: PERSON, PLACE, COMPANY (any organisation), TECHNOLOGY (products, software, technical
things), EVENT (named events, eras, periods), MISC (other named things).
search_query must pin down the intended sense for a Wikipedia search.
Return exactly one verdict for every id."""

REVIEW_HUMAN = """Episode: {source_label}
What the episode is about (opening of the transcript):
{topic}

Entities (id | type | name | mentions | context lines):
{entities}"""

VERIFY_SYSTEM = """You check source cards built for names mentioned in a podcast. For each card, compare
the Wikipedia summary and the photo description with how the episode uses the name (context lines).
Be strict: a different sense of the same word (a plant instead of a place, a film instead of a person)
is a mismatch. Judge only what is given; return one check for every card id."""

VERIFY_HUMAN = """Episode: {source_label}

Cards (id | name | type | context | Wikipedia title: summary | photo description):
{cards}"""


class ReviewNotConfigured(RuntimeError):
    """Raised when the review is requested but `OPENAI_API_KEY` is missing."""


def review_enabled() -> bool:
    """True when the entity review should run by default (setting on and an OpenAI key present)."""
    return settings.entity_review and settings.openai_configured


# ---------- helpers (pure, unit tested) ----------


def entity_key(entity_type: str, text: str) -> tuple[str, str]:
    """Identity of a unique entity: upper-cased type and lower-cased, trimmed name."""
    return (str(entity_type).strip().upper(), str(text).strip().lower())


def group_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unique entities in order of first mention: `{id, key, type, text, count, chunk_ids, starts}`."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for e in sorted(entities, key=lambda m: float(m.get("start_sec", 0))):
        text = str(e.get("text", "")).strip()
        if not text:
            continue
        key = entity_key(e.get("type", ""), text)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {
                "id": f"e{len(groups)}",
                "key": key,
                "type": key[0],
                "text": text,
                "count": 0,
                "chunk_ids": [],
                "starts": [],
            }
        g["count"] += 1
        g["chunk_ids"].append(e.get("chunk_id"))
        g["starts"].append(float(e.get("start_sec", 0)))
    return list(groups.values())


def context_lines(group: dict[str, Any], segments: list[dict[str, Any]]) -> list[str]:
    """Up to `CONTEXT_LINES` transcript lines where the entity is mentioned (by chunk id, else by time)."""
    by_id = {s.get("id"): s for s in segments}
    lines: list[str] = []
    for chunk_id, start in zip(group["chunk_ids"], group["starts"]):
        seg = by_id.get(chunk_id)
        if seg is None:
            seg = next((s for s in segments if float(s["start"]) <= start < float(s["end"])), None)
        text = str(seg.get("text", "")).strip() if seg else ""
        if text and text not in lines:
            lines.append(text[:CONTEXT_CHARS])
        if len(lines) >= CONTEXT_LINES:
            break
    return lines


def topic_excerpt(segments: list[dict[str, Any]]) -> str:
    """Opening of the transcript (about `TOPIC_CHARS` characters)."""
    text = " ".join(str(s.get("text", "")).strip() for s in segments).strip()
    return text[:TOPIC_CHARS] + ("…" if len(text) > TOPIC_CHARS else "")


def render_entities(groups: list[dict[str, Any]], segments: list[dict[str, Any]]) -> str:
    """Prompt block: one entity per line with its context lines."""
    rows = []
    for g in groups:
        ctx = " / ".join(context_lines(g, segments)) or "(no context)"
        rows.append(f"{g['id']} | {g['type']} | {g['text']} | {g['count']} | {ctx}")
    return "\n".join(rows)


def apply_verdicts(
    entities: list[dict[str, Any]], groups: list[dict[str, Any]], verdicts: list[EntityVerdict]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rewrite entity mentions according to the verdicts.

    - keep/fix: mention gets the canonical name and type, plus the `search_query` hint.
    - merge: mention takes the target entity's final name/type (one level; chains resolve to the
      target's own verdict). A missing or self-referencing target is treated as keep.
    - drop: mention removed.
    - no verdict for an id: kept unchanged (fail open).

    Returns:
        `(reviewed_mentions, report)` where report has `fixed`, `merged` and `dropped` lists.
    """
    by_id = {g["id"]: g for g in groups}
    verdict_by_id = {v.id: v for v in verdicts if v.id in by_id}

    def final(gid: str) -> tuple[str, str, str | None] | None:
        """(name, type, search_query) an entity ends up as, or None when dropped."""
        g = by_id[gid]
        v = verdict_by_id.get(gid)
        if v is None:
            return g["text"], g["type"], None
        if v.action == "drop":
            return None
        if v.action == "merge" and v.merge_into and v.merge_into != gid and v.merge_into in by_id:
            target = verdict_by_id.get(v.merge_into)
            if target is None or target.action not in ("drop", "merge"):
                return final(v.merge_into)
        return (v.name.strip() or g["text"]), v.type, v.search_query

    outcome = {gid: final(gid) for gid in by_id}
    report: dict[str, Any] = {"fixed": [], "merged": [], "dropped": []}
    for gid, g in by_id.items():
        v, out = verdict_by_id.get(gid), outcome[gid]
        if out is None:
            report["dropped"].append({"name": g["text"], "reason": v.reason if v else ""})
        elif v and v.action == "merge" and entity_key(out[1], out[0]) != g["key"]:
            report["merged"].append({"name": g["text"], "into": out[0]})
        elif entity_key(out[1], out[0]) != g["key"]:
            report["fixed"].append({"name": g["text"], "to": out[0], "type": out[1]})

    gid_by_key = {g["key"]: g["id"] for g in groups}
    reviewed: list[dict[str, Any]] = []
    for e in entities:
        gid = gid_by_key.get(entity_key(e.get("type", ""), e.get("text", "")))
        out = outcome.get(gid) if gid else (str(e.get("text", "")), str(e.get("type", "")), None)
        if out is None:
            continue
        name, typ, query = out
        reviewed.append({**e, "text": name, "type": typ, **({"search_query": query} if query else {})})
    report["kept"] = len({entity_key(m["type"], m["text"]) for m in reviewed})
    return reviewed, report


def render_cards(cards: list[dict[str, Any]], contexts: dict[tuple[str, str], str]) -> str:
    """Prompt block for the card check: one card per line."""
    rows = []
    for c in cards:
        wiki = c.get("wikipedia") or {}
        photo = c.get("unsplash") or {}
        summary = f"{wiki.get('title', '')}: {str(wiki.get('extract', ''))[:300]}" if wiki else "(none)"
        alt = str(photo.get("alt") or "").strip() or ("(no description)" if photo else "(none)")
        ctx = contexts.get(entity_key(c["type"], c["text"]), "")
        rows.append(f"{c['id']} | {c['text']} | {c['type']} | {ctx} | {summary} | {alt}")
    return "\n".join(rows)


def apply_card_checks(cards: list[dict[str, Any]], checks: list[CardCheck]) -> tuple[list[dict[str, Any]], int]:
    """Remove the Wikipedia section and/or photo from cards the model flagged; returns (cards, removed_count)."""
    by_id = {c.id: c for c in checks}
    removed = 0
    out = []
    for card in cards:
        check = by_id.get(card["id"])
        card = dict(card)
        if check and check.wikipedia_matches is False and card.get("wikipedia"):
            card["wikipedia"] = None
            removed += 1
        if check and check.photo_matches is False and card.get("unsplash"):
            card["unsplash"] = None
            removed += 1
        out.append(card)
    return out, removed


# ---------- LLM passes ----------


def _build_chain(schema: type[BaseModel], system: str, human: str):
    """`prompt | ChatOpenAI.with_structured_output(schema)` (strict JSON schema)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=REVIEW_MODEL, max_retries=2, timeout=120)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    return prompt | llm.with_structured_output(schema, method="json_schema", strict=True)


async def review_entities(
    entities: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    source_label: str | None = None,
    chain: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pass 1: keep / fix / merge / drop each unique entity, with a Wikipedia search hint.

    Batches that fail are left unchanged (their entities are kept as extracted).

    Returns:
        `(reviewed_mentions, report)`.
    """
    groups = group_entities(entities)
    if not groups:
        return [], {"fixed": [], "merged": [], "dropped": [], "kept": 0, "model": REVIEW_MODEL}
    if chain is None:
        if not settings.openai_configured:
            raise ReviewNotConfigured("OPENAI_API_KEY is not set on the API server.")
        chain = _build_chain(EntityReview, REVIEW_SYSTEM, REVIEW_HUMAN)

    label = (source_label or "podcast episode").strip()[:200]
    topic = topic_excerpt(segments)
    batches = [groups[i : i + REVIEW_BATCH] for i in range(0, len(groups), REVIEW_BATCH)]
    inputs = [{"source_label": label, "topic": topic, "entities": render_entities(b, segments)} for b in batches]
    results = await chain.abatch(inputs, config={"max_concurrency": REVIEW_MAX_CONCURRENCY}, return_exceptions=True)

    verdicts: list[EntityVerdict] = []
    failed = 0
    for i, r in enumerate(results):
        if isinstance(r, EntityReview):
            verdicts.extend(r.verdicts)
        else:
            failed += 1
            logger.warning("entity review batch %s failed: %s", i + 1, r)
    reviewed, report = apply_verdicts(entities, groups, verdicts)
    report.update(model=REVIEW_MODEL, failed_batches=failed)
    return reviewed, report


async def verify_cards(
    cards: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    source_label: str | None = None,
    chain: Any = None,
) -> tuple[list[dict[str, Any]], int]:
    """Pass 2: drop Wikipedia summaries and photos that do not match the episode's meaning.

    Returns:
        `(cards, removed_count)`; on any failure the cards are returned unchanged.
    """
    checkable = [c for c in cards if c.get("wikipedia") or c.get("unsplash")]
    if not checkable:
        return cards, 0
    if chain is None:
        if not settings.openai_configured:
            return cards, 0
        chain = _build_chain(CardReview, VERIFY_SYSTEM, VERIFY_HUMAN)

    contexts = {g["key"]: " / ".join(context_lines(g, segments)) for g in group_entities(entities)}
    label = (source_label or "podcast episode").strip()[:200]
    batches = [checkable[i : i + REVIEW_BATCH] for i in range(0, len(checkable), REVIEW_BATCH)]
    inputs = [{"source_label": label, "cards": render_cards(b, contexts)} for b in batches]
    results = await chain.abatch(inputs, config={"max_concurrency": REVIEW_MAX_CONCURRENCY}, return_exceptions=True)

    checks: list[CardCheck] = []
    for i, r in enumerate(results):
        if isinstance(r, CardReview):
            checks.extend(r.checks)
        else:
            logger.warning("card check batch %s failed: %s", i + 1, r)
    return apply_card_checks(cards, checks)
