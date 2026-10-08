"""NER with Claude (classic CoNLL-2003 labels), best for messy podcast speech.

Chunks are sent in batches (`CLAUDE_ENTITY_BATCH`, default 12) to bound tokens per request.
Claude returns `PERSON | ORG | LOC | MISC`; these are mapped to PodLens types
(ORG -> COMPANY, LOC -> PLACE). Each returned span is located in its chunk text to estimate
when it is spoken.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

from app.core.config import ENV_FILE, settings
from app.services.entities.text_cleaning import clean_entity_text, entity_time_span, filter_noise
from app.services.entities.types import ENTITY_TYPES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You perform classic Named Entity Recognition on podcast transcript chunks.
Use the standard CoNLL-2003 style taxonomy (4 types). Return ONLY valid JSON (no markdown). Schema:
{"entities":[{"type":"PERSON|ORG|LOC|MISC","text":"exact surface span from the chunk","chunk_id":number}]}

Definitions:
- PERSON: people, including titles if part of the name span (e.g. "Dr. Smith"). No generic roles alone ("the host").
- ORG: companies, agencies, institutions, political parties, sports teams when named as organizations.
- LOC: locations: cities, countries, mountains, rivers, named venues/regions, addresses. Include the Amazon River here, not under ORG.
- MISC: miscellaneous named entities that are not PERSON/ORG/LOC: languages, nationalities/ethnic groups, events, laws, works of art,
  religions, named products/technologies when they are not clearly an ORG (e.g. "Python" the language → MISC), wars, holidays.

Rules:
- type must be exactly one of: PERSON, ORG, LOC, MISC
- Copy "text" as the verbatim substring from that chunk (for alignment). One row per distinct span; do not duplicate the same span+type in a chunk.
- Use chunk wording to disambiguate homonyms: "Apple" + iPhone/Mac/Cupertino → ORG; "apple" + pie/orchard/fruit → omit as a company.
  "Amazon" + AWS/Prime/Bezos → ORG; Amazon as the river/rainforest → LOC. "Oracle" software company → ORG; Greek oracle/myth → MISC.
  "Meta" + Facebook/Instagram → ORG.
- Omit filler (um, uh), pronouns, generic words, and non-entity mentions
"""

#: Accepted label spellings -> canonical CoNLL bucket.
_LABEL_ALIASES: dict[str, str] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "FAC": "LOC",
    "MISC": "MISC",
    "MISCELLANEOUS": "MISC",
}

#: CoNLL bucket -> PodLens entity type.
_CONLL_TO_APP: dict[str, str] = {"PERSON": "PERSON", "ORG": "COMPANY", "LOC": "PLACE", "MISC": "MISC"}


def normalize_label(raw: str) -> str | None:
    """Map a Claude label (CoNLL or already a PodLens type) to a PodLens type, or None to drop."""
    label = str(raw or "").strip().upper()
    if not label:
        return None
    if label in ENTITY_TYPES:
        return label
    bucket = _LABEL_ALIASES.get(label)
    return _CONLL_TO_APP.get(bucket) if bucket else None


def _strip_code_fence(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _api_key() -> str:
    """Read `ANTHROPIC_API_KEY`, re-loading `.env` so a key added after startup is picked up.

    Raises:
        RuntimeError: no key configured.
    """
    load_dotenv(ENV_FILE, encoding="utf-8-sig")
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set for Claude entity extraction. "
            f"Set ANTHROPIC_API_KEY in {ENV_FILE} (no quotes) and restart the API server."
        )
    return key


def extract_with_claude(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag entities in `{id, start, end, text}` chunks using Claude, batch by batch."""
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    batch_size = max(1, settings.claude_entity_batch)
    all_entities: list[dict[str, Any]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # 1) Build one prompt with every non-empty chunk in the batch, tagged with its id and times.
        blocks = []
        for chunk in batch:
            cleaned = filter_noise(chunk.get("text") or "")
            if cleaned:
                blocks.append(
                    f"[chunk_id={int(chunk.get('id', 0))} start={chunk['start']:.2f} end={chunk['end']:.2f}]\n{cleaned}"
                )
        if not blocks:
            continue

        # 2) Ask Claude for JSON.
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Chunks:\n\n" + "\n\n---\n\n".join(blocks)}],
        )
        raw_text = _strip_code_fence("".join(b.text for b in message.content if b.type == "text"))
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("Claude JSON parse failed: %s | snippet=%r", e, raw_text[:400])
            continue  # skip this batch, keep the rest
        rows = data.get("entities") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            continue

        # 3) Validate each row and map its text back to a time span inside its chunk.
        by_id = {int(c["id"]): c for c in batch if "id" in c}
        for item in rows:
            entity = _row_to_entity(item, by_id)
            if entity:
                all_entities.append(entity)

    return all_entities


def _row_to_entity(item: Any, chunks_by_id: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    """Turn one Claude JSON row into a PodLens entity dict, or None if it is invalid."""
    if not isinstance(item, dict):
        return None
    raw_label = str(item.get("type", "")).strip().upper()
    typ = normalize_label(raw_label)
    text = clean_entity_text(str(item.get("text", "")))
    if not typ or not text:
        return None
    try:
        chunk_id = int(item.get("chunk_id", -1))
    except (TypeError, ValueError):
        return None
    chunk = chunks_by_id.get(chunk_id)
    if not chunk:
        return None
    cleaned = filter_noise(chunk.get("text") or "")
    if not cleaned:
        return None

    # Locate the span in the chunk; if Claude paraphrased it, fall back to the whole chunk's times.
    idx = cleaned.lower().find(text.lower())
    if idx < 0:
        start_sec, end_sec = float(chunk["start"]), float(chunk["end"])
    else:
        start_sec, end_sec = entity_time_span(
            float(chunk["start"]), float(chunk["end"]), cleaned, idx, idx + len(text)
        )

    row: dict[str, Any] = {
        "type": typ,
        "text": text,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "chunk_id": chunk_id,
        "source": "claude",
    }
    if raw_label and raw_label != typ:
        row["original_label"] = raw_label
    return row
