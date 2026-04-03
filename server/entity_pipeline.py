"""Entity extraction from timestamped transcript chunks (spaCy or Claude)."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- Allowed output types (user-facing) ---
ENTITY_TYPES = frozenset({"PLACE", "PERSON", "TECHNOLOGY", "EVENT", "COMPANY"})

# Filler / disfluency phrases (removed before NER)
_FILLER_PATTERN = re.compile(
    r"\b("
    r"um|uh|er|ah|hmm|hm|mm|uhm|erm|"
    r"like(?=\s+(?:i|you|the|a|an|this|that|it)\b)|"
    r"you know|i mean|i guess|sort of|kind of|"
    r"actually|basically|literally|obviously|honestly|"
    r"right\?|okay\?|ok\?"
    r")\b",
    re.IGNORECASE,
)

# Whole-line or clause noise
_NOISE_PHRASES = re.compile(
    r"\b(thanks for watching|subscribe|like and subscribe|see you next time)\b",
    re.IGNORECASE,
)

# Post-filter: drop entity text matching these (too generic)
_GENERIC_ENTITIES = frozenset(
    {
        "today",
        "tomorrow",
        "here",
        "there",
        "now",
        "something",
        "someone",
        "anything",
        "everything",
        "nothing",
        "people",
        "things",
        "stuff",
        "way",
        "thing",
        "lot",
        "bit",
    }
)


def filter_noise(text: str) -> str:
    """Strip filler words and common throwaway phrases."""
    if not text or not text.strip():
        return ""
    t = _NOISE_PHRASES.sub("", text)
    t = _FILLER_PATTERN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _entity_time_span(
    chunk_start: float, chunk_end: float, chunk_text: str, ent_start_char: int, ent_end_char: int
) -> tuple[float, float]:
    """Map character span inside chunk to approximate audio times."""
    n = max(len(chunk_text), 1)
    dur = max(chunk_end - chunk_start, 0.0)
    rel0 = max(0, min(ent_start_char, n)) / n
    rel1 = max(0, min(ent_end_char, n)) / n
    return chunk_start + rel0 * dur, chunk_start + rel1 * dur


def _clean_entity_text(text: str) -> str | None:
    t = text.strip()
    if len(t) < 2:
        return None
    low = t.lower()
    if low in _GENERIC_ENTITIES:
        return None
    return t


# Context cues for homonyms (company vs common noun, etc.). Chunk text is lowercased.
_APPLE_TECH = re.compile(
    r"\b(iphone|ipad|ipod|macbook|imac|ios|ipados|mac\s*os|app\s+store|cupertino|"
    r"tim\s+cook|steve\s+jobs|wwdc|airpods|apple\s+watch|icloud|m1\b|m2\b|m3\b|m4\b|"
    r"a\d+\s+bionic|vision\s+pro|apple\s+park|nasdaq|aapl|ecosystem|siri|facetime|"
    r"apple\s+music|apple\s+tv|mac\s+studio|ipad\s+pro|developer\s+conference|"
    r"silicon|osx|watchos|testflight|"
    r"apple\s+(announced|reported|reports|unveiled|released|launched|introduced|said|posted))\b",
    re.I,
)
_APPLE_FRUIT = re.compile(
    r"\b(fruit|apple\s+pie|apples\s+and|orchard|apple\s+juice|recipe|granny\s+smith|"
    r"cider|rotten\s+apple|red\s+delicious|gala\s+apple|honeycrisp|peel|crisp\s+apple|"
    r"\ban apple\b|\bthe apple\s+was\b|\bapples\s+(are|were|taste))\b",
    re.I,
)
_AMAZON_CORP = re.compile(
    r"\b(aws|amazon\s+prime|prime\s+video|kindle|alexa|bezos|e-?commerce|marketplace|"
    r"amazon\.com|fulfillment|amazon\s+web)\b",
    re.I,
)
_AMAZON_RIVER = re.compile(
    r"\b(rainforest|amazon\s+river|amazon\s+basin|manaus|peru|amazonia)\b",
    re.I,
)
_ORACLE_TECH = re.compile(
    r"\b(database|sql|java\b|oci\b|oracle\s+corp|larry\s+ellison|oracle\s+cloud|"
    r"enterprise\s+software|erp)\b",
    re.I,
)
_ORACLE_MYTH = re.compile(
    r"\b(myth|prophecy|greek|delphi|ancient|pythia|apollo)\b",
    re.I,
)
_META_CORP = re.compile(
    r"\b(facebook|instagram|whatsapp|zuckerberg|meta\s+quest|threads|reality\s+labs|"
    r"oculus|meta\s+platforms)\b",
    re.I,
)


def _apply_context_disambiguation(
    entity: dict[str, Any], chunk_lower: str
) -> dict[str, Any] | None:
    """Adjust or drop entities when chunk context resolves homonyms."""
    e = dict(entity)
    text = (e.get("text") or "").strip()
    low = text.lower()
    typ = str(e.get("type", ""))

    if low == "apple":
        tech = _APPLE_TECH.search(chunk_lower)
        fruit = _APPLE_FRUIT.search(chunk_lower)
        if tech and not fruit:
            e["type"] = "COMPANY"
            return e
        if fruit and not tech:
            return None
        if tech and fruit:
            e["type"] = "COMPANY"
            return e
        # Bare \"Apple\" / PRODUCT label in business podcasts: treat as the company, not the fruit.
        if typ == "TECHNOLOGY":
            e["type"] = "COMPANY"
            return e
        if typ == "COMPANY":
            return e
        if typ == "PERSON":
            e["type"] = "COMPANY"
            return e
        e["type"] = "COMPANY"
        return e

    if low == "amazon":
        corp = _AMAZON_CORP.search(chunk_lower)
        river = _AMAZON_RIVER.search(chunk_lower)
        if corp and not river:
            e["type"] = "COMPANY"
            return e
        if river and not corp and typ == "COMPANY":
            e["type"] = "PLACE"
            return e
        return e

    if low == "oracle":
        sw = _ORACLE_TECH.search(chunk_lower)
        myth = _ORACLE_MYTH.search(chunk_lower)
        if sw and not myth:
            e["type"] = "COMPANY"
            return e
        if myth and not sw:
            e["type"] = "EVENT"
            return e
        return e

    if low == "meta":
        if _META_CORP.search(chunk_lower):
            e["type"] = "COMPANY"
        return e

    return e


def _refine_entities_with_chunk_context(
    chunks: list[dict[str, Any]], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {int(c["id"]): c for c in chunks if c.get("id") is not None}
    out: list[dict[str, Any]] = []
    for ent in entities:
        try:
            cid = int(ent.get("chunk_id", -1))
        except (TypeError, ValueError):
            out.append(ent)
            continue
        ch = by_id.get(cid)
        chunk_lower = ""
        if ch is not None:
            chunk_lower = (ch.get("text_clean") or ch.get("text") or "").lower()
        refined = _apply_context_disambiguation(ent, chunk_lower)
        if refined is not None:
            out.append(refined)
    return out


# --- spaCy ---
_spacy_nlp = None


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        model = os.environ.get("SPACY_MODEL", "en_core_web_sm")
        try:
            _spacy_nlp = spacy.load(model)
        except OSError as e:
            logger.error("spaCy model %r not installed.", model)
            raise RuntimeError(
                f"spaCy model {model!r} is missing. From the project venv run one of:\n"
                f"  python -m spacy download {model}\n"
                "  pip install -r server/requirements.txt\n"
                "(requirements.txt includes the en_core_web_sm wheel.)"
            ) from e
    return _spacy_nlp


# spaCy label -> our tag (None = drop). Tune here if your domain needs more labels.
_SPACY_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "ORG": "COMPANY",
    "PRODUCT": "TECHNOLOGY",
    "EVENT": "EVENT",
    "WORK_OF_ART": "EVENT",
    "LANGUAGE": "TECHNOLOGY",  # spoken languages, stacks mentioned as languages
    "LAW": "EVENT",  # named laws, cases — often discussed like events in podcasts
}


def extract_with_spacy(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nlp = _get_spacy()
    entities: list[dict[str, Any]] = []
    for ch in chunks:
        cid = int(ch.get("id", 0))
        t0, t1 = float(ch["start"]), float(ch["end"])
        raw = ch.get("text") or ""
        cleaned = filter_noise(raw)
        if not cleaned:
            continue
        doc = nlp(cleaned)
        for ent in doc.ents:
            tag = _SPACY_MAP.get(ent.label_)
            if not tag or tag not in ENTITY_TYPES:
                continue
            text = _clean_entity_text(ent.text)
            if not text:
                continue
            es, ee = _entity_time_span(t0, t1, cleaned, ent.start_char, ent.end_char)
            entities.append(
                {
                    "type": tag,
                    "text": text,
                    "start_sec": round(es, 3),
                    "end_sec": round(ee, 3),
                    "chunk_id": cid,
                    "source": "spacy",
                    "original_label": ent.label_,
                }
            )
    return entities


# --- Claude ---
_CLAUDE_SYSTEM = """You extract named entities from podcast transcript chunks.
Return ONLY valid JSON (no markdown). Schema:
{"entities":[{"type":"PERSON|PLACE|TECHNOLOGY|EVENT|COMPANY","text":"exact span","chunk_id":number}]}
Rules:
- type must be one of: PERSON, PLACE, TECHNOLOGY, EVENT, COMPANY
- PLACE: locations, cities, countries, venues
- TECHNOLOGY: products, platforms, languages, frameworks, technical terms (e.g. Python, Kubernetes)
- EVENT: conferences, wars, named historical events, sports finals
- COMPANY: organizations, brands that are companies
- Disambiguate homonyms using the chunk: e.g. "Apple" + iPhone/Mac/Cupertino → COMPANY (Apple Inc.), not the fruit.
  "Apple" + pie/juice/orchard/fruit context → omit or do not tag as a tech company.
  Same idea for Amazon (company vs river), Oracle (software vs myth), Meta (holding company vs generic "meta").
- Omit filler (um, uh, like, you know), generic words, and non-entities
- Do not duplicate the same text+type within one chunk
"""


def extract_with_claude(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set for Claude entity extraction")

    model = os.environ.get("CLAUDE_MODEL", "claude-3-5-haiku-20241022")
    client = anthropic.Anthropic(api_key=api_key)

    # Batch chunks to control tokens (~15 per call)
    batch_size = int(os.environ.get("CLAUDE_ENTITY_BATCH", "12"))
    all_entities: list[dict[str, Any]] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        lines = []
        for ch in batch:
            cid = int(ch.get("id", 0))
            cleaned = filter_noise(ch.get("text") or "")
            if not cleaned:
                continue
            lines.append(f"[chunk_id={cid} start={ch['start']:.2f} end={ch['end']:.2f}]\n{cleaned}")
        if not lines:
            continue
        user_block = "Chunks:\n\n" + "\n\n---\n\n".join(lines)

        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": user_block}],
        )
        raw_text = ""
        for block in msg.content:
            if block.type == "text":
                raw_text += block.text
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning("Claude JSON parse failed: %s | snippet=%r", e, raw_text[:400])
            continue

        ents = data.get("entities") if isinstance(data, dict) else None
        if not isinstance(ents, list):
            continue

        by_id = {int(c["id"]): c for c in batch if "id" in c}
        for item in ents:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "")).upper()
            if typ not in ENTITY_TYPES:
                continue
            text = _clean_entity_text(str(item.get("text", "")))
            if not text:
                continue
            try:
                cid = int(item.get("chunk_id", -1))
            except (TypeError, ValueError):
                continue
            ch = by_id.get(cid)
            if not ch:
                continue
            cleaned = filter_noise(ch.get("text") or "")
            if not cleaned:
                continue
            # Approximate char span: find first occurrence
            idx = cleaned.lower().find(text.lower())
            if idx < 0:
                es = float(ch["start"])
                ee = float(ch["end"])
            else:
                es, ee = _entity_time_span(
                    float(ch["start"]), float(ch["end"]), cleaned, idx, idx + len(text)
                )
            all_entities.append(
                {
                    "type": typ,
                    "text": text,
                    "start_sec": round(es, 3),
                    "end_sec": round(ee, 3),
                    "chunk_id": cid,
                    "source": "claude",
                }
            )

    return all_entities


def run_extraction(
    chunks: list[dict[str, Any]],
    backend: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (chunks_with_cleaned_text, entities).
    Each chunk dict should have id, start, end, text.
    """
    b = (backend or os.environ.get("ENTITY_BACKEND", "spacy")).lower()
    if b not in ("spacy", "claude"):
        b = "spacy"

    normalized: list[dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        cid = ch.get("id")
        if cid is None:
            cid = i
        cleaned = filter_noise(str(ch.get("text", "")))
        normalized.append(
            {
                "id": int(cid),
                "start": float(ch["start"]),
                "end": float(ch["end"]),
                "text": str(ch.get("text") or ""),
                "text_clean": cleaned,
            }
        )

    if b == "claude":
        entities = extract_with_claude([{**c, "text": c["text_clean"] or c["text"]} for c in normalized])
    else:
        entities = extract_with_spacy([{**c, "text": c["text_clean"] or c["text"]} for c in normalized])

    entities = _refine_entities_with_chunk_context(normalized, entities)
    return normalized, entities


def build_document(
    *,
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    source_label: str | None,
    backend: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "source_label": source_label,
        "chunks": [
            {
                "id": c["id"],
                "start_sec": round(float(c["start"]), 3),
                "end_sec": round(float(c["end"]), 3),
                "text_raw": c.get("text", ""),
                "text_clean": c.get("text_clean", ""),
            }
            for c in chunks
        ],
        "entities": entities,
    }


def save_document(doc: dict[str, Any], directory: Path, basename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", basename, flags=re.UNICODE)[:80] or "entities"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = directory / f"{safe}_{ts}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.resolve()
