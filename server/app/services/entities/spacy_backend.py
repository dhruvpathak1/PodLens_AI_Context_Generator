"""Local NER with spaCy (free, offline, fast; weaker on messy speech than Claude)."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.services.entities.text_cleaning import clean_entity_text, entity_time_span, filter_noise
from app.services.entities.types import ENTITY_TYPES

logger = logging.getLogger(__name__)

#: spaCy label -> PodLens entity type. Labels not listed here are dropped.
SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "GPE": "PLACE",  # countries, cities, states
    "LOC": "PLACE",  # mountains, rivers, regions
    "FAC": "PLACE",  # buildings, airports, bridges
    "ORG": "COMPANY",
    "PRODUCT": "TECHNOLOGY",
    "EVENT": "EVENT",
    "WORK_OF_ART": "EVENT",
    "LANGUAGE": "TECHNOLOGY",  # spoken languages and stacks mentioned as languages
    "LAW": "EVENT",  # named laws and cases are discussed like events in podcasts
    "NORP": "MISC",  # nationalities, religions, political groups (CoNLL-style misc)
}

_nlp: Any = None


def get_nlp() -> Any:
    """Load (once) and return the spaCy pipeline named by `SPACY_MODEL`.

    Raises:
        RuntimeError: the model package is not installed, with the command that fixes it.
    """
    global _nlp
    if _nlp is None:
        import spacy

        model = settings.spacy_model
        try:
            _nlp = spacy.load(model)
        except OSError as e:
            logger.error("spaCy model %r not installed.", model)
            raise RuntimeError(
                f"spaCy model {model!r} is missing. From the project venv run one of:\n"
                f"  python -m spacy download {model}\n"
                "  pip install -r server/requirements.txt\n"
                "(requirements.txt includes the en_core_web_sm wheel.)"
            ) from e
    return _nlp


def extract_with_spacy(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag entities in each `{id, start, end, text}` chunk and estimate when each is spoken."""
    nlp = get_nlp()
    entities: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = int(chunk.get("id", 0))
        t0, t1 = float(chunk["start"]), float(chunk["end"])
        cleaned = filter_noise(chunk.get("text") or "")
        if not cleaned:
            continue
        for ent in nlp(cleaned).ents:
            tag = SPACY_LABEL_MAP.get(ent.label_)
            if not tag or tag not in ENTITY_TYPES:
                continue
            text = clean_entity_text(ent.text)
            if not text:
                continue
            start_sec, end_sec = entity_time_span(t0, t1, cleaned, ent.start_char, ent.end_char)
            entities.append(
                {
                    "type": tag,
                    "text": text,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "chunk_id": chunk_id,
                    "source": "spacy",
                    "original_label": ent.label_,
                }
            )
    return entities
