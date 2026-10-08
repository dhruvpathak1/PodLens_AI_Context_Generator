"""Entity extraction pipeline: clean -> tag (spaCy or Claude) -> disambiguate -> package.

The output "entity document" is what the frontend consumes and what gets saved to
`entity_exports/` as JSON.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services import storage
from app.services.entities.disambiguation import refine_entities_with_chunk_context
from app.services.entities.text_cleaning import filter_noise
from app.services.entities.types import BACKENDS

#: Bump when the entity document shape changes in a way the frontend must know about.
SCHEMA_VERSION = 1


def resolve_entity_backend(explicit: str | None = None) -> str:
    """Pick the NER backend.

    Priority: explicit request value -> `ENTITY_BACKEND` env -> Claude if `ANTHROPIC_API_KEY`
    is set -> spaCy. Unknown values fall back to spaCy.
    """
    if explicit and str(explicit).strip():
        backend = str(explicit).strip().lower()
    else:
        env_choice = os.environ.get("ENTITY_BACKEND", "").strip().lower()
        if env_choice in BACKENDS:
            backend = env_choice
        elif os.environ.get("ANTHROPIC_API_KEY", "").strip():
            backend = "claude"
        else:
            backend = "spacy"
    return backend if backend in BACKENDS else "spacy"


def normalize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce raw `{id?, start, end, text}` chunks into typed dicts with a `text_clean` field.

    Chunks without an id get their list index, so ids are always present downstream.
    """
    normalized: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("id")
        normalized.append(
            {
                "id": int(i if chunk_id is None else chunk_id),
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "text": str(chunk.get("text") or ""),
                "text_clean": filter_noise(str(chunk.get("text", ""))),
            }
        )
    return normalized


def run_extraction(
    chunks: list[dict[str, Any]], backend: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tag entities in timestamped chunks.

    Args:
        chunks: `{id, start, end, text}` dicts (Whisper segments).
        backend: "spacy" | "claude" | None (auto, see `resolve_entity_backend`).

    Returns:
        `(normalized_chunks, entities)`; each entity is
        `{type, text, start_sec, end_sec, chunk_id, source, original_label?}`.
    """
    chosen = resolve_entity_backend(backend)
    normalized = normalize_chunks(chunks)
    # Backends see the cleaned text (falls back to raw text if cleaning removed everything).
    backend_input = [{**c, "text": c["text_clean"] or c["text"]} for c in normalized]

    if chosen == "claude":
        from app.services.entities.claude_backend import extract_with_claude

        entities = extract_with_claude(backend_input)
    else:
        from app.services.entities.spacy_backend import extract_with_spacy

        entities = extract_with_spacy(backend_input)

    return normalized, refine_entities_with_chunk_context(normalized, entities)


def build_document(
    *,
    chunks: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    source_label: str | None,
    backend: str,
) -> dict[str, Any]:
    """Package chunks and entities into the versioned entity document returned by the API."""
    return {
        "schema_version": SCHEMA_VERSION,
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
    """Persist an entity document as JSON; returns the absolute file path."""
    return storage.save_json(directory, basename, doc)


def extract_document(
    segments: list[dict[str, Any]], *, source_label: str | None, backend: str | None
) -> dict[str, Any]:
    """Convenience wrapper: run extraction on segments and return the packaged document."""
    chosen = resolve_entity_backend(backend)
    chunks = [{"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]
    normalized, entities = run_extraction(chunks, backend=chosen)
    return build_document(chunks=normalized, entities=entities, source_label=source_label, backend=chosen)
