"""Entity endpoints: tag entities in existing chunks, and enrich entities into source cards."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import EnrichEntitiesRequest, ExtractEntitiesRequest
from app.core.config import settings
from app.services import storage
from app.services.enrichment import enrich_entities_payload
from app.services.entities import build_document, resolve_entity_backend, run_extraction, save_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["entities"])


@router.post("/api/extract-entities")
async def extract_entities(body: ExtractEntitiesRequest) -> dict:
    """Tag entities in caller-supplied chunks. Response: `{document, saved_path}`."""
    if not body.chunks:
        raise HTTPException(status_code=400, detail="chunks must be non-empty")
    backend = resolve_entity_backend(body.backend)
    try:
        normalized, entities = run_extraction([c.model_dump() for c in body.chunks], backend=backend)
    except Exception as e:
        logger.exception("entity extraction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    document = build_document(chunks=normalized, entities=entities, source_label=body.source_label, backend=backend)
    saved_path: str | None = None
    if body.persist:
        base = storage.safe_stem(body.source_label, fallback="entities") if body.source_label else "entities"
        saved_path = str(save_document(document, settings.entity_json_dir, base))
    return {"document": document, "saved_path": saved_path}


@router.post("/api/enrich-entities")
async def enrich_entities(body: EnrichEntitiesRequest) -> dict:
    """Build source cards (Wikipedia, map, photo). Response: `{cards[], count, unsplash_enabled}`."""
    if not body.entities:
        raise HTTPException(status_code=400, detail="entities must be non-empty")
    try:
        return await enrich_entities_payload([e.model_dump() for e in body.entities])
    except Exception as e:
        logger.exception("enrich-entities failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
