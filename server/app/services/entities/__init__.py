"""Named-entity recognition for transcripts.

Public API:
    resolve_entity_backend  choose spaCy or Claude
    run_extraction          tag entities in timestamped chunks
    build_document          package results as the versioned entity document
    extract_document        run_extraction + build_document in one call
    save_document           write an entity document to disk

Backends live in `spacy_backend.py` and `claude_backend.py`; post-processing rules live in
`text_cleaning.py` and `disambiguation.py`.
"""

from app.services.entities.pipeline import (
    build_document,
    extract_document,
    resolve_entity_backend,
    run_extraction,
    save_document,
)
from app.services.entities.types import BACKENDS, ENTITY_TYPES

__all__ = [
    "BACKENDS",
    "ENTITY_TYPES",
    "build_document",
    "extract_document",
    "resolve_entity_backend",
    "run_extraction",
    "save_document",
]
