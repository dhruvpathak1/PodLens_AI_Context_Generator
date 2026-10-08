"""Entity type vocabulary shared by all NER backends and the frontend."""

from __future__ import annotations

#: User-facing entity types. MISC follows CoNLL "miscellaneous" (not person/org/location).
ENTITY_TYPES = frozenset({"PLACE", "PERSON", "TECHNOLOGY", "EVENT", "COMPANY", "MISC"})

#: Supported NER backends.
BACKENDS = ("spacy", "claude")
