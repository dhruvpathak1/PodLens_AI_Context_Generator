"""Entity enrichment: turn tagged entities into source cards.

Sources: Wikipedia (`wikipedia.py`), OpenStreetMap Nominatim (`geocoding.py`) and
Unsplash (`unsplash.py`). `service.py` combines them per entity.
"""

from app.services.enrichment.service import dedupe_entities, enrich_entities_payload, enrich_entity

__all__ = ["dedupe_entities", "enrich_entities_payload", "enrich_entity"]
