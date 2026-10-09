"""Request bodies for the JSON endpoints (FastAPI validates these automatically).

Responses are plain dicts produced by the services; their shapes are documented on each route.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChunkIn(BaseModel):
    """A timestamped transcript segment."""

    id: int = 0
    start: float
    end: float
    text: str = ""


class EntityRefIn(BaseModel):
    """A tagged entity mention (as produced by `/api/transcribe`)."""

    type: str
    text: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    chunk_id: int = 0


class ChatTurnIn(BaseModel):
    """One earlier message in an Ask conversation."""

    role: str  # "user" | "assistant"
    content: str


class ExtractEntitiesRequest(BaseModel):
    """Body of `POST /api/extract-entities`."""

    chunks: list[ChunkIn]
    source_label: str | None = None
    persist: bool = True
    backend: str | None = None  # "spacy" | "claude" | None (server default)


class EnrichEntitiesRequest(BaseModel):
    """Body of `POST /api/enrich-entities`."""

    entities: list[EntityRefIn]
    #: Transcript segments; enables the LLM entity review (it needs the context lines).
    segments: list[ChunkIn] = []
    source_label: str | None = None
    #: Force the review on/off; null = server default (ENTITY_REVIEW and an OpenAI key).
    review: bool | None = None


class TimelineRequest(BaseModel):
    """Body of `POST /api/timeline`."""

    segments: list[ChunkIn]
    entities: list[EntityRefIn] = []
    source_label: str | None = None


class AskRequest(BaseModel):
    """Body of `POST /api/ask`."""

    question: str
    segments: list[ChunkIn]
    entities: list[EntityRefIn] = []
    history: list[ChatTurnIn] = []
    source_label: str | None = None
