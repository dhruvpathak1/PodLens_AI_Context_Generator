"""HTTP-level tests for routing, validation and error mapping (services are stubbed)."""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SEGMENTS = [{"id": 0, "start": 0, "end": 4, "text": "Nvidia was founded in 1993."}]


@pytest.fixture()
def client() -> TestClient:
    """FastAPI test client for a freshly created app."""
    return TestClient(create_app())


def test_health_reports_features(client: TestClient) -> None:
    """/api/health reports which optional features are configured."""
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["openai_configured"] is False
    assert body["live_mode"] is False


def test_heavy_modules_not_imported_at_startup() -> None:
    # Whisper/torch and LangChain load lazily, so the API starts fast and survives missing extras.
    """Whisper and LangChain are imported lazily, not at app startup."""
    create_app()
    assert "whisper" not in sys.modules
    assert "app.services.timeline" not in sys.modules or "langchain_openai" not in sys.modules


def test_live_routes_are_parked(client: TestClient) -> None:
    """Live endpoints are not mounted unless ENABLE_LIVE_MODE is set."""
    assert client.post("/api/transcribe-chunk").status_code == 404


def test_timeline_validation_and_missing_key(client: TestClient) -> None:
    """/api/timeline validates input and returns 503 without an OpenAI key."""
    assert client.post("/api/timeline", json={"segments": []}).status_code == 400
    r = client.post("/api/timeline", json={"segments": SEGMENTS})
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_ask_validation_and_missing_key(client: TestClient) -> None:
    """/api/ask validates input and returns 503 without an OpenAI key."""
    assert client.post("/api/ask", json={"question": " ", "segments": SEGMENTS}).status_code == 400
    assert client.post("/api/ask", json={"question": "When?", "segments": SEGMENTS}).status_code == 503


def test_transcribe_delegates_to_episode_service(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/transcribe passes the upload to the episode service."""
    from app.api.routes import transcription

    calls = {}

    def fake_process(contents, filename, **kw):
        """Stand-in for process_upload that records its arguments."""
        calls.update(filename=filename, size=len(contents), **kw)
        return {"transcript": "hi", "segments": [], "saved_path": "x", "document": None,
                "entity_saved_path": None, "entity_error": None}

    monkeypatch.setattr(transcription, "process_upload", fake_process)
    r = client.post("/api/transcribe", files={"audio": ("ep.mp3", b"abc")}, data={"entity_backend": "spacy"})
    assert r.status_code == 200 and r.json()["transcript"] == "hi"
    assert calls == {"filename": "ep.mp3", "size": 3, "language": None, "extract_entities": True, "entity_backend": "spacy"}


def test_transcribe_rejects_empty_upload(client: TestClient) -> None:
    """An empty upload is rejected with 400."""
    assert client.post("/api/transcribe", files={"audio": ("ep.mp3", b"")}).status_code == 400


def test_extract_entities_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/extract-entities tags entities in posted segments."""
    from app.api.routes import entities

    monkeypatch.setattr(entities, "run_extraction", lambda chunks, backend: (
        [{**c, "text_clean": c["text"]} for c in chunks],
        [{"type": "COMPANY", "text": "Nvidia", "start_sec": 0, "end_sec": 1, "chunk_id": 0}],
    ))
    r = client.post("/api/extract-entities", json={"chunks": SEGMENTS, "persist": False, "backend": "spacy"})
    assert r.status_code == 200
    doc = r.json()["document"]
    assert doc["backend"] == "spacy" and doc["entities"][0]["text"] == "Nvidia"
