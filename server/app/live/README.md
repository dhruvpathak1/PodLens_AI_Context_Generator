# Live mode (parked)

Live microphone transcription is **future scope**. The code is kept here, documented and
compiling, but it is **not mounted** by the API unless you opt in.

## What is here

| File | Purpose |
|---|---|
| `realtime_relay.py` | WebSocket `/api/live`. Relays browser PCM16 audio to the OpenAI Realtime API (`gpt-live-transcribe`), runs a speech gate that commits turns, tags entities per turn, and returns turns in order. |
| `chunk_transcription.py` | `POST /api/transcribe-chunk`. Legacy fallback: 10 s WebM slices transcribed by local Whisper. |
| `__init__.py` | `build_live_router()` that mounts both endpoints. |

The matching frontend code lives in `src/features/live/` (also parked and not imported by the app).

## Re-enabling

1. Backend: set `ENABLE_LIVE_MODE=true` in `.env` and restart the API.
2. Frontend: follow `src/features/live/README.md`.
3. Known issue to fix first: the realtime connection closed after 2 to 3 seconds in testing.
   Verify `OPENAI_REALTIME_URL` and model access before shipping.
