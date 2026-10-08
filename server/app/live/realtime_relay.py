"""PARKED: live mic transcription via OpenAI Realtime (transcription session).

Only mounted when `ENABLE_LIVE_MODE=true` (see `app/live/README.md`).

Browser  --(binary PCM16 mono 24 kHz frames)-->  FastAPI /api/live  --(JSON events)-->  OpenAI Realtime
Browser  <--(JSON: partial / turn / status / error)--  FastAPI

Why a server relay instead of a browser-direct connection:
- OPENAI_API_KEY never leaves the server (no ephemeral-token plumbing needed).
- Entity extraction (spaCy / Claude) runs right after each completed turn, in the same process.

gpt-live-transcribe does not support server_vad / semantic_vad, so this module runs a small
energy-based speech gate and sends `input_audio_buffer.commit` itself.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
from array import array
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.services.entities import build_document, resolve_entity_backend, run_extraction

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
BYTES_PER_SAMPLE = 2

OPENAI_REALTIME_URL = os.environ.get(
    "OPENAI_REALTIME_URL", "wss://api.openai.com/v1/realtime?intent=transcription"
)
OPENAI_TRANSCRIBE_MODEL = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-live-transcribe")
# minimal | low | medium | high | xhigh  (lower = earlier partials, higher = more accurate)
OPENAI_TRANSCRIBE_DELAY = os.environ.get("OPENAI_TRANSCRIBE_DELAY", "low").strip()
OPENAI_TRANSCRIBE_PROMPT = os.environ.get(
    "OPENAI_TRANSCRIBE_PROMPT",
    "A podcast conversation that mentions people, companies, places and technologies.",
).strip()
# Comma-separated ISO 639-1 codes, e.g. "en" or "en,hi". Empty = let the model decide.
OPENAI_TRANSCRIBE_LANGUAGES = [
    c.strip() for c in os.environ.get("OPENAI_TRANSCRIBE_LANGUAGES", "").split(",") if c.strip()
]

# Speech gate (client-side VAD from OpenAI's point of view).
VAD_RMS_THRESHOLD = float(os.environ.get("LIVE_VAD_RMS", "350"))  # int16 RMS
VAD_END_SILENCE_SEC = float(os.environ.get("LIVE_VAD_END_SILENCE_SEC", "0.7"))
VAD_MAX_TURN_SEC = float(os.environ.get("LIVE_VAD_MAX_TURN_SEC", "12"))
VAD_IDLE_CLEAR_SEC = float(os.environ.get("LIVE_VAD_IDLE_CLEAR_SEC", "3"))
VAD_PAD_SEC = 0.25
MIN_COMMIT_SEC = 0.12  # API rejects commits under ~100 ms


def realtime_configured() -> bool:
    """True when an OpenAI key is available for the realtime relay."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _rms_int16(pcm: bytes) -> float:
    """Root-mean-square loudness of a PCM16 little-endian buffer (0 for empty input)."""
    if len(pcm) < 2:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples))


@dataclass
class _Turn:
    """A committed speech turn and its place on the session timeline."""

    seq: int
    start_sec: float
    end_sec: float


@dataclass
class _GateState:
    """Tracks the currently open (uncommitted) input buffer."""

    buffered_sec: float = 0.0
    speech_seen: bool = False
    first_speech_sec: float = 0.0
    last_speech_sec: float = 0.0
    silence_run_sec: float = 0.0

    def reset(self) -> None:
        """Forget the current turn (called after commit or clear)."""
        self.buffered_sec = 0.0
        self.speech_seen = False
        self.first_speech_sec = 0.0
        self.last_speech_sec = 0.0
        self.silence_run_sec = 0.0


@dataclass
class LiveSession:
    """State for one browser connection: audio clock, speech gate, and ordered turn output."""

    client: WebSocket
    entity_backend: str | None
    extract_entities: bool
    upstream: Any = None
    session_sec: float = 0.0  # total audio seconds received (timeline clock)
    gate: _GateState = field(default_factory=_GateState)
    next_seq: int = 0
    awaiting_commit_ack: list[_Turn] = field(default_factory=list)
    turn_by_item: dict[str, _Turn] = field(default_factory=dict)
    early_completed: dict[str, str] = field(default_factory=dict)
    results: dict[int, dict | None] = field(default_factory=dict)
    next_emit: int = 0
    pending_tasks: set[asyncio.Task] = field(default_factory=set)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ---------- browser I/O ----------
    async def to_client(self, payload: dict) -> None:
        """Send a JSON message to the browser (silently ignored if it disconnected)."""
        async with self.send_lock:
            try:
                await self.client.send_text(json.dumps(payload))
            except Exception:  # client already gone
                pass

    async def status(self) -> None:
        """Tell the browser how many committed turns are still being transcribed."""
        await self.to_client({"type": "status", "pending": self.next_seq - self.next_emit})

    # ---------- upstream I/O ----------
    async def to_openai(self, event: dict) -> None:
        """Send one Realtime API event upstream."""
        await self.upstream.send(json.dumps(event))

    def session_update(self) -> dict:
        """Build the `session.update` event that configures a transcription-only session."""
        transcription: dict[str, Any] = {"model": OPENAI_TRANSCRIBE_MODEL}
        if OPENAI_TRANSCRIBE_PROMPT:
            transcription["prompt"] = OPENAI_TRANSCRIBE_PROMPT
        if OPENAI_TRANSCRIBE_LANGUAGES:
            transcription["languages"] = OPENAI_TRANSCRIBE_LANGUAGES
        if OPENAI_TRANSCRIBE_DELAY and OPENAI_TRANSCRIBE_MODEL == "gpt-live-transcribe":
            transcription["delay"] = OPENAI_TRANSCRIBE_DELAY
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "transcription": transcription,
                        "turn_detection": None,
                    }
                },
            },
        }

    # ---------- speech gate ----------
    async def on_audio(self, pcm: bytes) -> None:
        """Forward a PCM frame upstream and run the speech gate that decides when to commit."""
        if not pcm:
            return
        frame_sec = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        frame_start = self.session_sec
        self.session_sec += frame_sec

        await self.to_openai(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}
        )

        g = self.gate
        g.buffered_sec += frame_sec
        if _rms_int16(pcm) >= VAD_RMS_THRESHOLD:
            if not g.speech_seen:
                g.speech_seen = True
                g.first_speech_sec = frame_start
            g.last_speech_sec = self.session_sec
            g.silence_run_sec = 0.0
        else:
            g.silence_run_sec += frame_sec

        if g.speech_seen:
            turn_len = self.session_sec - g.first_speech_sec
            if g.silence_run_sec >= VAD_END_SILENCE_SEC or turn_len >= VAD_MAX_TURN_SEC:
                await self.commit()
        elif g.buffered_sec >= VAD_IDLE_CLEAR_SEC:
            # Pure silence: drop it so we don't pay for / transcribe noise.
            await self.to_openai({"type": "input_audio_buffer.clear"})
            g.reset()

    async def commit(self) -> None:
        """Close the current speech turn and ask OpenAI to finalize its transcript."""
        g = self.gate
        if not g.speech_seen or g.buffered_sec < MIN_COMMIT_SEC:
            return
        turn = _Turn(
            seq=self.next_seq,
            start_sec=max(0.0, g.first_speech_sec - VAD_PAD_SEC),
            end_sec=min(self.session_sec, g.last_speech_sec + VAD_PAD_SEC),
        )
        self.next_seq += 1
        self.awaiting_commit_ack.append(turn)
        g.reset()
        await self.to_openai({"type": "input_audio_buffer.commit"})
        await self.status()

    # ---------- upstream events ----------
    async def on_openai_event(self, ev: dict) -> None:
        """Handle one upstream event: map item ids to turns, stream partials, finalize turns."""
        etype = ev.get("type", "")
        if etype == "input_audio_buffer.committed":
            item_id = ev.get("item_id")
            if self.awaiting_commit_ack and item_id:
                turn = self.awaiting_commit_ack.pop(0)
                self.turn_by_item[item_id] = turn
                if item_id in self.early_completed:
                    self._spawn_finalize(turn, self.early_completed.pop(item_id), item_id)
        elif etype == "conversation.item.input_audio_transcription.delta":
            await self.to_client(
                {"type": "partial", "item_id": ev.get("item_id"), "delta": ev.get("delta") or ""}
            )
        elif etype == "conversation.item.input_audio_transcription.completed":
            item_id = ev.get("item_id") or ""
            text = (ev.get("transcript") or "").strip()
            turn = self.turn_by_item.pop(item_id, None)
            if turn is None and self.awaiting_commit_ack and not self.turn_by_item:
                # No committed ack seen (older event shape): fall back to commit order.
                turn = self.awaiting_commit_ack.pop(0)
            if turn is None:
                self.early_completed[item_id] = text
                return
            self._spawn_finalize(turn, text, item_id)
        elif etype == "conversation.item.input_audio_transcription.failed":
            item_id = ev.get("item_id") or ""
            turn = self.turn_by_item.pop(item_id, None)
            err = (ev.get("error") or {}).get("message", "transcription failed")
            await self.to_client({"type": "error", "message": f"Turn failed: {err}"})
            if turn is not None:
                self.results[turn.seq] = None
                await self.flush()
        elif etype == "error":
            err = ev.get("error") or {}
            msg = err.get("message") or "OpenAI realtime error"
            # Benign race: commit/clear on an empty buffer.
            if err.get("code") == "input_audio_buffer_commit_empty":
                if self.awaiting_commit_ack:
                    t = self.awaiting_commit_ack.pop(0)
                    self.results[t.seq] = None
                    await self.flush()
                return
            logger.warning("realtime error: %s", err)
            await self.to_client({"type": "error", "message": msg})
        elif etype in ("session.created", "session.updated", "transcription_session.updated"):
            if etype != "session.created":
                await self.to_client({"type": "ready", "model": OPENAI_TRANSCRIBE_MODEL})

    def _spawn_finalize(self, turn: _Turn, text: str, item_id: str) -> None:
        """Run `_finalize` in the background and track the task for cleanup."""
        task = asyncio.create_task(self._finalize(turn, text, item_id))
        self.pending_tasks.add(task)
        task.add_done_callback(self.pending_tasks.discard)

    async def _finalize(self, turn: _Turn, text: str, item_id: str) -> None:
        """Tag entities for a finished turn (off the event loop) and queue it for ordered delivery."""
        if not text:
            self.results[turn.seq] = None
            await self.to_client({"type": "partial_done", "item_id": item_id})
            await self.flush()
            return
        seg_id = turn.seq * 1000
        segment = {"id": seg_id, "start": turn.start_sec, "end": turn.end_sec, "text": text}
        document = None
        entity_error = None
        if self.extract_entities:
            backend = resolve_entity_backend(self.entity_backend)
            try:
                normalized, entities = await asyncio.to_thread(
                    run_extraction, [dict(segment)], backend=backend
                )
                document = build_document(
                    chunks=normalized,
                    entities=entities,
                    source_label=f"live#turn{turn.seq}",
                    backend=backend,
                )
            except Exception as e:  # keep the transcript even if NER fails
                logger.exception("entity extraction failed for live turn %s", turn.seq)
                entity_error = str(e)
        self.results[turn.seq] = {
            "type": "turn",
            "seq": turn.seq,
            "item_id": item_id,
            "transcript": text,
            "segments": [segment],
            "document": document,
            "entity_error": entity_error,
        }
        await self.flush()

    async def flush(self) -> None:
        """Emit finished turns strictly in commit order (completions can arrive out of order)."""
        while self.next_emit in self.results:
            payload = self.results.pop(self.next_emit)
            self.next_emit += 1
            if payload is not None:
                await self.to_client(payload)
        await self.status()


async def handle_live_socket(ws: WebSocket) -> None:
    """Bridge one browser WebSocket to an OpenAI Realtime transcription session until either side closes."""
    import websockets  # shipped with uvicorn[standard]; pinned in requirements.txt

    await ws.accept()
    if not realtime_configured():
        await ws.send_text(json.dumps({"type": "error", "message": "OPENAI_API_KEY is not set on the server"}))
        await ws.close(code=1011)
        return

    qp = ws.query_params
    session = LiveSession(
        client=ws,
        entity_backend=qp.get("entity_backend") or None,
        extract_entities=qp.get("extract_entities", "true").lower() != "false",
    )
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY'].strip()}"}

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL, additional_headers=headers, max_size=None, ping_interval=20
        ) as upstream:
            session.upstream = upstream
            await session.to_openai(session.session_update())

            async def pump_upstream() -> None:
                """Relay every upstream OpenAI event to the session until the socket closes."""
                async for raw in upstream:
                    try:
                        ev = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    await session.on_openai_event(ev)
                await session.to_client({"type": "error", "message": "OpenAI realtime connection closed"})

            upstream_task = asyncio.create_task(pump_upstream())
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("bytes"):
                        await session.on_audio(msg["bytes"])
                    elif msg.get("text"):
                        try:
                            ctrl = json.loads(msg["text"])
                        except ValueError:
                            continue
                        if ctrl.get("type") == "stop":
                            await session.commit()
                            # Give in-flight turns a moment to complete, then tell the client we're done.
                            deadline = asyncio.get_running_loop().time() + 8.0
                            while (
                                session.next_emit < session.next_seq
                                and asyncio.get_running_loop().time() < deadline
                            ):
                                await asyncio.sleep(0.1)
                            await session.to_client({"type": "stopped"})
                            break
            finally:
                upstream_task.cancel()
                for t in list(session.pending_tasks):
                    t.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("live relay failed")
        await session.to_client({"type": "error", "message": f"Live relay failed: {e}"})
    finally:
        try:
            await ws.close()
        except Exception:
            pass
