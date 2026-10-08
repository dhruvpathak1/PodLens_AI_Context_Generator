# Live mode (parked, frontend)

Live microphone transcription is **future scope**. Everything below compiles with the app but
nothing imports it, so it is tree-shaken out of the production bundle.

## Files

| File | Purpose |
|---|---|
| `useRealtimeTranscription.ts` | Preferred engine. Streams mic PCM over WebSocket `/api/live` to the backend relay (OpenAI Realtime). Emits partial text and ordered final turns. |
| `pcmCaptureWorklet.ts` | AudioWorklet: mono, 24 kHz, PCM16 frames (~100 ms) for the realtime engine. |
| `useLiveMicRecorder.ts` | Fallback engine. MediaRecorder emits 10 s WebM slices. |
| `webmInitPrefix.ts` | Prepends the WebM header to each slice so ffmpeg can decode slices on their own. |
| `api.ts` | `transcribeLiveAudioChunk()` for `POST /api/transcribe-chunk`. |
| `liveChunkPipeline.ts` | Transcribes slices in parallel (max 2) and applies results strictly in order. |
| `liveSessionMerge.ts` | Grows one session transcript and entity document as turns arrive. |
| `unknownTranscript.ts` | Placeholder segment for a failed slice. |
| `live.css` | Styles for the live controls and partial-text line (not imported). |

## How it was wired (reference)

The original `App.tsx` used these pieces as follows. Rebuild it as a `useLiveSession` hook
in this folder rather than putting it back into `App.tsx`.

```tsx
// Engines
const livePipeline = useMemo(() => createOrderedLiveChunkPipeline({
  maxConcurrent: LIVE_TRANSCRIBE_MAX_CONCURRENT,
  transcribe: (blob, i) => transcribeLiveAudioChunk(blob, {
    chunkSeq: i, timeOffsetSec: i * (LIVE_CHUNK_INTERVAL_MS / 1000), backend: ENTITY_BACKEND,
  }),
  apply: (i, res) => applyLiveChunkRef.current(i, res),
  onTranscribeFailure: (i, err) => onLiveFailRef.current(i, err),
  setBusy: setLiveProcessing,
}), [])
const liveMic = useLiveMicRecorder({ onChunk: (blob, i) => livePipeline.submit(blob, i) })
const realtime = useRealtimeTranscription({
  entityBackend: ENTITY_BACKEND,
  onTurn: (turn) => applyLiveChunkRef.current(turn.seq, turn),
  onError: setLiveChunkError,
})

// Start: reset all episode state, then pick the engine from /api/health
const useRealtime = await isRealtimeTranscriptionAvailable()
if (useRealtime) await realtime.start(); else await liveMic.start()

// Apply one turn/slice (in order)
applyLiveChunkRef.current = async (i, res) => {
  // 1. append res.transcript to the transcript text (or UNKNOWN_TRANSCRIPT_SENTENCE)
  // 2. setSegments(prev => mergeTranscriptSegments(prev, res.segments
  //      .length ? res.segments : unknownLiveTranscriptSegments(i, 10)))
  // 3. setEntityDoc(prev => mergeEntityDocuments(prev, res.document))
  // 4. enrich only entities with no card yet (dedupe with entityMatchKey), and update a
  //    cards ref synchronously so the next turn sees the new cards before React re-renders
}

// Failed slice: show the error and insert a placeholder segment
onLiveFailRef.current = async (i, err) => {
  setLiveChunkError(String(err))
  setSegments(prev => mergeTranscriptSegments(prev, unknownLiveTranscriptSegments(i, 10)))
}

// Stop: stop both engines; set playback time/duration to the elapsed session length so
// the transcript highlight and Live cards treat the session like a finished recording.
```

## Re-enabling checklist

1. Backend: `ENABLE_LIVE_MODE=true` (see `server/app/live/README.md`).
2. Set `LIVE_MODE_ENABLED = true` in `src/config/env.ts`.
3. Add `useLiveSession.ts` here (wiring above) and a Start/Stop button in `TranscriptSidebar`.
4. `@import '../features/live/live.css';` in `src/styles/index.css`.
5. Fix first: the realtime session closed after 2 to 3 seconds in testing.
