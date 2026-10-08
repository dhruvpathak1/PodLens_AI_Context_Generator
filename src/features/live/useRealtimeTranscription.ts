// PARKED (future scope): not imported by the app. See ./README.md for how to re-enable live mode.
/** OpenAI Realtime transcription over the backend WebSocket relay (the newer live mode). */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { EntityDocument } from '../../types/entities'
import type { TranscribeResult } from '../../api/transcribe'
import type { TranscriptSegment } from '../../types/transcript'
import { apiUrl } from '../../config/env'
import { pcmCaptureWorkletUrl } from './pcmCaptureWorklet'

/** One finished speech turn from the server, already in commit (time) order. */
export type RealtimeTurn = TranscribeResult & { seq: number }

/**
 * Messages sent by the backend relay (app/live/realtime_relay.py):
 * - `ready`: OpenAI session configured; audio may flow.
 * - `status`: number of committed turns still being tagged.
 * - `partial` / `partial_done`: streaming text for an in-progress turn.
 * - `turn`: final text + segments + entity document for one speech turn.
 * - `error` / `stopped`: failure, or clean end after the client sent `stop`.
 */
type ServerMsg =
  | { type: 'ready'; model?: string }
  | { type: 'status'; pending: number }
  | { type: 'partial'; item_id: string; delta: string }
  | { type: 'partial_done'; item_id: string }
  | {
      type: 'turn'
      seq: number
      item_id: string
      transcript: string
      segments: TranscriptSegment[]
      document: EntityDocument | null
      entity_error: string | null
    }
  | { type: 'error'; message: string }
  | { type: 'stopped' }

type Options = {
  /** Called once per finished turn, strictly in order. */
  onTurn: (turn: RealtimeTurn) => void | Promise<void>
  /** Called for relay or socket errors. */
  onError?: (message: string) => void
  /** NER backend override forwarded as `?entity_backend=`. */
  entityBackend?: 'spacy' | 'claude'
}

/** WebSocket URL of the relay: `ws(s)://<api host>/api/live` (same origin via the Vite proxy in dev). */
export function resolveLiveSocketUrl(): string {
  const url = new URL(apiUrl('/api/live'), window.location.href)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

/** True when the API reports OPENAI_API_KEY is configured for realtime transcription. */
export async function isRealtimeTranscriptionAvailable(): Promise<boolean> {
  try {
    const r = await fetch(apiUrl('/api/health'))
    if (!r.ok) return false
    const j = (await r.json()) as { realtime_transcription?: boolean }
    return j.realtime_transcription === true
  } catch {
    return false
  }
}

/**
 * Streams microphone PCM to the FastAPI relay (/api/live), which forwards it to OpenAI Realtime
 * transcription. Exposes live partial text and calls `onTurn` with final, time-stamped turns
 * (plus NER document) in order.
 *
 * Pipeline: getUserMedia -> AudioWorklet (PCM16 @ 24 kHz, pcmCaptureWorklet.ts) -> WebSocket binary
 * frames -> relay -> OpenAI -> JSON messages back (see `ServerMsg`).
 *
 * Known issue: sessions have been seen closing after 2 to 3 seconds; see ./README.md.
 */
export function useRealtimeTranscription({ onTurn, onError, entityBackend }: Options) {
  const [active, setActive] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [micError, setMicError] = useState<string | null>(null)
  const [pendingTurns, setPendingTurns] = useState(0)
  const [partialText, setPartialText] = useState('')

  // Live resources, held in refs so callbacks always see the current ones.
  const wsRef = useRef<WebSocket | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const nodeRef = useRef<AudioWorkletNode | null>(null)
  /** In-progress text per OpenAI item id, joined for display. */
  const partialsRef = useRef<Map<string, string>>(new Map())
  const onTurnRef = useRef(onTurn)
  const onErrorRef = useRef(onError)
  /** Promise chain that serializes `onTurn` calls. */
  const turnChainRef = useRef<Promise<void>>(Promise.resolve())

  useEffect(() => {
    onTurnRef.current = onTurn
    onErrorRef.current = onError
  }, [onTurn, onError])

  /** Push the joined partial text to state. */
  const renderPartials = () => {
    setPartialText(Array.from(partialsRef.current.values()).join(' ').trim())
  }

  /** Release the mic, worklet and AudioContext (socket stays open). */
  const teardownAudio = useCallback(() => {
    nodeRef.current?.port.close()
    nodeRef.current?.disconnect()
    nodeRef.current = null
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    void ctxRef.current?.close().catch(() => undefined)
    ctxRef.current = null
  }, [])

  /** Release everything, including the socket, and reset state. */
  const teardownAll = useCallback(() => {
    teardownAudio()
    const ws = wsRef.current
    wsRef.current = null
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close()
    partialsRef.current.clear()
    setPartialText('')
    setPendingTurns(0)
    setActive(false)
    setConnecting(false)
  }, [teardownAudio])

  /** Graceful stop: ask the relay to flush, force-close after 10 s if it never answers. */
  const stop = useCallback(() => {
    // Stop capturing immediately; let the server flush the last turn, then it sends `stopped`.
    teardownAudio()
    setActive(false)
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }))
      window.setTimeout(() => {
        if (wsRef.current === ws) teardownAll()
      }, 10_000)
    } else {
      teardownAll()
    }
  }, [teardownAudio, teardownAll])

  /** Open the mic and socket, wait for `open`, then start streaming PCM frames. */
  const start = useCallback(async () => {
    setMicError(null)
    setConnecting(true)
    partialsRef.current.clear()
    setPartialText('')
    turnChainRef.current = Promise.resolve()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
      streamRef.current = stream

      const url = new URL(resolveLiveSocketUrl())
      if (entityBackend) url.searchParams.set('entity_backend', entityBackend)
      const ws = new WebSocket(url.toString())
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return
        let msg: ServerMsg
        try {
          msg = JSON.parse(ev.data) as ServerMsg
        } catch {
          return
        }
        switch (msg.type) {
          case 'ready':
            setConnecting(false)
            break
          case 'status':
            setPendingTurns(msg.pending)
            break
          case 'partial': {
            const m = partialsRef.current
            m.set(msg.item_id, (m.get(msg.item_id) ?? '') + msg.delta)
            renderPartials()
            break
          }
          case 'partial_done':
            partialsRef.current.delete(msg.item_id)
            renderPartials()
            break
          case 'turn': {
            partialsRef.current.delete(msg.item_id)
            renderPartials()
            const turn: RealtimeTurn = {
              seq: msg.seq,
              transcript: msg.transcript,
              segments: msg.segments,
              document: msg.document,
              entityError: msg.entity_error,
            }
            // Serialize so enrichment for turn N finishes before turn N+1 is applied.
            turnChainRef.current = turnChainRef.current
              .then(() => onTurnRef.current(turn))
              .catch((err) => console.warn('[PodLens live] onTurn error', err))
            break
          }
          case 'error':
            onErrorRef.current?.(msg.message)
            break
          case 'stopped':
            teardownAll()
            break
        }
      }
      ws.onerror = () => onErrorRef.current?.('Live transcription socket error')
      ws.onclose = () => {
        if (wsRef.current === ws) teardownAll()
      }

      // Wait for the socket before starting audio so no frames are dropped.
      await new Promise<void>((resolve, reject) => {
        ws.addEventListener('open', () => resolve(), { once: true })
        ws.addEventListener('close', () => reject(new Error('Could not connect to /api/live')), { once: true })
      })

      const ctx = new AudioContext()
      ctxRef.current = ctx
      await ctx.audioWorklet.addModule(pcmCaptureWorkletUrl())
      const source = ctx.createMediaStreamSource(stream)
      const node = new AudioWorkletNode(ctx, 'pcm-capture', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        processorOptions: { targetRate: 24_000 },
      })
      nodeRef.current = node
      node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        const sock = wsRef.current
        if (sock && sock.readyState === WebSocket.OPEN) sock.send(e.data)
      }
      source.connect(node)
      if (ctx.state === 'suspended') await ctx.resume()
      setActive(true)
    } catch (e) {
      setMicError(e instanceof Error ? e.message : 'Microphone access failed')
      teardownAll()
    }
  }, [entityBackend, teardownAll])

  // Unmount: release mic and socket.
  useEffect(() => () => teardownAll(), [teardownAll])

  return {
    start,
    stop,
    active,
    connecting,
    micError,
    pendingTurns,
    partialText,
    clearMicError: () => setMicError(null),
  }
}
