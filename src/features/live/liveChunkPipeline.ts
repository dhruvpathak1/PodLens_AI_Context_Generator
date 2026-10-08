// PARKED (future scope): not imported by the app. See ./README.md for how to re-enable live mode.
/**
 * Ordered live chunk processing: up to `maxConcurrent` transcribe requests run in parallel,
 * while results are applied to UI/state strictly in chunk index order (so the next 10s can
 * transcribe while earlier chunks finish).
 */

import type { TranscribeResult } from '../../api/transcribe'

/** How many chunk uploads may be in flight at once. */
export const LIVE_TRANSCRIBE_MAX_CONCURRENT = 2

/** Minimal counting semaphore: caps concurrent async work at `max`. */
class AsyncSemaphore {
  private active = 0
  private readonly wait: Array<() => void> = []
  private readonly max: number

  constructor(max: number) {
    this.max = max
  }

  /** Wait for a free slot, then take it. */
  async acquire(): Promise<void> {
    if (this.active < this.max) {
      this.active++
      return
    }
    await new Promise<void>((resolve) => {
      this.wait.push(() => {
        this.active++
        resolve()
      })
    })
  }

  /** Free a slot and wake the next waiter (if any). */
  release(): void {
    this.active--
    const next = this.wait.shift()
    if (next) next()
  }
}

/** A finished chunk waiting for its turn to be applied (success or failure). */
type BufferOk = { kind: 'ok'; chunkIndex: number; res: TranscribeResult }
type BufferErr = { kind: 'err'; chunkIndex: number; error: unknown }

export type LiveChunkPipeline = {
  /** Queue one recorded chunk for transcription. */
  submit: (blob: Blob, chunkIndex: number) => void
  /** Drop all in-flight and buffered work (new session). */
  reset: () => void
}

/**
 * Create a pipeline that transcribes chunks concurrently but applies results in order.
 *
 * - `gen` is a session generation counter: `reset()` bumps it, so late results from an
 *   earlier session are silently discarded.
 * - `buffer` holds finished chunks keyed by index; `nextApply` is the next index to apply.
 * - `drainTail` chains drains so `apply` callbacks never overlap.
 */
export function createOrderedLiveChunkPipeline(opts: {
  /** Max parallel `transcribe` calls. */
  maxConcurrent: number
  /** Upload one chunk and return its transcript. */
  transcribe: (blob: Blob, chunkIndex: number) => Promise<TranscribeResult>
  /** Merge a successful result into app state (called in chunk order). */
  apply: (chunkIndex: number, res: TranscribeResult) => void | Promise<void>
  /** Handle a failed chunk (called in chunk order, e.g. insert a placeholder). */
  onTranscribeFailure: (chunkIndex: number, error: unknown) => void | Promise<void>
  /** Busy flag for the UI: true while any upload or apply is running. */
  setBusy: (busy: boolean) => void
}): LiveChunkPipeline {
  const sem = new AsyncSemaphore(opts.maxConcurrent)
  const buffer = new Map<number, BufferOk | BufferErr>()
  let nextApply = 0
  let gen = 0
  let drainTail = Promise.resolve()
  let busyDepth = 0

  /** Track nested busy work and report a single boolean. */
  const bumpBusy = (delta: number) => {
    busyDepth += delta
    opts.setBusy(busyDepth > 0)
  }

  /** Apply every buffered chunk that is next in line; stops at the first gap. */
  const enqueueDrain = () => {
    const g = gen
    drainTail = drainTail
      .catch(() => undefined)
      .then(async () => {
        if (g !== gen) return
        while (buffer.has(nextApply)) {
          if (g !== gen) return
          const entry = buffer.get(nextApply)!
          buffer.delete(nextApply)
          nextApply++
          bumpBusy(1)
          try {
            if (entry.kind === 'err') {
              await Promise.resolve(opts.onTranscribeFailure(entry.chunkIndex, entry.error))
            } else {
              await Promise.resolve(opts.apply(entry.chunkIndex, entry.res))
            }
          } finally {
            bumpBusy(-1)
          }
        }
      })
      .catch((err) => {
        console.warn('[PodLens live] drain error', err)
      })
  }

  /** Transcribe in the background, then buffer the result and try to drain. */
  const submit = (blob: Blob, chunkIndex: number) => {
    const myGen = gen
    void (async () => {
      await sem.acquire()
      bumpBusy(1)
      try {
        const res = await opts.transcribe(blob, chunkIndex)
        if (myGen !== gen) return
        buffer.set(chunkIndex, { kind: 'ok', chunkIndex, res })
      } catch (e) {
        if (myGen !== gen) return
        buffer.set(chunkIndex, { kind: 'err', chunkIndex, error: e })
      } finally {
        bumpBusy(-1)
        sem.release()
      }
      if (myGen !== gen) return
      enqueueDrain()
    })()
  }

  /** Start a new session: invalidate in-flight work and clear the buffer. */
  const reset = () => {
    gen++
    buffer.clear()
    nextApply = 0
    busyDepth = 0
    drainTail = Promise.resolve()
    opts.setBusy(false)
  }

  return { submit, reset }
}
