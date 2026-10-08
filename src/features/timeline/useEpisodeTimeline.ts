import { useCallback, useRef, useState } from 'react'
import { createEpisodeTimeline } from '../../api/timeline'
import type { EntityRecord } from '../../types/entities'
import type { EpisodeTimelineData } from '../../types/timeline'
import type { TranscriptSegment } from '../../types/transcript'

type Params = {
  /** True once transcription and entity tagging have finished. */
  ready: boolean
  segments: TranscriptSegment[]
  entities: EntityRecord[]
  sourceLabel: string | null
}

/**
 * Timeline generation state. `create()` sends the transcript and entities to the LLM;
 * `reset()` aborts any request in flight and clears the result (new episode).
 */
export function useEpisodeTimeline({ ready, segments, entities, sourceLabel }: Params) {
  const [timeline, setTimeline] = useState<EpisodeTimelineData | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setTimeline(null)
    setBusy(false)
    setError(null)
  }, [])

  const create = useCallback(async () => {
    if (!ready || busy) return
    const ctrl = new AbortController()
    abortRef.current?.abort()
    abortRef.current = ctrl
    setBusy(true)
    setError(null)
    try {
      const result = await createEpisodeTimeline(segments, entities, sourceLabel, ctrl.signal)
      if (!ctrl.signal.aborted) setTimeline(result)
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : 'Timeline failed')
    } finally {
      // Only the latest request may clear the busy flag.
      if (abortRef.current === ctrl) {
        abortRef.current = null
        setBusy(false)
      }
    }
  }, [ready, busy, segments, entities, sourceLabel])

  return { timeline, busy, error, create, reset }
}

export type EpisodeTimelineState = ReturnType<typeof useEpisodeTimeline>
