import { useCallback, useState } from 'react'
import { enrichEntityCards } from '../../api/entities'
import { transcribeAudio } from '../../api/transcribe'
import type { EnrichedEntityCard, EntityDocument } from '../../types/entities'
import type { TranscriptSegment } from '../../types/transcript'

/** Which stage of the upload pipeline is running. */
export type JobState = 'idle' | 'transcribing' | 'enriching'

type Options = {
  /** NER backend override forwarded to the server. */
  backend?: 'spacy' | 'claude'
}

/**
 * The core PodLens pipeline for one uploaded file:
 *   1. `transcribing`: upload to `/api/transcribe` (Whisper + entity tagging on the server)
 *   2. `enriching`:    send the tagged entities to `/api/enrich-entities` for source cards
 *
 * Owns the selected file and every piece of state those two requests produce.
 */
export function useEpisodeProcessing({ backend }: Options = {}) {
  const [file, setFile] = useState<File | null>(null)
  const [job, setJob] = useState<JobState>('idle')
  const [error, setError] = useState<string | null>(null)

  const [transcript, setTranscript] = useState<string | null>(null)
  const [segments, setSegments] = useState<TranscriptSegment[]>([])
  const [savedPath, setSavedPath] = useState<string | null>(null)

  const [entityDoc, setEntityDoc] = useState<EntityDocument | null>(null)
  const [entitySavedPath, setEntitySavedPath] = useState<string | null>(null)
  const [entityError, setEntityError] = useState<string | null>(null)

  const [enrichedCards, setEnrichedCards] = useState<EnrichedEntityCard[]>([])
  const [enrichError, setEnrichError] = useState<string | null>(null)
  /** `false` when the server reports Unsplash is not configured (shows a setup hint). */
  const [unsplashHint, setUnsplashHint] = useState<boolean | null>(null)

  const busy = job !== 'idle'

  /** Clear every result from a previous run. */
  const clearResults = useCallback(() => {
    setError(null)
    setTranscript(null)
    setSegments([])
    setSavedPath(null)
    setEntityDoc(null)
    setEntitySavedPath(null)
    setEntityError(null)
    setEnrichedCards([])
    setEnrichError(null)
    setUnsplashHint(null)
  }, [])

  /** Run transcription, then enrichment, for the selected file. No-op while already busy. */
  const run = useCallback(async () => {
    if (!file || busy) return
    clearResults()
    setJob('transcribing')
    try {
      const result = await transcribeAudio(file, { backend })
      setTranscript(result.transcript)
      setSegments(result.segments)
      setSavedPath(result.savedPath ?? null)
      setEntityDoc(result.document ?? null)
      setEntitySavedPath(result.entitySavedPath ?? null)
      setEntityError(result.entityError ?? null)

      const entities = result.document?.entities ?? []
      if (!entities.length) return

      setJob('enriching')
      try {
        const enriched = await enrichEntityCards(entities)
        setEnrichedCards(enriched.cards)
        setUnsplashHint(enriched.unsplash_enabled ?? null)
      } catch (e) {
        // Enrichment is optional: keep the transcript and entities, just report the failure.
        setEnrichError(e instanceof Error ? e.message : 'Enrichment failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Transcription failed')
    } finally {
      setJob('idle')
    }
  }, [file, busy, backend, clearResults])

  return {
    file,
    selectFile: setFile,
    job,
    busy,
    error,
    transcript,
    segments,
    savedPath,
    entityDoc,
    entitySavedPath,
    entityError,
    enrichedCards,
    enrichError,
    unsplashHint,
    run,
  }
}

export type EpisodeState = ReturnType<typeof useEpisodeProcessing>
