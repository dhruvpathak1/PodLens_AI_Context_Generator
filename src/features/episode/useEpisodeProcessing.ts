import { useCallback, useState } from 'react'
import { enrichEntityCards } from '../../api/entities'
import { transcribeAudio } from '../../api/transcribe'
import type { EnrichedEntityCard, EntityDocument, EntityReviewReport } from '../../types/entities'
import type { TranscriptSegment } from '../../types/transcript'

/** Which stage of the upload pipeline is running. */
export type JobState = 'idle' | 'transcribing' | 'enriching'

/** Results computed elsewhere (demo mode): loaded straight into state, no API calls. */
export type PreparedEpisode = {
  name: string
  transcript: string
  segments: TranscriptSegment[]
  document: EntityDocument
  cards: EnrichedEntityCard[]
  unsplashEnabled: boolean
  review?: EntityReviewReport | null
}

type Options = {
  /** NER backend override forwarded to the server. */
  backend?: 'spacy' | 'claude'
}

/**
 * The core PodLens pipeline for one uploaded file:
 *   1. `transcribing`: upload to `/api/transcribe` (Whisper + entity tagging on the server)
 *   2. `enriching`:    send the tagged entities (with the transcript, for the AI review) to
 *                      `/api/enrich-entities` for source cards
 *
 * Owns the selected file and every piece of state those two requests produce.
 * `loadPrepared()` fills the same state from pre-computed results (demo mode).
 */
export function useEpisodeProcessing({ backend }: Options = {}) {
  const [file, setFile] = useState<File | null>(null)
  /** Name of a pre-computed episode loaded with `loadPrepared` (demo mode). */
  const [preparedName, setPreparedName] = useState<string | null>(null)
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
  /** What the AI entity review changed, if it ran. */
  const [review, setReview] = useState<EntityReviewReport | null>(null)

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
    setReview(null)
  }, [])

  /** Show pre-computed results (demo mode) exactly as if the pipeline had just produced them. */
  const loadPrepared = useCallback(
    (prepared: PreparedEpisode) => {
      clearResults()
      setFile(null)
      setPreparedName(prepared.name)
      setTranscript(prepared.transcript)
      setSegments(prepared.segments)
      setEntityDoc(prepared.document)
      setEnrichedCards(prepared.cards)
      setUnsplashHint(prepared.unsplashEnabled)
      setReview(prepared.review ?? null)
    },
    [clearResults]
  )

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
        const enriched = await enrichEntityCards(entities, { segments: result.segments, sourceLabel: file.name })
        // The AI review may have corrected, merged or dropped entities: show the reviewed list.
        if (enriched.entities) {
          const reviewed = enriched.entities
          setEntityDoc((doc) => (doc ? { ...doc, entities: reviewed } : doc))
        }
        setReview(enriched.review ?? null)
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
    /** Display name of the current episode: the uploaded file or the loaded demo. */
    sourceName: file?.name ?? preparedName,
    loadPrepared,
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
    review,
    run,
  }
}

export type EpisodeState = ReturnType<typeof useEpisodeProcessing>
