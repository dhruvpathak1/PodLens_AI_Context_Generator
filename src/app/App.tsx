import { useCallback, useEffect, useMemo, useState } from 'react'
import { APP_TITLE, DEMO_MODE, ENTITY_BACKEND, MISSING_PROD_API_URL } from '../config/env'
import { suggestQuestions } from '../features/ask/suggestions'
import { useEpisodeChat, type AnswerFn } from '../features/ask/useEpisodeChat'
import { DemoBanner } from '../features/demo/DemoBanner'
import { demoAnswer, demoAudioUrl, demoTimeline } from '../features/demo/demoData'
import { DemoPicker } from '../features/demo/DemoPicker'
import { useDemoCatalog } from '../features/demo/useDemoCatalog'
import { EntityFilterBar } from '../features/entities/EntityFilterBar'
import { LiveCardsPanel } from '../features/entities/LiveCardsPanel'
import { SourceCardsSection } from '../features/entities/SourceCardsSection'
import { useEntityFilter } from '../features/entities/useEntityFilter'
import { useLiveRollingCards } from '../features/entities/useLiveRollingCards'
import { useEpisodeProcessing } from '../features/episode/useEpisodeProcessing'
import { useAudioPlayback } from '../features/playback/useAudioPlayback'
import { EpisodeRail } from '../features/rail/EpisodeRail'
import { useEpisodeTimeline } from '../features/timeline/useEpisodeTimeline'
import { TranscriptSidebar } from '../features/transcript/TranscriptSidebar'
import { useTranscriptView } from '../features/transcript/useTranscriptView'
import type { DemoEpisode } from '../types/demo'
import { DeployHint } from './DeployHint'

/**
 * Root component. Owns no UI logic of its own: it wires the feature hooks together and lays
 * out the three columns (transcript sidebar | main stage | Episode AI rail).
 *
 * Data flow: upload -> useEpisodeProcessing (transcript, entities, cards) -> playback time drives
 * the transcript highlight, Live cards and timeline reveal -> Timeline/Ask call the LLM endpoints.
 *
 * Demo mode (`VITE_DEMO_MODE=true`, the public site): a picker loads pre-processed episodes from
 * `public/demo/` into the same state, and Timeline/Ask answer from the stored results. The rest
 * of the UI is identical and never calls the API.
 */
export default function App() {
  useEffect(() => {
    document.title = APP_TITLE
  }, [])

  // --- Core pipeline and playback -------------------------------------------------------
  const episode = useEpisodeProcessing({ backend: ENTITY_BACKEND })
  /** The loaded demo episode (demo mode only). */
  const [demoEpisode, setDemoEpisode] = useState<DemoEpisode | null>(null)
  const playback = useAudioPlayback(demoEpisode ? demoAudioUrl(demoEpisode) : episode.file)
  const entities = episode.entityDoc?.entities ?? []

  // --- Views over the results -----------------------------------------------------------
  const transcriptView = useTranscriptView({
    transcript: episode.transcript,
    segments: episode.segments,
    playbackTime: playback.playbackTime,
    playbackDuration: playback.playbackDuration,
    audioUrl: playback.audioUrl,
    seekTo: playback.seekTo,
  })
  const entityFilter = useEntityFilter(episode.entityDoc, episode.enrichedCards)
  const liveCards = useLiveRollingCards({
    entities: entityFilter.filteredEntities,
    playbackTime: playback.playbackTime,
    cards: episode.enrichedCards,
    enabled: !!playback.audioUrl,
    resetKey: `${playback.audioUrl}|${entityFilter.filter}|${episode.entityDoc?.extracted_at ?? ''}`,
  })

  // --- AI features (available once transcription and tagging finished) ----------------------
  const episodeReady = !!episode.sourceName && episode.segments.length > 0 && episode.entityDoc != null && !episode.busy
  // A demo built without an OpenAI key has no timeline to show.
  const timelineReady = episodeReady && (!DEMO_MODE || demoEpisode?.timeline != null)
  // In demo mode the LLM calls are replaced by the pre-computed results.
  const demoGenerate = useMemo(() => (demoEpisode ? () => demoTimeline(demoEpisode) : undefined), [demoEpisode])
  const demoAnswerFn = useMemo<AnswerFn | undefined>(
    () => (demoEpisode ? (question) => demoAnswer(demoEpisode, question) : undefined),
    [demoEpisode]
  )
  const timeline = useEpisodeTimeline({
    ready: timelineReady,
    segments: episode.segments,
    entities,
    sourceLabel: episode.sourceName,
    generate: demoGenerate,
  })
  const chat = useEpisodeChat(
    () => ({ segments: episode.segments, entities, sourceLabel: episode.sourceName }),
    demoAnswerFn
  )
  // Demo suggestions are exactly the questions that have stored answers.
  const askSuggestions = useMemo(
    () =>
      demoEpisode?.qa.length
        ? demoEpisode.qa.map((qa) => qa.question)
        : suggestQuestions(episode.entityDoc?.entities ?? []),
    [demoEpisode, episode.entityDoc]
  )
  const showRail = episodeReady || timeline.timeline != null || timeline.busy || timeline.error != null

  // --- User actions -----------------------------------------------------------------------
  const { reset: resetTimeline } = timeline
  const { reset: resetChat } = chat
  const { resetFilter } = entityFilter
  const { clearSelection } = transcriptView
  const { selectFile, run, loadPrepared } = episode

  /** A different file invalidates the timeline and conversation. */
  const handleFileChange = useCallback(
    (file: File | null) => {
      if (file) {
        resetTimeline()
        resetChat()
      }
      selectFile(file)
    },
    [resetTimeline, resetChat, selectFile]
  )

  /** Run PodLens: clear per-episode views, then transcribe -> tag -> enrich. */
  const handleRun = useCallback(() => {
    resetTimeline()
    resetChat()
    resetFilter()
    clearSelection()
    void run()
  }, [resetTimeline, resetChat, resetFilter, clearSelection, run])

  /** Demo episode picked: clear per-episode views, then show its pre-computed results. */
  const handleDemoLoaded = useCallback(
    (demo: DemoEpisode) => {
      resetTimeline()
      resetChat()
      resetFilter()
      clearSelection()
      setDemoEpisode(demo)
      loadPrepared({
        name: demo.title,
        transcript: demo.transcript,
        segments: demo.segments,
        document: demo.document,
        cards: demo.cards,
        unsplashEnabled: demo.unsplash_enabled,
      })
    },
    [resetTimeline, resetChat, resetFilter, clearSelection, loadPrepared]
  )
  const demoCatalog = useDemoCatalog(DEMO_MODE, handleDemoLoaded)

  return (
    <div className="dashboard">
      {MISSING_PROD_API_URL && <DeployHint />}
      {DEMO_MODE && <DemoBanner credit={demoEpisode?.credit || null} />}
      <div className={`dashboard__grid${showRail ? ' dashboard__grid--rail' : ''}`}>
        <TranscriptSidebar
          episode={episode}
          view={transcriptView}
          playerRef={playback.playerRef}
          audioUrl={playback.audioUrl}
          onPlaybackTick={playback.handlePlaybackTick}
          onFileChange={handleFileChange}
          onRun={handleRun}
          ingest={DEMO_MODE ? <DemoPicker catalog={demoCatalog} /> : undefined}
        />

        <main className="main-stage">
          <div className="main-stage__canvas" role="region" aria-label="Workspace, entities, and source cards">
            {entities.length > 0 && (
              <LiveCardsPanel
                cards={liveCards}
                hasMatchingEntities={entityFilter.filteredEntities.length > 0}
                hasAudio={!!playback.audioUrl}
                playbackTime={playback.playbackTime}
                playbackDuration={playback.playbackDuration}
              />
            )}
            {episode.entityDoc && (
              <EntityFilterBar
                entityDoc={episode.entityDoc}
                filter={entityFilter}
                entitySavedPath={episode.entitySavedPath}
                // The "add an Unsplash key" hint is server setup advice; meaningless on the public demo.
                unsplashHint={DEMO_MODE ? null : episode.unsplashHint}
                hasCards={episode.enrichedCards.length > 0}
              />
            )}
            {episode.enrichedCards.length > 0 && <SourceCardsSection cards={entityFilter.visibleCards} />}
          </div>
        </main>

        {showRail && (
          <EpisodeRail
            fileName={episode.sourceName}
            timeline={timeline}
            timelineReady={timelineReady}
            chat={chat}
            askSuggestions={askSuggestions}
            playbackTime={playback.playbackTime}
            seekTo={playback.seekTo}
            onCite={transcriptView.jumpToCitation}
          />
        )}
      </div>
    </div>
  )
}
