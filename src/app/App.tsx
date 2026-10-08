import { useCallback, useEffect, useMemo } from 'react'
import { APP_TITLE, ENTITY_BACKEND, MISSING_PROD_API_URL } from '../config/env'
import { suggestQuestions } from '../features/ask/suggestions'
import { useEpisodeChat } from '../features/ask/useEpisodeChat'
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
import { DeployHint } from './DeployHint'

/**
 * Root component. Owns no UI logic of its own: it wires the feature hooks together and lays
 * out the three columns (transcript sidebar | main stage | Episode AI rail).
 *
 * Data flow: upload -> useEpisodeProcessing (transcript, entities, cards) -> playback time drives
 * the transcript highlight, Live cards and timeline reveal -> Timeline/Ask call the LLM endpoints.
 */
export default function App() {
  useEffect(() => {
    document.title = APP_TITLE
  }, [])

  // --- Core pipeline and playback -------------------------------------------------------
  const episode = useEpisodeProcessing({ backend: ENTITY_BACKEND })
  const playback = useAudioPlayback(episode.file)
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
  const episodeReady = !!episode.file && episode.segments.length > 0 && episode.entityDoc != null && !episode.busy
  const timeline = useEpisodeTimeline({
    ready: episodeReady,
    segments: episode.segments,
    entities,
    sourceLabel: episode.file?.name ?? null,
  })
  const chat = useEpisodeChat(() => ({ segments: episode.segments, entities, sourceLabel: episode.file?.name ?? null }))
  const askSuggestions = useMemo(() => suggestQuestions(episode.entityDoc?.entities ?? []), [episode.entityDoc])
  const showRail = episodeReady || timeline.timeline != null || timeline.busy || timeline.error != null

  // --- User actions -----------------------------------------------------------------------
  const { reset: resetTimeline } = timeline
  const { reset: resetChat } = chat
  const { resetFilter } = entityFilter
  const { clearSelection } = transcriptView
  const { selectFile, run } = episode

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

  return (
    <div className="dashboard">
      {MISSING_PROD_API_URL && <DeployHint />}
      <div className={`dashboard__grid${showRail ? ' dashboard__grid--rail' : ''}`}>
        <TranscriptSidebar
          episode={episode}
          view={transcriptView}
          playerRef={playback.playerRef}
          audioUrl={playback.audioUrl}
          onPlaybackTick={playback.handlePlaybackTick}
          onFileChange={handleFileChange}
          onRun={handleRun}
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
                unsplashHint={episode.unsplashHint}
                hasCards={episode.enrichedCards.length > 0}
              />
            )}
            {episode.enrichedCards.length > 0 && <SourceCardsSection cards={entityFilter.visibleCards} />}
          </div>
        </main>

        {showRail && (
          <EpisodeRail
            fileName={episode.file?.name ?? null}
            timeline={timeline}
            timelineReady={episodeReady}
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
