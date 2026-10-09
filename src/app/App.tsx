import { useCallback, useEffect, useMemo, useState } from 'react'
import { APP_TITLE, DEMO_MODE, ENTITY_BACKEND, MISSING_PROD_API_URL } from '../config/env'
import { useEpisodeChat, type AnswerFn } from '../features/ask/useEpisodeChat'
import { AboutPanel } from '../features/about/AboutPanel'
import { AskEpisode } from '../features/ask/AskEpisode'
import { demoAnswer, demoAudioUrl, demoTimeline } from '../features/demo/demoData'
import { DemoPicker } from '../features/demo/DemoPicker'
import { useDemoCatalog } from '../features/demo/useDemoCatalog'
import { EntitiesPanel } from '../features/entities/EntitiesPanel'
import { LiveGrid } from '../features/entities/LiveGrid'
import { useEntityFilter } from '../features/entities/useEntityFilter'
import { useLiveRollingCards } from '../features/entities/useLiveRollingCards'
import { useEpisodeProcessing } from '../features/episode/useEpisodeProcessing'
import { useAudioPlayback } from '../features/playback/useAudioPlayback'
import { TimelinePanel } from '../features/timeline/TimelinePanel'
import { useEpisodeTimeline } from '../features/timeline/useEpisodeTimeline'
import { TranscriptPanel } from '../features/transcript/TranscriptPanel'
import { useTranscriptView } from '../features/transcript/useTranscriptView'
import type { DemoEpisode } from '../types/demo'
import { AudioDropZone } from '../features/upload/AudioDropZone'
import { DeployHint } from './DeployHint'

/**
 * Root component. Owns no UI logic of its own: it wires the feature hooks together and lays
 * out the dashboard as three columns, each split vertically:
 *
 *   | Transcript (player + text) | Live cards (2 x 2)  | Timeline        |
 *   | PodLens (about + samples)  | Entities (NER)      | Ask the episode |
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
  const entityFilter = useEntityFilter(episode.entityDoc)
  const liveCards = useLiveRollingCards({
    entities: entityFilter.filteredEntities,
    playbackTime: playback.playbackTime,
    cards: episode.enrichedCards,
    enabled: !!playback.audioUrl,
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

  const hasEpisode = episode.segments.length > 0
  const timelineUnavailable = !hasEpisode
    ? 'Load an episode first.'
    : DEMO_MODE && !demoEpisode?.timeline
      ? 'This sample was published without a timeline.'
      : episode.busy
        ? 'Waiting for transcription to finish.'
        : undefined

  return (
    <div className="app">
      {MISSING_PROD_API_URL && <DeployHint />}
      <div className="board">
        <div className="board__col board__col--left">
          <TranscriptPanel
            episode={episode}
            view={transcriptView}
            playerRef={playback.playerRef}
            audioUrl={playback.audioUrl}
            onPlaybackTick={playback.handlePlaybackTick}
          />
          <AboutPanel
            note={
              DEMO_MODE ? (
                <>
                  {demoEpisode?.credit ? (
                    <>
                      {demoEpisode.credit}
                      <br />
                    </>
                  ) : null}
                  Samples were processed ahead of time. Run PodLens locally to analyse your own audio.
                </>
              ) : null
            }
          >
            {DEMO_MODE ? (
              <DemoPicker catalog={demoCatalog} />
            ) : (
              <>
                <AudioDropZone file={episode.file} onFileChange={handleFileChange} disabled={episode.busy} compact />
                <button
                  type="button"
                  className="btn btn--primary btn--block"
                  disabled={!episode.file || episode.busy}
                  onClick={handleRun}
                >
                  {episode.job === 'transcribing'
                    ? 'Transcribing and tagging…'
                    : episode.job === 'enriching'
                      ? 'Fetching sources…'
                      : 'Analyse episode'}
                </button>
              </>
            )}
          </AboutPanel>
        </div>

        <div className="board__col board__col--mid">
          <LiveGrid
            cards={liveCards}
            hasEntities={entityFilter.filteredEntities.length > 0}
            hasAudio={!!playback.audioUrl}
            playbackTime={playback.playbackTime}
            playbackDuration={playback.playbackDuration}
          />
          <EntitiesPanel
            entityDoc={episode.entityDoc}
            filter={entityFilter}
            playbackTime={playback.playbackTime}
            onSeek={playback.seekTo}
            // Server setup advice is meaningless on the public demo.
            note={
              !DEMO_MODE && episode.unsplashHint === false && episode.enrichedCards.length > 0
                ? 'Photos are off: add UNSPLASH_ACCESS_KEY to .env and restart the API.'
                : null
            }
          />
        </div>

        <div className="board__col board__col--right">
          <TimelinePanel
            timeline={timeline}
            ready={timelineReady}
            unavailableReason={timelineUnavailable}
            onSeek={playback.seekTo}
          />
          <AskEpisode
            messages={chat.messages}
            busy={chat.busy}
            enabled={episodeReady}
            onAsk={(q) => void chat.ask(q)}
            onStop={chat.stop}
            onClear={chat.reset}
            onCite={transcriptView.jumpToCitation}
          />
        </div>
      </div>
    </div>
  )
}
