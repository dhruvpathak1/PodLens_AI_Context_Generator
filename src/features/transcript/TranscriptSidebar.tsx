import type { ReactNode, RefObject } from 'react'
import { ErrorAlert } from '../../shared/components/Alert'
import { SearchIcon } from '../../shared/icons'
import type { EpisodeState } from '../episode/useEpisodeProcessing'
import { MiniAudioPlayer, type MiniAudioPlayerHandle } from '../playback/MiniAudioPlayer'
import { ThemeToggle } from '../theme/ThemeToggle'
import { AudioDropZone } from '../upload/AudioDropZone'
import { TranscriptFeed } from './TranscriptFeed'
import type { TranscriptView } from './useTranscriptView'

type Props = {
  episode: EpisodeState
  view: TranscriptView
  playerRef: RefObject<MiniAudioPlayerHandle | null>
  audioUrl: string | null
  onPlaybackTick: (time: number, duration: number) => void
  /** Called when the user picks or removes a file. */
  onFileChange: (file: File | null) => void
  /** Called by the PodLens button. */
  onRun: () => void
  /** Replaces the upload box and PodLens button (demo mode shows an episode picker here). */
  ingest?: ReactNode
}

/** Short status shown under the sidebar title. */
function statusLabel(episode: EpisodeState): string {
  if (episode.job === 'transcribing') return 'Transcribing…'
  if (episode.job === 'enriching') return 'Fetching sources…'
  return episode.transcript ? 'Synced' : 'Ready'
}

/** Label of the main PodLens button for each pipeline stage. */
function runButtonLabel(episode: EpisodeState): string {
  if (episode.job === 'transcribing') return 'Transcribing & tagging…'
  if (episode.job === 'enriching') return 'Fetching Wikipedia, maps & photos…'
  return 'PodLens'
}

/**
 * Left column: title/status, theme + search buttons, upload and PodLens button, audio player,
 * error messages, and the transcript feed.
 */
export function TranscriptSidebar({
  episode,
  view,
  playerRef,
  audioUrl,
  onPlaybackTick,
  onFileChange,
  onRun,
  ingest,
}: Props) {
  const { busy } = episode
  return (
    <aside className="transcript-sidebar" aria-label="Live transcript">
      <div className="transcript-sidebar__head">
        <div className="transcript-sidebar__titles">
          <h2 className="transcript-sidebar__label">Live transcript</h2>
          <p className={`transcript-sidebar__status${busy ? ' transcript-sidebar__status--busy' : ''}`}>
            <span className="transcript-sidebar__status-dot" aria-hidden />
            {statusLabel(episode)}
          </p>
        </div>
        <div className="transcript-sidebar__actions">
          <ThemeToggle />
          <button
            type="button"
            className="icon-btn"
            aria-expanded={view.searchOpen}
            aria-label={view.searchOpen ? 'Close search' : 'Search transcript'}
            onClick={view.toggleSearch}
          >
            <SearchIcon />
          </button>
        </div>
      </div>

      {view.searchOpen && (
        <label className="transcript-search">
          <span className="sr-only">Filter transcript</span>
          <input
            type="search"
            className="transcript-search__input"
            placeholder="Filter sentences…"
            value={view.searchQuery}
            onChange={(e) => view.setSearchQuery(e.target.value)}
            autoComplete="off"
          />
        </label>
      )}

      <div className="transcript-sidebar__ingest">
        {ingest ?? (
          <>
            <AudioDropZone file={episode.file} onFileChange={onFileChange} disabled={busy} compact />
            <button
              type="button"
              className="btn btn--primary btn--block btn--podlens"
              disabled={!episode.file || busy}
              onClick={onRun}
            >
              {runButtonLabel(episode)}
            </button>
          </>
        )}
      </div>

      <MiniAudioPlayer key={audioUrl ?? 'no-audio'} ref={playerRef} src={audioUrl} onPlaybackTick={onPlaybackTick} />

      <ErrorAlert>{episode.error}</ErrorAlert>
      <ErrorAlert>{episode.entityError}</ErrorAlert>
      <ErrorAlert>{episode.enrichError}</ErrorAlert>

      <TranscriptFeed
        view={view}
        busy={busy}
        hasTranscript={!!episode.transcript}
        hasError={!!episode.error}
      />

      {episode.savedPath && (
        <p className="transcript-sidebar__saved" title={episode.savedPath}>
          Saved: <code>{episode.savedPath.split('/').pop()}</code>
        </p>
      )}
    </aside>
  )
}
