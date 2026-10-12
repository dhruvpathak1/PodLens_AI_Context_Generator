import type { RefObject } from 'react'
import { ErrorAlert } from '../../shared/components/Alert'
import { Panel } from '../../shared/components/Panel'
import type { EpisodeState } from '../episode/useEpisodeProcessing'
import { MiniAudioPlayer, type MiniAudioPlayerHandle } from '../playback/MiniAudioPlayer'
import { TranscriptFeed } from './TranscriptFeed'
import type { TranscriptView } from './useTranscriptView'

type Props = {
  episode: EpisodeState
  view: TranscriptView
  playerRef: RefObject<MiniAudioPlayerHandle | null>
  audioUrl: string | null
  onPlaybackTick: (time: number, duration: number) => void
}

/** Short status shown next to the panel title. */
function statusLabel(episode: EpisodeState): string {
  if (episode.job === 'transcribing') return 'Transcribing…'
  if (episode.job === 'enriching') return 'Fetching sources…'
  return episode.transcript ? 'Synced to audio' : 'No episode loaded'
}

/**
 * Left column, top: audio player with waveform scrubber, then the transcript that follows
 * playback. Its search box sits in the Live cards header (`LiveGrid`).
 */
export function TranscriptPanel({ episode, view, playerRef, audioUrl, onPlaybackTick }: Props) {
  const { busy } = episode
  return (
    <Panel
      className="panel--transcript"
      title="Transcript"
      meta={
        <span className={`status${busy ? ' status--busy' : ''}${episode.transcript ? ' status--live' : ''}`}>
          <span className="status__dot" aria-hidden />
          {statusLabel(episode)}
        </span>
      }
      toolbar={
        <div className="transcript-toolbar">
          <MiniAudioPlayer key={audioUrl ?? 'no-audio'} ref={playerRef} src={audioUrl} onPlaybackTick={onPlaybackTick} />
          <ErrorAlert>{episode.error}</ErrorAlert>
          <ErrorAlert>{episode.entityError}</ErrorAlert>
          <ErrorAlert>{episode.enrichError}</ErrorAlert>
        </div>
      }
      bodyClassName="panel__body--flush"
    >
      <TranscriptFeed view={view} busy={busy} hasTranscript={!!episode.transcript} hasError={!!episode.error} />
    </Panel>
  )
}
