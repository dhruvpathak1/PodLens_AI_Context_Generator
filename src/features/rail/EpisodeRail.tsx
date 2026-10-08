import { useState } from 'react'
import { ErrorAlert } from '../../shared/components/Alert'
import { Skeleton } from '../../shared/components/Skeleton'
import { ChatIcon, ListIcon, SparkleIcon } from '../../shared/icons'
import { AskEpisode } from '../ask/AskEpisode'
import type { EpisodeChat } from '../ask/useEpisodeChat'
import { EpisodeTimeline } from '../timeline/EpisodeTimeline'
import type { EpisodeTimelineState } from '../timeline/useEpisodeTimeline'

type Tab = 'timeline' | 'ask'

type Props = {
  fileName: string | null
  timeline: EpisodeTimelineState
  /** True once the episode is transcribed and tagged (enables Create timeline). */
  timelineReady: boolean
  chat: EpisodeChat
  askSuggestions: string[]
  playbackTime: number
  seekTo: (sec: number) => void
  /** Ask citation click: seek and highlight the matching transcript sentence. */
  onCite: (sec: number, quote: string) => void
}

/** Right column ("Episode AI"): a raised panel with Timeline and Ask tabs. */
export function EpisodeRail({ fileName, timeline, timelineReady, chat, askSuggestions, playbackTime, seekTo, onCite }: Props) {
  const [tab, setTab] = useState<Tab>('timeline')
  const questionCount = chat.messages.filter((m) => m.role === 'user').length

  return (
    <aside className="timeline-col" aria-label="Timeline and questions">
      <div className="rail-head">
        <span className="rail-head__spark" aria-hidden>
          <SparkleIcon />
        </span>
        <div className="rail-head__text">
          <h2 className="rail-head__title">Episode AI</h2>
          <p className="rail-head__sub" title={fileName ?? undefined}>
            {fileName ?? 'Current episode'}
          </p>
        </div>
      </div>

      <div className="rail-tabs" role="tablist" aria-label="Episode tools">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'timeline'}
          className={`rail-tabs__tab${tab === 'timeline' ? ' rail-tabs__tab--on' : ''}`}
          onClick={() => setTab('timeline')}
        >
          <ListIcon />
          Timeline
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'ask'}
          className={`rail-tabs__tab${tab === 'ask' ? ' rail-tabs__tab--on' : ''}`}
          onClick={() => setTab('ask')}
        >
          <ChatIcon />
          Ask
          {questionCount > 0 ? <span className="rail-tabs__badge">{questionCount}</span> : null}
        </button>
        {tab === 'timeline' && timeline.timeline ? (
          <span className="rail-tabs__aside">
            <button
              type="button"
              className="rail-tabs__link"
              onClick={() => void timeline.create()}
              disabled={timeline.busy || !timelineReady}
            >
              {timeline.busy ? 'Rebuilding…' : 'Rebuild'}
            </button>
          </span>
        ) : null}
      </div>

      {tab === 'timeline' ? (
        <div className="rail-panel" role="tabpanel" aria-label="Timeline">
          <TimelineTab timeline={timeline} ready={timelineReady} playbackTime={playbackTime} seekTo={seekTo} />
        </div>
      ) : (
        <div className="rail-panel" role="tabpanel" aria-label="Ask the episode">
          <AskEpisode
            messages={chat.messages}
            busy={chat.busy}
            suggestions={askSuggestions}
            onAsk={(q) => void chat.ask(q)}
            onStop={chat.stop}
            onClear={chat.reset}
            onCite={onCite}
          />
        </div>
      )}
    </aside>
  )
}

type TimelineTabProps = {
  timeline: EpisodeTimelineState
  ready: boolean
  playbackTime: number
  seekTo: (sec: number) => void
}

/** Timeline tab body: the timeline, a loading skeleton, or the "Create timeline" call to action. */
function TimelineTab({ timeline, ready, playbackTime, seekTo }: TimelineTabProps) {
  if (timeline.timeline) {
    return (
      <>
        {timeline.error ? (
          <ErrorAlert className="timeline-col__state">Rebuild failed: {timeline.error}</ErrorAlert>
        ) : null}
        <EpisodeTimeline timeline={timeline.timeline} currentTime={playbackTime} onSeek={seekTo} />
      </>
    )
  }
  if (timeline.busy) {
    return (
      <div className="timeline-col__state">
        <Skeleton />
      </div>
    )
  }
  return (
    <div className="rail-cta">
      <h3 className="rail-cta__title">Episode timeline</h3>
      <p className="rail-cta__lead">
        Order every person, company and event mentioned, using the transcript and tagged entities. Events stack up
        here as playback reaches them.
      </p>
      <ErrorAlert>{timeline.error}</ErrorAlert>
      <button type="button" className="btn btn--primary btn--block" onClick={() => void timeline.create()} disabled={!ready}>
        Create timeline
      </button>
    </div>
  )
}
