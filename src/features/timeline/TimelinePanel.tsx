import { ErrorAlert } from '../../shared/components/Alert'
import { Panel } from '../../shared/components/Panel'
import { Skeleton } from '../../shared/components/Skeleton'
import { EpisodeTimeline } from './EpisodeTimeline'
import type { EpisodeTimelineState } from './useEpisodeTimeline'

type Props = {
  timeline: EpisodeTimelineState
  /** True when a timeline can be created for the current episode. */
  ready: boolean
  /** Shown instead of the button text when `ready` is false. */
  unavailableReason?: string
  onSeek: (sec: number) => void
}

/** Right column, top: "Create timeline", then every dated event in order. */
export function TimelinePanel({ timeline, ready, unavailableReason, onSeek }: Props) {
  const data = timeline.timeline
  return (
    <Panel
      className="panel--timeline"
      title="Timeline"
      meta={data ? `${data.events.length} events` : null}
      actions={
        data ? (
          <button type="button" className="link-quiet" onClick={() => void timeline.create()} disabled={timeline.busy || !ready}>
            {timeline.busy ? 'Rebuilding…' : 'Rebuild'}
          </button>
        ) : null
      }
    >
      {data ? (
        <>
          <ErrorAlert>{timeline.error ? `Rebuild failed: ${timeline.error}` : null}</ErrorAlert>
          <EpisodeTimeline timeline={data} onSeek={onSeek} />
        </>
      ) : timeline.busy ? (
        <Skeleton />
      ) : (
        <div className="cta">
          <p className="cta__text">
            Put every event mentioned in the episode in date order, from the earliest to the most recent.
          </p>
          <ErrorAlert>{timeline.error}</ErrorAlert>
          <button type="button" className="btn btn--primary" onClick={() => void timeline.create()} disabled={!ready}>
            Create timeline
          </button>
          {!ready && unavailableReason ? <p className="cta__hint">{unavailableReason}</p> : null}
        </div>
      )}
    </Panel>
  )
}
