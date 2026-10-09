import { useMemo } from 'react'
import { formatClock } from '../../shared/format'
import type { EpisodeTimelineData, TimelineEvent } from '../../types/timeline'
import { compactWhen, formatEventDate, sortByHistory } from './timelineSort'

type Props = {
  timeline: EpisodeTimelineData
  /** Jump the audio player to a mention. */
  onSeek: (sec: number) => void
}

/** Stable key for "same date" so repeated dates are shown only once on the spine. */
const whenKey = (e: TimelineEvent) => (e.year == null ? 'undated' : `${e.year}-${e.month ?? ''}`)

/**
 * The whole timeline at once, in real-world order: a quiet date spine (date on the left,
 * events on the right). Undated events are listed last. Each event can play its first mention.
 */
export function EpisodeTimeline({ timeline, onSeek }: Props) {
  const events = useMemo(() => sortByHistory(timeline.events), [timeline.events])

  if (events.length === 0) return <p className="empty">The model found nothing with a date in this episode.</p>

  return (
    <ol className="spine">
      {events.map((e, i) => {
        const showWhen = i === 0 || whenKey(events[i - 1]) !== whenKey(e)
        const when = compactWhen(e)
        const label = formatEventDate(e)
        // Only repeat the model's wording when it says more than the compact date ("late 1700s").
        const showLabel = label && when && label !== when.value && label !== `${when.unit} ${when.value}`
        return (
          <li key={e.id} className={`spine__item${showWhen ? ' spine__item--dated' : ''}`}>
            <div className="spine__when" aria-hidden={!showWhen}>
              {showWhen ? (
                when ? (
                  <>
                    <span className="spine__value num">{when.value}</span>
                    {when.unit ? <span className="spine__unit">{when.unit}</span> : null}
                  </>
                ) : (
                  <span className="spine__unit">Undated</span>
                )
              ) : null}
            </div>
            <div className="spine__what">
              <h3 className="spine__title">{e.title}</h3>
              {e.description ? <p className="spine__desc">{e.description}</p> : null}
              <p className="spine__meta">
                {showLabel ? <span>{label}</span> : null}
                <button
                  type="button"
                  className="spine__play"
                  onClick={() => onSeek(e.mentioned_at_sec)}
                  aria-label={`Play mention of ${e.title} at ${formatClock(e.mentioned_at_sec)}`}
                >
                  Play <span className="num">{formatClock(e.mentioned_at_sec)}</span>
                </button>
              </p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
