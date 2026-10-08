import { useMemo, useState } from 'react'
import { formatTimestamp } from '../../shared/format'
import type { EpisodeTimelineData, TimelineCategory, TimelineEvent } from '../../types/timeline'
import { formatEventDate, sortByEpisode, sortByHistory } from './timelineSort'

/** `stack` = newest mention on top; `date` = real-world order grouped by year. */
type View = 'stack' | 'date'

type Props = {
  timeline: EpisodeTimelineData
  /** Current playback position (seconds). Events are revealed once playback reaches their mention. */
  currentTime: number
  /** Jump the audio player to a mention. */
  onSeek: (sec: number) => void
}

/** Display order of the category filter chips. */
const CATEGORY_ORDER: TimelineCategory[] = [
  'COMPANY',
  'PERSON',
  'PRODUCT',
  'TECHNOLOGY',
  'DEAL',
  'EVENT',
  'PLACE',
  'OTHER',
]

/** Small lead so an event lands as the words are spoken, not a beat after. */
const REVEAL_LEAD_SEC = 0.5

/** A run of events sharing a year (or the "Undated" bucket) in the By date view. */
type Group = { key: string; label: string; events: TimelineEvent[] }

/** Sort events historically and group consecutive events by year. */
function groupByYear(events: TimelineEvent[]): Group[] {
  const groups: Group[] = []
  for (const e of sortByHistory(events)) {
    const key = e.year == null ? 'undated' : String(e.year)
    const last = groups[groups.length - 1]
    if (last && last.key === key) last.events.push(e)
    else groups.push({ key, label: e.year == null ? 'Undated' : String(e.year), events: [e] })
  }
  return groups
}

/**
 * Right-rail timeline. The full timeline is generated once, then revealed progressively:
 * each event is pushed onto the top of the stack when playback reaches the moment it is mentioned.
 */
export function EpisodeTimeline({ timeline, currentTime, onSeek }: Props) {
  const [view, setView] = useState<View>('stack')
  const [showAll, setShowAll] = useState(false)
  const [category, setCategory] = useState<TimelineCategory | 'ALL'>('ALL')

  // Every event, in order of first mention; the stack reveals a prefix of this list.
  const byMention = useMemo(() => sortByEpisode(timeline.events), [timeline.events])

  const revealed = useMemo(
    () => (showAll ? byMention : byMention.filter((e) => e.mentioned_at_sec <= currentTime + REVEAL_LEAD_SEC)),
    [byMention, showAll, currentTime]
  )

  // Per-category counts of revealed events, for the filter chips.
  const counts = useMemo(() => {
    const m = new Map<TimelineCategory, number>()
    for (const e of revealed) m.set(e.category, (m.get(e.category) ?? 0) + 1)
    return m
  }, [revealed])

  const visible = useMemo(
    () => (category === 'ALL' ? revealed : revealed.filter((e) => e.category === category)),
    [revealed, category]
  )

  // Newest revealed event (highlighted), next upcoming mention, and reveal progress.
  const latestId = revealed.length ? revealed[revealed.length - 1].id : null
  const next = showAll ? null : byMention.find((e) => e.mentioned_at_sec > currentTime + REVEAL_LEAD_SEC) ?? null
  const progress = timeline.events.length ? revealed.length / timeline.events.length : 0

  /** One event card on the vertical line. `showDate` hides the date when the group header already shows it. */
  const renderItem = (e: TimelineEvent, showDate: boolean) => {
    const date = formatEventDate(e)
    return (
      <li
        key={e.id}
        className={`tl-rail__item tl-rail__item--i${e.importance}${e.id === latestId && !showAll ? ' tl-rail__item--latest' : ''}`}
        data-cat={e.category}
      >
        <span className="tl-rail__dot" aria-hidden />
        <div className="tl-rail__card">
          <div className="tl-rail__row">
            <span className="tl-rail__cat">{e.category.toLowerCase()}</span>
            {date && showDate ? <span className="tl-rail__date">{date}</span> : null}
            <button
              type="button"
              className="tl-rail__seek"
              onClick={() => onSeek(e.mentioned_at_sec)}
              title="Play from this mention"
            >
              ▶ {formatTimestamp(e.mentioned_at_sec)}
            </button>
          </div>
          <h4 className="tl-rail__title">{e.title}</h4>
          <p className="tl-rail__desc">{e.description}</p>
          {e.entities.length > 0 ? (
            <ul className="tl-rail__entities" aria-label="Mentioned">
              {e.entities.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          ) : null}
        </div>
      </li>
    )
  }

  return (
    <section className="tl-rail" aria-labelledby="tl-rail-heading">
      <header className="tl-rail__head">
        <div className="tl-rail__title-row">
          <h2 id="tl-rail-heading" className="tl-rail__heading">
            <span className="tl-rail__pulse" aria-hidden />
            Mentioned so far
          </h2>
          <span className="tl-rail__count" aria-live="polite">
            {revealed.length}/{timeline.events.length}
          </span>
        </div>
        <div className="tl-rail__progress" aria-hidden>
          <span style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>

        <div className="tl-rail__controls">
          <div className="tl-rail__seg-group" role="group" aria-label="Timeline view">
            <button
              type="button"
              className={`tl-rail__seg${view === 'stack' ? ' tl-rail__seg--on' : ''}`}
              aria-pressed={view === 'stack'}
              onClick={() => setView('stack')}
            >
              As mentioned
            </button>
            <button
              type="button"
              className={`tl-rail__seg${view === 'date' ? ' tl-rail__seg--on' : ''}`}
              aria-pressed={view === 'date'}
              onClick={() => setView('date')}
            >
              By date
            </button>
          </div>
          <label className="tl-rail__toggle">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
            Show all
          </label>
        </div>

        {revealed.length > 0 ? (
          <div className="tl-rail__chips" role="group" aria-label="Filter by category">
            <button
              type="button"
              className={`tl-rail__chip${category === 'ALL' ? ' tl-rail__chip--on' : ''}`}
              aria-pressed={category === 'ALL'}
              onClick={() => setCategory('ALL')}
            >
              All {revealed.length}
            </button>
            {CATEGORY_ORDER.filter((c) => counts.get(c)).map((c) => (
              <button
                key={c}
                type="button"
                data-cat={c}
                className={`tl-rail__chip${category === c ? ' tl-rail__chip--on' : ''}`}
                aria-pressed={category === c}
                onClick={() => setCategory(c)}
              >
                {c.toLowerCase()} {counts.get(c)}
              </button>
            ))}
          </div>
        ) : null}
      </header>

      <div className="tl-rail__body">
        {revealed.length === 0 ? (
          <div className="tl-rail__empty">
            <p>Press play. Events stack up here the moment they are mentioned.</p>
            {next ? (
              <button type="button" className="tl-rail__next" onClick={() => onSeek(next.mentioned_at_sec)}>
                First mention at {formatTimestamp(next.mentioned_at_sec)}
              </button>
            ) : null}
          </div>
        ) : visible.length === 0 ? (
          <p className="tl-rail__empty">No {category.toLowerCase()} mentions yet.</p>
        ) : view === 'stack' ? (
          <ol className="tl-rail__list">{[...visible].reverse().map((e) => renderItem(e, true))}</ol>
        ) : (
          groupByYear(visible).map((g) => (
            <div key={g.key} className="tl-rail__group">
              <h3 className="tl-rail__group-label">{g.label}</h3>
              <ol className="tl-rail__list">
                {g.events.map((e) => renderItem(e, g.key === 'undated' || formatEventDate(e) !== g.label))}
              </ol>
            </div>
          ))
        )}
      </div>

      {next && revealed.length > 0 ? (
        <footer className="tl-rail__foot">
          {timeline.events.length - revealed.length} more ahead · next at {formatTimestamp(next.mentioned_at_sec)}
        </footer>
      ) : null}
    </section>
  )
}
