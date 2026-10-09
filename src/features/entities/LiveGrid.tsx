import { Panel } from '../../shared/components/Panel'
import { formatClock } from '../../shared/format'
import type { EnrichedEntityCard } from '../../types/entities'
import { entityMatchKey } from './entityKeys'
import { IntroCards } from './IntroCards'
import { LiveCard } from './LiveCard'
import { LIVE_QUEUE_MAX } from './useLiveRollingCards'

type Props = {
  /** Cards currently in the rolling queue (oldest first). */
  cards: EnrichedEntityCard[]
  hasAudio: boolean
  playbackTime: number
  playbackDuration: number
}

/**
 * Middle column, top: a 2 x 2 grid of source cards for the latest mentions up to the playhead.
 * The newest card takes the first slot; older ones move along and drop off after four.
 * Before any mention (first load, or before playback reaches a name) it shows four intro cards.
 */
export function LiveGrid({ cards, hasAudio, playbackTime, playbackDuration }: Props) {
  const newestFirst = [...cards].reverse()
  const slots = Array.from({ length: LIVE_QUEUE_MAX }, (_, i) => newestFirst[i] ?? null)
  return (
    <Panel
      className="panel--live"
      title="Live cards"
      meta={hasAudio ? <span className="num">{formatClock(playbackTime)} / {formatClock(playbackDuration)}</span> : null}
      bodyClassName="panel__body--grid"
    >
      <div className="live-grid" aria-live="polite" aria-label="Source cards for the latest mentions">
        {/* Until the first name is mentioned, the grid introduces the project instead of sitting empty. */}
        {cards.length === 0 ? <IntroCards /> : slots.map((card, i) =>
          card ? (
            <LiveCard key={entityMatchKey(card.type, card.text)} card={card} />
          ) : (
            <div key={`empty-${i}`} className={`live-slot${i === cards.length ? ' live-slot--next' : ''}`}>
              {i === cards.length ? <p className="live-slot__text">The next name mentioned appears here.</p> : null}
            </div>
          )
        )}
      </div>
    </Panel>
  )
}
