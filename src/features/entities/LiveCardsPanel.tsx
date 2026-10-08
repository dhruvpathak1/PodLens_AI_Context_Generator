import { formatTimestamp } from '../../shared/format'
import type { EnrichedEntityCard } from '../../types/entities'
import { EntitySourceCard } from './EntitySourceCard'
import { entityMatchKey } from './entityKeys'
import { LIVE_QUEUE_MAX } from './useLiveRollingCards'

type Props = {
  /** Cards currently in the rolling queue (newest last). */
  cards: EnrichedEntityCard[]
  /** Whether any mentions match the current entity filter. */
  hasMatchingEntities: boolean
  hasAudio: boolean
  playbackTime: number
  playbackDuration: number
}

/** "Live" panel: full source cards for whatever is being mentioned at the current playback time. */
export function LiveCardsPanel({ cards, hasMatchingEntities, hasAudio, playbackTime, playbackDuration }: Props) {
  return (
    <section className="live-workspace" aria-labelledby="live-workspace-heading">
      <div className="live-workspace__head">
        <h2 id="live-workspace-heading" className="live-workspace__title">
          <span className="live-workspace__pulse" aria-hidden />
          Live
        </h2>
        {hasAudio ? (
          <time className="live-workspace__clock" dateTime={`PT${Math.floor(playbackTime)}S`}>
            {formatTimestamp(playbackTime)}
            {playbackDuration > 0 ? ` / ${formatTimestamp(playbackDuration)}` : ''}
          </time>
        ) : null}
      </div>
      <div className="live-workspace__body" aria-live="polite">
        {!hasAudio ? (
          <p className="live-workspace__hint">
            Load audio and press play. Up to three source cards queue here as each new mention <strong>starts</strong>;
            the oldest drops off when a fourth begins.
          </p>
        ) : !hasMatchingEntities ? (
          <p className="live-workspace__hint">
            No tags match the current <strong>type filter</strong> for live follow-along.
          </p>
        ) : cards.length === 0 ? (
          <p className="live-workspace__hint">
            Play the audio. Full cards appear when a tagged mention begins. Up to {LIVE_QUEUE_MAX} stay on screen; older
            ones roll off automatically.
          </p>
        ) : (
          <div className="live-workspace__cards" aria-label="Live rolling source cards">
            {cards.map((c) => (
              <div key={entityMatchKey(c.type, c.text)} className="live-workspace__card-wrap">
                <EntitySourceCard card={c} defaultExpanded />
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
