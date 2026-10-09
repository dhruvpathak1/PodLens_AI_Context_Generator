import { useMemo } from 'react'
import type { EnrichedEntityCard, EntityRecord } from '../../types/entities'
import { entityMatchKey, findCardForEntity, placeholderCard } from './entityKeys'

/** Number of live card slots (a 2 x 2 grid). */
export const LIVE_QUEUE_MAX = 4

/** Show a card a moment before the word is spoken, so it lands with the speech. */
const LEAD_SEC = 0.3

type Params = {
  /** Mentions to follow (already filtered by the entity type filter). */
  entities: EntityRecord[]
  playbackTime: number
  /** Enriched cards; a placeholder is shown for any entity without one. */
  cards: EnrichedEntityCard[]
  /** Only follow playback when audio is loaded. */
  enabled: boolean
}

/**
 * Live cards: the `LIVE_QUEUE_MAX` most recently mentioned distinct entities up to the playhead,
 * oldest first. Derived purely from the playback time, so seeking anywhere (scrubber, transcript,
 * timeline, citations) immediately shows the right cards.
 */
export function useLiveRollingCards({ entities, playbackTime, cards, enabled }: Params): EnrichedEntityCard[] {
  // Mentions sorted by start time, computed once per entity list.
  const byStart = useMemo(() => [...entities].sort((a, b) => a.start_sec - b.start_sec), [entities])

  return useMemo(() => {
    if (!enabled) return []
    const t = playbackTime + LEAD_SEC
    const picked: EntityRecord[] = []
    const seen = new Set<string>()
    // Walk backwards from the latest mention that has started.
    for (let i = byStart.length - 1; i >= 0 && picked.length < LIVE_QUEUE_MAX; i--) {
      const e = byStart[i]
      if (e.start_sec > t) continue
      const key = entityMatchKey(e.type, e.text)
      if (seen.has(key)) continue
      seen.add(key)
      picked.push(e)
    }
    return picked.reverse().map((e) => findCardForEntity(cards, e) ?? placeholderCard(e))
  }, [byStart, playbackTime, cards, enabled])
}
