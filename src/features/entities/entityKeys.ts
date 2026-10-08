/** Identity helpers: the same entity can appear many times, but gets one card. */

import type { EnrichedEntityCard, EntityRecord } from '../../types/entities'

/** Stable key for an entity: type + case-insensitive text. */
export function entityMatchKey(type: string, text: string): string {
  return `${type}\0${text.trim().toLowerCase()}`
}

/** Find the enriched card for an entity mention, if enrichment has produced one. */
export function findCardForEntity(
  cards: EnrichedEntityCard[],
  e: { type: string; text: string }
): EnrichedEntityCard | undefined {
  const key = entityMatchKey(e.type, e.text)
  return cards.find((c) => entityMatchKey(c.type, c.text) === key)
}

/** Placeholder card shown until enrichment finishes for this mention. */
export function placeholderCard(e: EntityRecord): EnrichedEntityCard {
  return {
    id: `pending-${entityMatchKey(e.type, e.text)}`,
    type: e.type,
    text: e.text,
    start_sec: e.start_sec,
    end_sec: e.end_sec,
    chunk_id: e.chunk_id,
    wikipedia: null,
    location: null,
    unsplash: null,
  }
}
