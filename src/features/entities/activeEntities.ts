import type { EntityRecord } from '../../types/entities'

/** Minimum on-screen duration for a mention whose end time is missing or not after its start. */
const MIN_SPAN_SEC = 0.12

/** End time of a mention, padded to `MIN_SPAN_SEC` when the data has no usable end. */
function normalizedEnd(e: EntityRecord): number {
  return Number.isFinite(e.end_sec) && e.end_sec > e.start_sec ? e.end_sec : e.start_sec + MIN_SPAN_SEC
}

/**
 * Entities being spoken at `currentTimeSec` (half-open `[start, end)`), one per unique entity.
 * Overlapping mentions are ordered shortest span first (usually the most specific).
 */
export function getEntitiesActiveAtPlayback(entities: EntityRecord[], currentTimeSec: number): EntityRecord[] {
  if (!entities.length) return []
  const t = Math.max(0, currentTimeSec)

  const active = entities
    .filter((e) => t >= e.start_sec && t < normalizedEnd(e))
    .sort((a, b) => {
      const spanDiff = normalizedEnd(a) - a.start_sec - (normalizedEnd(b) - b.start_sec)
      return Math.abs(spanDiff) > 1e-6 ? spanDiff : a.text.localeCompare(b.text)
    })

  const seen = new Set<string>()
  return active.filter((e) => {
    const key = `${e.type}\0${e.text.toLowerCase()}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}
