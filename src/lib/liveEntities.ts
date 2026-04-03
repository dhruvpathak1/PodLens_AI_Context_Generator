import type { EntityRecord } from './extractEntities'

/**
 * Entities whose time span contains `currentTimeSec` (half-open [start, end)).
 * Overlapping spans are sorted with shorter spans first (typically more specific).
 */
function normalizeEntityEnd(e: EntityRecord): number {
  const s = e.start_sec
  let end = e.end_sec
  if (!Number.isFinite(end) || end <= s) end = s + 0.12
  return end
}

export function getEntitiesActiveAtPlayback(entities: EntityRecord[], currentTimeSec: number): EntityRecord[] {
  if (!entities.length) return []
  const t = Math.max(0, currentTimeSec)

  const active = entities.filter((e) => {
    const s = e.start_sec
    const end = normalizeEntityEnd(e)
    return t >= s && t < end
  })

  active.sort((a, b) => {
    const da = normalizeEntityEnd(a) - a.start_sec
    const db = normalizeEntityEnd(b) - b.start_sec
    if (Math.abs(da - db) > 1e-6) return da - db
    return a.text.localeCompare(b.text)
  })

  const seen = new Set<string>()
  const deduped: EntityRecord[] = []
  for (const e of active) {
    const k = `${e.type}\0${e.text.toLowerCase()}`
    if (seen.has(k)) continue
    seen.add(k)
    deduped.push(e)
  }
  return deduped
}
