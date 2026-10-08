import { useEffect, useMemo, useRef, useState } from 'react'
import type { EnrichedEntityCard, EntityRecord } from '../../types/entities'
import { getEntitiesActiveAtPlayback } from './activeEntities'
import { entityMatchKey, findCardForEntity, placeholderCard } from './entityKeys'

/** Maximum cards shown at once in the Live panel; the oldest rolls off. */
export const LIVE_QUEUE_MAX = 3

type Params = {
  /** Mentions to follow (already filtered by the entity type filter). */
  entities: EntityRecord[]
  playbackTime: number
  /** Enriched cards; placeholders are swapped for these as they arrive. */
  cards: EnrichedEntityCard[]
  /** Only follow playback when an audio file is loaded. */
  enabled: boolean
  /** Any change (new file, new filter, new results) clears the queue. */
  resetKey: unknown
}

/**
 * The Live panel queue: when playback reaches a new mention, its card is pushed onto a
 * small queue (max `LIVE_QUEUE_MAX`); the oldest card drops off.
 */
export function useLiveRollingCards({ entities, playbackTime, cards, enabled, resetKey }: Params) {
  /** Queue items tagged with the `resetKey` they belong to; a new key means an empty queue. */
  const [state, setState] = useState<{ key: unknown; items: EnrichedEntityCard[] }>({ key: resetKey, items: [] })
  /** Keys active on the previous tick, so each mention is added only when it *starts*. */
  const prevActiveKeysRef = useRef<Set<string>>(new Set())
  /** The `resetKey` that `prevActiveKeysRef` belongs to. */
  const prevResetKeyRef = useRef<unknown>(resetKey)

  const activeEntities = useMemo(() => getEntitiesActiveAtPlayback(entities, playbackTime), [entities, playbackTime])

  // Push newly started mentions onto the queue.
  useEffect(() => {
    // New file, filter or results: forget which mentions were already seen.
    if (prevResetKeyRef.current !== resetKey) {
      prevResetKeyRef.current = resetKey
      prevActiveKeysRef.current = new Set()
    }
    if (!enabled) return
    if (!entities.length) {
      prevActiveKeysRef.current = new Set()
      return
    }
    const currentKeys = new Set(activeEntities.map((e) => entityMatchKey(e.type, e.text)))
    const newlyActive = activeEntities.filter((e) => !prevActiveKeysRef.current.has(entityMatchKey(e.type, e.text)))
    prevActiveKeysRef.current = currentKeys
    if (!newlyActive.length) return

    setState((prev) => {
      const base = prev.key === resetKey ? prev.items : []
      const inQueue = new Set(base.map((c) => entityMatchKey(c.type, c.text)))
      const next = [...base]
      for (const e of newlyActive) {
        const key = entityMatchKey(e.type, e.text)
        if (inQueue.has(key)) continue
        next.push(findCardForEntity(cards, e) ?? placeholderCard(e))
        inQueue.add(key)
      }
      while (next.length > LIVE_QUEUE_MAX) next.shift()
      return { key: resetKey, items: next }
    })
  }, [enabled, activeEntities, entities.length, cards, resetKey])

  // Items from an older resetKey are hidden; placeholders are swapped for enriched cards once available.
  const items = state.key === resetKey ? state.items : EMPTY
  return useMemo(() => items.map((c) => findCardForEntity(cards, c) ?? c), [items, cards])
}

const EMPTY: EnrichedEntityCard[] = []
