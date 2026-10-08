import { useMemo, useState } from 'react'
import { ENTITY_FILTER_OPTIONS, type EnrichedEntityCard, type EntityDocument, type EntityFilterOption } from '../../types/entities'

/**
 * Entity type filter shared by the Live panel and Source cards.
 * Returns the selection, per-type counts for the filter buttons, and filtered lists.
 */
export function useEntityFilter(entityDoc: EntityDocument | null, cards: EnrichedEntityCard[]) {
  const [filter, setFilter] = useState<EntityFilterOption>('ALL')

  /** Mention count per filter option (`ALL` = every mention). Null until entities exist. */
  const counts = useMemo(() => {
    const entities = entityDoc?.entities ?? []
    if (!entities.length) return null
    const result = Object.fromEntries(ENTITY_FILTER_OPTIONS.map((o) => [o, 0])) as Record<EntityFilterOption, number>
    result.ALL = entities.length
    for (const e of entities) {
      if (e.type in result) result[e.type as EntityFilterOption] += 1
    }
    return result
  }, [entityDoc])

  /** Source cards matching the filter. */
  const visibleCards = useMemo(
    () => (filter === 'ALL' ? cards : cards.filter((c) => c.type === filter)),
    [cards, filter]
  )

  /** Entity mentions matching the filter (drives the Live panel). */
  const filteredEntities = useMemo(() => {
    const entities = entityDoc?.entities ?? []
    return filter === 'ALL' ? entities : entities.filter((e) => e.type === filter)
  }, [entityDoc, filter])

  return { filter, setFilter, counts, visibleCards, filteredEntities, resetFilter: () => setFilter('ALL') }
}

export type EntityFilterState = ReturnType<typeof useEntityFilter>
