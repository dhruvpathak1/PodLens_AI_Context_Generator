import { useMemo, useState } from 'react'
import { ENTITY_FILTER_OPTIONS, type EntityDocument, type EntityFilterOption } from '../../types/entities'

/**
 * Entity type filter shared by the Entities panel and the live cards.
 * Returns the selection, per-type mention counts, and the mentions matching the filter.
 */
export function useEntityFilter(entityDoc: EntityDocument | null) {
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

  /** Entity mentions matching the filter (drives the live cards). */
  const filteredEntities = useMemo(() => {
    const entities = entityDoc?.entities ?? []
    return filter === 'ALL' ? entities : entities.filter((e) => e.type === filter)
  }, [entityDoc, filter])

  return { filter, setFilter, counts, filteredEntities, resetFilter: () => setFilter('ALL') }
}

export type EntityFilterState = ReturnType<typeof useEntityFilter>
