import { useMemo } from 'react'
import { Panel } from '../../shared/components/Panel'
import { ENTITY_FILTER_OPTIONS, type EntityDocument } from '../../types/entities'
import { getEntitiesActiveAtPlayback } from './activeEntities'
import { entityMatchKey, typeLabel } from './entityKeys'
import type { EntityFilterState } from './useEntityFilter'

type Props = {
  entityDoc: EntityDocument | null
  filter: EntityFilterState
  playbackTime: number
  /** Jump to an entity's first mention. */
  onSeek: (sec: number) => void
  /** Optional setup note (e.g. Unsplash not configured), shown under the list. */
  note?: string | null
}

/** One unique entity with its mention count and first mention time. */
type Row = { key: string; type: string; text: string; count: number; first: number }

/**
 * Middle column, bottom: every entity found (NER), filterable by type. Names being spoken right
 * now are highlighted; clicking a name plays its first mention.
 */
export function EntitiesPanel({ entityDoc, filter, playbackTime, onSeek, note }: Props) {
  const entities = useMemo(() => entityDoc?.entities ?? [], [entityDoc])

  // Unique entities in order of first mention.
  const rows = useMemo(() => {
    const map = new Map<string, Row>()
    for (const e of entities) {
      const key = entityMatchKey(e.type, e.text)
      const row = map.get(key)
      if (row) {
        row.count++
        row.first = Math.min(row.first, e.start_sec)
      } else map.set(key, { key, type: e.type, text: e.text, count: 1, first: e.start_sec })
    }
    return [...map.values()].sort((a, b) => a.first - b.first)
  }, [entities])

  const visible = filter.filter === 'ALL' ? rows : rows.filter((r) => r.type === filter.filter)
  const speaking = useMemo(
    () => new Set(getEntitiesActiveAtPlayback(entities, playbackTime).map((e) => entityMatchKey(e.type, e.text))),
    [entities, playbackTime]
  )
  const types = ENTITY_FILTER_OPTIONS.filter((o) => o === 'ALL' || (filter.counts?.[o] ?? 0) > 0)

  return (
    <Panel
      className="panel--entities"
      title="Entities"
      meta={entityDoc ? `${rows.length} found${entityDoc.backend ? ` with ${entityDoc.backend === 'claude' ? 'Claude' : 'spaCy'}` : ''}` : null}
      toolbar={
        rows.length > 0 ? (
          <div className="seg" role="group" aria-label="Filter by type">
            {types.map((o) => {
              const n = o === 'ALL' ? rows.length : rows.filter((r) => r.type === o).length
              return (
                <button
                  key={o}
                  type="button"
                  className={`seg__opt${filter.filter === o ? ' seg__opt--on' : ''}`}
                  aria-pressed={filter.filter === o}
                  onClick={() => filter.setFilter(o)}
                >
                  {o === 'ALL' ? 'All' : typeLabel(o)} <span className="num seg__count">{n}</span>
                </button>
              )
            })}
          </div>
        ) : null
      }
    >
      {rows.length === 0 ? (
        <p className="empty">People, places and organisations found in the episode are listed here.</p>
      ) : (
        <ul className="chips" aria-label="Entities">
          {visible.map((r) => (
            <li key={r.key}>
              <button
                type="button"
                className={`chip${speaking.has(r.key) ? ' chip--speaking' : ''}`}
                data-type={r.type}
                onClick={() => onSeek(r.first)}
                title={`${typeLabel(r.type)}, ${r.count} ${r.count === 1 ? 'mention' : 'mentions'}. Play first mention.`}
              >
                <span className="chip__dot" aria-hidden />
                {r.text}
                {r.count > 1 ? <span className="chip__count num">{r.count}</span> : null}
              </button>
            </li>
          ))}
        </ul>
      )}
      {note ? <p className="about__note">{note}</p> : null}
    </Panel>
  )
}
