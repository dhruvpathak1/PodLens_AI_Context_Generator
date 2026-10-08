import { ENTITY_FILTER_OPTIONS, type EntityDocument } from '../../types/entities'
import type { EntityFilterState } from './useEntityFilter'

type Props = {
  entityDoc: EntityDocument
  filter: EntityFilterState
  /** Server file the entities were saved to, shown next to the backend name. */
  entitySavedPath: string | null
  /** False when Unsplash is not configured: shows a setup hint (only if cards exist). */
  unsplashHint: boolean | null
  hasCards: boolean
}

/** "Tagged entities" panel: backend info, per-type filter buttons with counts, Unsplash hint. */
export function EntityFilterBar({ entityDoc, filter, entitySavedPath, unsplashHint, hasCards }: Props) {
  return (
    <div className="entity-panel entity-panel--filter-only">
      {entityDoc.entities.length === 0 ? (
        <p className="entity-panel__empty">No entities found after tagging and noise filtering.</p>
      ) : (
        <div className="entity-filter-toolbar">
          <div className="entity-filter-toolbar__top">
            <h2 className="entity-filter-toolbar__title">Tagged entities</h2>
            <span className="entity-filter-toolbar__meta">
              {entityDoc.backend}
              {entitySavedPath && (
                <>
                  {' · '}
                  <code title={entitySavedPath}>{entitySavedPath.split('/').pop()}</code>
                </>
              )}
            </span>
          </div>
          <p className="entity-filter-toolbar__hint">
            Choose a type to limit <strong>Live</strong> highlights and <strong>Source cards</strong>. Numbers are
            how many tagged spans of that type appear in the export.
          </p>
          <div className="entity-filter__group" role="group" aria-label="Filter by entity type">
            {ENTITY_FILTER_OPTIONS.map((opt) => {
              const count = filter.counts?.[opt] ?? 0
              const label = opt === 'ALL' ? 'All types' : opt
              const isActive = filter.filter === opt
              const isEmptyType = opt !== 'ALL' && count === 0
              return (
                <button
                  key={opt}
                  type="button"
                  className={`entity-filter__opt${isActive ? ' entity-filter__opt--active' : ''}${
                    isEmptyType && !isActive ? ' entity-filter__opt--dim' : ''
                  }`}
                  aria-pressed={isActive}
                  aria-label={`${label}, ${count} ${count === 1 ? 'mention' : 'mentions'}`}
                  onClick={() => filter.setFilter(opt)}
                >
                  <span className="entity-filter__opt-label">{label}</span>
                  <span className="entity-filter__opt-count" aria-hidden>
                    {count}
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      )}
      {unsplashHint === false && hasCards && (
        <p className="entity-panel__hint">
          Unsplash photos skipped. Add <code className="app-inline-code">UNSPLASH_ACCESS_KEY</code> (Client ID from{' '}
          <a href="https://unsplash.com/developers" target="_blank" rel="noopener noreferrer" className="entity-panel__hint-link">
            unsplash.com/developers
          </a>
          ) to the API server env.
        </p>
      )}
    </div>
  )
}
