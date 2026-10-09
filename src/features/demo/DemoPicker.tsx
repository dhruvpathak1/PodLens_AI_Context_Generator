import { REPO_URL } from '../../config/env'
import { ErrorAlert } from '../../shared/components/Alert'
import { formatTimestamp } from '../../shared/format'
import type { DemoCatalog } from './useDemoCatalog'

type Props = { catalog: DemoCatalog }

/** Sidebar episode picker shown instead of the upload box in demo mode. */
export function DemoPicker({ catalog }: Props) {
  const { episodes, indexLoaded, activeSlug, loadingSlug, error, select } = catalog

  if (indexLoaded && episodes.length === 0) {
    return (
      <div className="demo-picker demo-picker--empty">
        <p className="demo-picker__empty">
          No demo episodes yet. Build one with <code>npm run demo:build</code> (see the{' '}
          <a href={`${REPO_URL}#-live-demo`} target="_blank" rel="noreferrer">
            README
          </a>
          ).
        </p>
      </div>
    )
  }

  return (
    <div className="demo-picker" role="group" aria-label="Demo episodes">
      <p className="demo-picker__label">Pick an episode</p>
      <ul className="demo-picker__list">
        {episodes.map((ep) => {
          const active = ep.slug === activeSlug
          const loading = ep.slug === loadingSlug
          return (
            <li key={ep.slug}>
              <button
                type="button"
                className={`demo-picker__item${active ? ' demo-picker__item--active' : ''}`}
                aria-pressed={active}
                aria-busy={loading}
                onClick={() => void select(ep.slug)}
              >
                <span className="demo-picker__title">{ep.title}</span>
                {ep.description ? <span className="demo-picker__desc">{ep.description}</span> : null}
                <span className="demo-picker__meta">
                  {loading ? 'Loading…' : `${formatTimestamp(ep.duration_sec)} · ${ep.entity_count} entities`}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <ErrorAlert>{error}</ErrorAlert>
    </div>
  )
}
