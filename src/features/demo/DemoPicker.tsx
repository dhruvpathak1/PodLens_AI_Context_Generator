import { REPO_URL } from '../../config/env'
import { ErrorAlert } from '../../shared/components/Alert'
import { formatClock } from '../../shared/format'
import type { DemoCatalog } from './useDemoCatalog'

type Props = { catalog: DemoCatalog }

/** Sample episode buttons shown in the PodLens panel in demo mode. */
export function DemoPicker({ catalog }: Props) {
  const { episodes, indexLoaded, activeSlug, loadingSlug, error, select } = catalog

  if (indexLoaded && episodes.length === 0) {
    return (
      <p className="about__note">
        No sample episodes yet. Build one with <code>npm run demo:build</code> (see the{' '}
        <a href={`${REPO_URL}#-live-demo`} target="_blank" rel="noreferrer">
          README
        </a>
        ).
      </p>
    )
  }

  return (
    <div className="samples" role="group" aria-label="Sample episodes">
      {episodes.map((ep, i) => {
        const active = ep.slug === activeSlug
        return (
          <button
            key={ep.slug}
            type="button"
            className={`sample${active ? ' sample--active' : ''}`}
            aria-pressed={active}
            aria-busy={ep.slug === loadingSlug}
            onClick={() => void select(ep.slug)}
            title={ep.description || ep.title}
          >
            <span className="sample__index">Sample {i + 1}</span>
            <span className="sample__title">{ep.title}</span>
            <span className="sample__meta">
              {ep.slug === loadingSlug ? 'Loading…' : `${formatClock(ep.duration_sec)}, ${ep.entity_count} entities`}
            </span>
          </button>
        )
      })}
      <ErrorAlert>{error}</ErrorAlert>
    </div>
  )
}
