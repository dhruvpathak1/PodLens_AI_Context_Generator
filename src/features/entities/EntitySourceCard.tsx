import { useId, useState } from 'react'
import { formatTimeRange } from '../../shared/format'
import { ChevronDownIcon } from '../../shared/icons'
import type { EnrichedEntityCard } from '../../types/entities'

type Props = {
  card: EnrichedEntityCard
  /** Start open (full summary, image, photo, map). Used for cards in the Live panel. */
  defaultExpanded?: boolean
}

/** Summaries longer than this are clamped to three lines while collapsed. */
const LONG_EXTRACT_CHARS = 160

/**
 * Card for one entity. Collapsed: badge, time, title and a three-line Wikipedia summary.
 * Expanded (chevron or "Read more"): full summary, Wikipedia image, Unsplash photo and map.
 * Heavy media (images, map iframe) only mounts when expanded.
 */
export function EntitySourceCard({ card, defaultExpanded = false }: Props) {
  // Hide broken images instead of showing the browser's broken-image icon.
  const [wikiImgOk, setWikiImgOk] = useState(true)
  const [unsplashImgOk, setUnsplashImgOk] = useState(true)
  const [expanded, setExpanded] = useState(defaultExpanded)
  const bodyId = useId()
  const showMap = card.type === 'PLACE' && card.location
  // Collapsed = title + 3-line summary. Expanding reveals full text, image, photo and map.
  const hasMore =
    (card.wikipedia?.extract?.length ?? 0) > LONG_EXTRACT_CHARS ||
    !!card.wikipedia?.thumbnail ||
    !!card.unsplash ||
    !!showMap

  return (
    <article className={`entity-source-card${expanded ? ' entity-source-card--expanded' : ''}`}>
      <header className="entity-source-card__head">
        <span className="entity-source-card__badge" data-entity-type={card.type}>
          {card.type}
        </span>
        <span className="entity-source-card__head-right">
          <time className="entity-source-card__time">{formatTimeRange(card.start_sec, card.end_sec)}</time>
          {hasMore ? (
            <button
              type="button"
              className="entity-source-card__toggle"
              aria-expanded={expanded}
              aria-controls={bodyId}
              aria-label={expanded ? `Collapse ${card.text}` : `Expand ${card.text}`}
              title={expanded ? 'Show less' : 'Read more'}
              onClick={() => setExpanded((v) => !v)}
            >
              <ChevronDownIcon />
            </button>
          ) : null}
        </span>
      </header>
      <h3 className="entity-source-card__title">{card.text}</h3>

      <div className="entity-source-card__cols" id={bodyId}>
        {card.wikipedia && (
          <section className="entity-source-card__block">
            <h4 className="entity-source-card__label">Wikipedia</h4>
            <div className="entity-source-card__wiki">
              {expanded && card.wikipedia.thumbnail && wikiImgOk ? (
                <img
                  src={card.wikipedia.thumbnail}
                  alt=""
                  className="entity-source-card__thumb"
                  loading="lazy"
                  referrerPolicy="no-referrer"
                  onError={() => setWikiImgOk(false)}
                />
              ) : null}
              <p className="entity-source-card__extract">{card.wikipedia.extract}</p>
              <a
                href={card.wikipedia.url}
                target="_blank"
                rel="noopener noreferrer"
                className="entity-source-card__link"
              >
                Open article →
              </a>
            </div>
          </section>
        )}

        {expanded && card.unsplash ? (
          <section className="entity-source-card__block">
            <h4 className="entity-source-card__label">Photo</h4>
            <div className="entity-source-card__unsplash">
              {unsplashImgOk ? (
                <img
                  src={card.unsplash.thumb_url || card.unsplash.image_url}
                  alt={card.unsplash.alt || `Photo related to ${card.text}`}
                  className="entity-source-card__thumb entity-source-card__thumb--unsplash"
                  loading="lazy"
                  referrerPolicy="no-referrer"
                  onError={() => setUnsplashImgOk(false)}
                />
              ) : null}
              <p className="entity-source-card__unsplash-credit">
                {card.unsplash.photographer_name ? (
                  <>
                    Photo by{' '}
                    {card.unsplash.photographer_url ? (
                      <a
                        href={card.unsplash.photographer_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {card.unsplash.photographer_name}
                      </a>
                    ) : (
                      card.unsplash.photographer_name
                    )}{' '}
                    on{' '}
                  </>
                ) : (
                  'Photo on '
                )}
                {card.unsplash.unsplash_url ? (
                  <a href={card.unsplash.unsplash_url} target="_blank" rel="noopener noreferrer">
                    Unsplash
                  </a>
                ) : (
                  'Unsplash'
                )}
              </p>
            </div>
          </section>
        ) : null}

        {expanded && showMap && card.location ? (
          <section className="entity-source-card__block">
            <h4 className="entity-source-card__label">Map</h4>
            <p className="entity-source-card__geo">{card.location.display_name}</p>
            <iframe
              title={`Map: ${card.text}`}
              className="entity-source-card__map"
              src={card.location.map_embed_url}
              loading="lazy"
              referrerPolicy="no-referrer-when-downgrade"
            />
            <a
              href={card.location.openstreetmap_url}
              target="_blank"
              rel="noopener noreferrer"
              className="entity-source-card__link"
            >
              OpenStreetMap →
            </a>
          </section>
        ) : null}
      </div>

      {!card.wikipedia && !showMap && !card.unsplash && (
        <p className="entity-source-card__empty">No external matches for this tag.</p>
      )}

      {hasMore ? (
        <button type="button" className="entity-source-card__more" onClick={() => setExpanded((v) => !v)}>
          {expanded
            ? 'Show less'
            : `Read more${card.unsplash ? ' · photo' : ''}${showMap ? ' · map' : ''}`}
        </button>
      ) : null}
    </article>
  )
}
