import { useState } from 'react'
import { formatClock } from '../../shared/format'
import { ExternalIcon } from '../../shared/icons'
import type { EnrichedEntityCard } from '../../types/entities'
import { typeLabel } from './entityKeys'

type Props = { card: EnrichedEntityCard }

/**
 * One source card in the live grid: photo (Unsplash, else the Wikipedia image), type and time,
 * name, Wikipedia summary (scrolls inside the card), and links. Built to fit a quarter of the grid.
 */
export function LiveCard({ card }: Props) {
  const [imgOk, setImgOk] = useState(true)
  const img = card.unsplash?.thumb_url || card.unsplash?.image_url || card.wikipedia?.thumbnail || null
  const credit = card.unsplash?.photographer_name

  return (
    <article className="live-card">
      {img && imgOk ? (
        <figure className="live-card__media">
          <img src={img} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setImgOk(false)} />
          {credit ? (
            <figcaption className="live-card__credit">
              Photo:{' '}
              {card.unsplash?.photographer_url ? (
                <a href={card.unsplash.photographer_url} target="_blank" rel="noopener noreferrer">
                  {credit}
                </a>
              ) : (
                credit
              )}{' '}
              /{' '}
              <a href={card.unsplash?.unsplash_url || 'https://unsplash.com'} target="_blank" rel="noopener noreferrer">
                Unsplash
              </a>
            </figcaption>
          ) : null}
        </figure>
      ) : null}
      <div className="live-card__body">
        <p className="live-card__meta">
          <span>{typeLabel(card.type)}</span>
          <time className="num">{formatClock(card.start_sec)}</time>
        </p>
        <h3 className="live-card__title">{card.text}</h3>
        {card.wikipedia?.extract ? (
          <p className="live-card__text">{card.wikipedia.extract}</p>
        ) : (
          <p className="live-card__text live-card__text--muted">No encyclopedia entry found for this name.</p>
        )}
        <p className="live-card__links">
          {card.wikipedia?.url ? (
            <a href={card.wikipedia.url} target="_blank" rel="noopener noreferrer">
              Wikipedia <ExternalIcon />
            </a>
          ) : null}
          {card.location?.openstreetmap_url ? (
            <a href={card.location.openstreetmap_url} target="_blank" rel="noopener noreferrer">
              Map <ExternalIcon />
            </a>
          ) : null}
        </p>
      </div>
    </article>
  )
}
