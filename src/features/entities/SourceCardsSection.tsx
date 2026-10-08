import type { EnrichedEntityCard } from '../../types/entities'
import { EntitySourceCard } from './EntitySourceCard'

type Props = {
  /** Cards after the entity type filter is applied. */
  cards: EnrichedEntityCard[]
}

/** Grid of collapsible source cards, one per unique entity. */
export function SourceCardsSection({ cards }: Props) {
  return (
    <div className="entity-cards-section">
      <h2 className="entity-cards-section__title">Source cards</h2>
      <p className="entity-cards-section__lead">
        One card per unique tag: Wikipedia REST API, Unsplash when configured, and OpenStreetMap Nominatim. Use the{' '}
        <strong>type filter</strong> above to limit which cards appear.
      </p>
      {cards.length > 0 ? (
        <div className="entity-cards-grid">
          {cards.map((c) => (
            <EntitySourceCard key={c.id} card={c} />
          ))}
        </div>
      ) : (
        <p className="entity-cards-section__empty">
          No source cards for this type. Choose <strong>ALL</strong> or another category.
        </p>
      )}
    </div>
  )
}
