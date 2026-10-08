/** Entity (NER) and source-card data shared across the app. Mirrors the API's JSON. */

/** Entity types produced by the backend. */
export const ENTITY_TYPES = ['PLACE', 'PERSON', 'TECHNOLOGY', 'EVENT', 'COMPANY', 'MISC'] as const
export type EntityType = (typeof ENTITY_TYPES)[number]

/** Options for the entity filter bar (`ALL` plus every entity type, in display order). */
export const ENTITY_FILTER_OPTIONS = ['ALL', ...ENTITY_TYPES] as const
export type EntityFilterOption = (typeof ENTITY_FILTER_OPTIONS)[number]

/** One tagged mention of an entity in the transcript. */
export type EntityRecord = {
  /** Entity type (kept as `string` so unknown future types do not break parsing). */
  type: string
  text: string
  /** Estimated time the mention is spoken (seconds). */
  start_sec: number
  end_sec: number
  /** Id of the transcript segment the mention came from. */
  chunk_id: number
  /** Which backend produced it: "spacy" | "claude". */
  source?: string
  /** The backend's own label before mapping (e.g. "GPE", "ORG"). */
  original_label?: string
}

/** Versioned NER output for one episode (also saved to disk by the server). */
export type EntityDocument = {
  schema_version: number
  extracted_at: string
  backend: string
  source_label: string | null
  chunks: Array<{
    id: number
    start_sec: number
    end_sec: number
    text_raw: string
    text_clean: string
  }>
  entities: EntityRecord[]
}

/** Wikipedia summary attached to a source card. */
export type WikipediaInfo = {
  title: string
  extract: string
  url: string
  thumbnail?: string | null
}

/** Geocoded place attached to PLACE cards. */
export type LocationInfo = {
  lat: number
  lon: number
  display_name: string
  map_embed_url: string
  openstreetmap_url: string
}

/** Unsplash photo plus the attribution fields Unsplash requires us to show. */
export type UnsplashPhotoInfo = {
  image_url: string
  thumb_url?: string | null
  alt?: string
  photographer_name: string
  photographer_url: string
  unsplash_url: string
}

/** One unique entity enriched with external sources. */
export type EnrichedEntityCard = {
  id: string
  type: string
  text: string
  start_sec: number
  end_sec: number
  chunk_id: number
  wikipedia: WikipediaInfo | null
  location: LocationInfo | null
  unsplash: UnsplashPhotoInfo | null
}
