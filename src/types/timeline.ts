/** Episode timeline data returned by `POST /api/timeline`. */

export type TimelineCategory = 'COMPANY' | 'PERSON' | 'PRODUCT' | 'TECHNOLOGY' | 'EVENT' | 'PLACE' | 'DEAL' | 'OTHER'

/** One thing mentioned in the episode that can be placed on a timeline. */
export type TimelineEvent = {
  id: string
  title: string
  description: string
  category: TimelineCategory
  /** Real-world date as stated, e.g. "1993", "late 1990s"; null when undated. */
  date_label: string | null
  year: number | null
  month: number | null
  /** Seconds into the episode where it is first mentioned. */
  mentioned_at_sec: number
  entities: string[]
  /** 1 = passing mention, 2 = notable, 3 = central to the episode. */
  importance: 1 | 2 | 3
}

/** Full timeline response. */
export type EpisodeTimelineData = {
  model: string
  events: TimelineEvent[]
  /** Number of transcript windows sent to the model, and how many failed. */
  windows: number
  failed_windows: number
}
