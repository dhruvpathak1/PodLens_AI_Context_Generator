/** Ordering and date formatting for timeline events. */

import type { TimelineEvent } from '../../types/timeline'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** Human date for an event: the model's own label, else `Mon YYYY` / `YYYY`, else null. */
export function formatEventDate(e: TimelineEvent): string | null {
  if (e.date_label && e.date_label.trim()) return e.date_label.trim()
  if (e.year == null) return null
  return e.month ? `${MONTHS[e.month - 1]} ${e.year}` : String(e.year)
}

/** Real-world order (year, then month); undated events go last, in episode order. */
export function sortByHistory(events: TimelineEvent[]): TimelineEvent[] {
  return [...events].sort((a, b) => {
    if (a.year == null && b.year == null) return a.mentioned_at_sec - b.mentioned_at_sec
    if (a.year == null) return 1
    if (b.year == null) return -1
    return a.year - b.year || (a.month ?? 0) - (b.month ?? 0) || a.mentioned_at_sec - b.mentioned_at_sec
  })
}

/** Order of first mention in the episode. */
export function sortByEpisode(events: TimelineEvent[]): TimelineEvent[] {
  return [...events].sort((a, b) => a.mentioned_at_sec - b.mentioned_at_sec || a.title.localeCompare(b.title))
}
