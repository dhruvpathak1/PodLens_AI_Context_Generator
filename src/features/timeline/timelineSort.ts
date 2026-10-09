/** Ordering and date formatting for timeline events. */

import type { TimelineEvent } from '../../types/timeline'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** Human date for an event: the model's own label, else `Mon YYYY` / `YYYY`, else null. */
export function formatEventDate(e: TimelineEvent): string | null {
  if (e.date_label && e.date_label.trim()) return e.date_label.trim()
  if (e.year == null) return null
  return e.month ? `${MONTHS[e.month - 1]} ${e.year}` : String(e.year)
}

/** Trim trailing ".0" from a fixed-point number ("4.0" -> "4"). */
const trim = (n: number) => n.toFixed(1).replace(/\.0$/, '')

/**
 * Compact "when" for the date column: a short headline value plus an optional unit line.
 *   label "late 1700s" -> { value: "late 1700s" } (short labels win)
 *   1849 -> { value: "1849" }, Jul 1969 -> { value: "1969", unit: "Jul" }
 *   -4.5e9 -> { value: "4.5B", unit: "years ago" } , -66e6 -> { value: "66M", unit: "years ago" }
 *   -500 -> { value: "500", unit: "BCE" }
 */
export function compactWhen(e: TimelineEvent): { value: string; unit?: string } | null {
  const y = e.year
  if (y == null) return null
  // Short wording from the model ("late 1700s", "1960s") reads better than the bare year.
  const label = e.date_label?.trim()
  if (label && label.length <= 11 && !/years? ago/i.test(label)) return { value: label }
  if (y <= -1e9) return { value: `${trim(-y / 1e9)}B`, unit: 'years ago' }
  if (y <= -1e6) return { value: `${trim(-y / 1e6)}M`, unit: 'years ago' }
  if (y <= -1e4) return { value: `${trim(-y / 1e3)}K`, unit: 'years ago' }
  if (y < 0) return { value: String(-y), unit: 'BCE' }
  return e.month ? { value: String(y), unit: MONTHS[e.month - 1] } : { value: String(y) }
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
