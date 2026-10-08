/** Display formatting helpers (pure functions, no React). */

/** Seconds -> `HH:MM:SS` (used for transcript, cards and timeline timestamps). */
export function formatTimestamp(sec: number): string {
  const s = Math.max(0, sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const secPart = Math.floor(s % 60)
  return [h, m, secPart].map((n) => String(n).padStart(2, '0')).join(':')
}

/** Two timestamps as a range, e.g. `00:01:05 – 00:01:12`. */
export function formatTimeRange(start: number, end: number): string {
  return `${formatTimestamp(start)} – ${formatTimestamp(end)}`
}

/** Seconds -> compact player clock: `m:ss`, or `h:mm:ss` past an hour. */
export function formatClock(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00'
  const s = Math.floor(sec)
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m >= 60) {
    return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}:${String(r).padStart(2, '0')}`
  }
  return `${m}:${String(r).padStart(2, '0')}`
}

/** Byte count -> `B`, `KB` or `MB` with sensible precision. */
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}
