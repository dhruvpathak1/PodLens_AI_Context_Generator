/**
 * Build-time configuration (Vite `import.meta.env`) and feature flags.
 * Only variables prefixed with `VITE_` are exposed to the browser.
 */

/** Browser tab title. */
export const APP_TITLE = 'PodLens'

/**
 * Live microphone mode is future scope. Its code is parked in `features/live/` and is not
 * imported by the app. See `features/live/README.md` to re-enable it.
 */
export const LIVE_MODE_ENABLED = false

/** Raw `VITE_TRANSCRIBE_URL`, e.g. `https://api.example.com/api/transcribe` (empty in local dev). */
const transcribeUrl = (import.meta.env.VITE_TRANSCRIBE_URL ?? '').trim()

/**
 * API origin derived from `VITE_TRANSCRIBE_URL` by dropping its `/api/transcribe` suffix.
 * Empty string means "same origin": in dev, Vite proxies `/api/*` to the FastAPI server.
 */
const API_BASE = transcribeUrl.replace(/\/+$/, '').replace(/\/api\/transcribe$/i, '')

/** Build the URL for an API path, e.g. `apiUrl('/api/ask')`. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

/**
 * True for a production build with no API URL configured. A static host (GitHub Pages) cannot
 * run the API, so the app shows a setup banner instead of failing silently.
 */
export const MISSING_PROD_API_URL = import.meta.env.PROD && !transcribeUrl

/** Optional client-side override of the NER backend (`VITE_ENTITY_BACKEND=spacy|claude`). */
export const ENTITY_BACKEND: 'spacy' | 'claude' | undefined = (() => {
  const v = import.meta.env.VITE_ENTITY_BACKEND
  return v === 'spacy' || v === 'claude' ? v : undefined
})()
