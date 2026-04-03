const DEFAULT_URL = '/api/extract-entities'

export type EntityChunk = {
  id: number
  start: number
  end: number
  text: string
}

export type ExtractEntitiesOptions = {
  endpoint?: string
  sourceLabel?: string
  persist?: boolean
  /** Override server default: "spacy" | "claude" */
  backend?: 'spacy' | 'claude'
}

export type EntityRecord = {
  type: string
  text: string
  start_sec: number
  end_sec: number
  chunk_id: number
  source?: string
  original_label?: string
}

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

export type ExtractEntitiesResult = {
  document: EntityDocument
  saved_path: string | null
}

function resolveEndpoint(): string {
  const base = import.meta.env.VITE_TRANSCRIBE_URL
  if (!base) return DEFAULT_URL
  const root = String(base).replace(/\/$/, '').replace(/\/api\/transcribe$/i, '')
  return `${root}/api/extract-entities`
}

function formatErrorDetail(j: unknown): string | null {
  if (!j || typeof j !== 'object') return null
  const o = j as Record<string, unknown>
  const detail = o.detail
  if (typeof detail === 'string') return detail
  return null
}

export async function extractEntities(
  chunks: EntityChunk[],
  options: ExtractEntitiesOptions = {}
): Promise<ExtractEntitiesResult> {
  const endpoint = (options.endpoint ?? resolveEndpoint()).replace(/\/$/, '')
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chunks: chunks.map((c) => ({
        id: c.id,
        start: c.start,
        end: c.end,
        text: c.text,
      })),
      source_label: options.sourceLabel ?? null,
      persist: options.persist !== false,
      backend: options.backend ?? null,
    }),
  })

  const ct = res.headers.get('content-type') ?? ''
  if (!res.ok) {
    let detail = res.statusText
    try {
      if (ct.includes('application/json')) {
        const j = await res.json()
        const d = formatErrorDetail(j)
        if (d) detail = d
      } else {
        const t = await res.text()
        if (t) detail = t.slice(0, 500)
      }
    } catch {
      /* keep */
    }
    throw new Error(detail || `Request failed (${res.status})`)
  }

  const data = (await res.json()) as ExtractEntitiesResult
  if (!data.document || !Array.isArray(data.document.entities)) {
    throw new Error('Invalid extract-entities response')
  }
  return data
}
