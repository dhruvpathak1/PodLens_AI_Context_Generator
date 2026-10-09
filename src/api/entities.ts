/** Entity endpoints: tag existing chunks, and enrich entities into source cards. */

import { apiUrl } from '../config/env'
import type { EnrichedEntityCard, EntityDocument, EntityRecord, EntityReviewReport } from '../types/entities'
import type { TranscriptSegment } from '../types/transcript'
import { postJson } from './http'

/** A transcript chunk to tag. */
export type EntityChunk = { id: number; start: number; end: number; text: string }

export type ExtractEntitiesOptions = {
  sourceLabel?: string
  /** Save the document to the server's entity_exports folder (default true). */
  persist?: boolean
  backend?: 'spacy' | 'claude'
}

export type ExtractEntitiesResult = { document: EntityDocument; saved_path: string | null }

export type EnrichEntitiesResponse = {
  cards: EnrichedEntityCard[]
  count: number
  /** False when the server has no Unsplash key (photos skipped). */
  unsplash_enabled?: boolean
  /** Reviewed entity mentions to display (present when the AI review ran). */
  entities?: EntityRecord[]
  review?: EntityReviewReport
}

export type EnrichOptions = {
  /** Transcript segments: lets the server review entities in context before the lookups. */
  segments?: TranscriptSegment[]
  sourceLabel?: string | null
}

/** `POST /api/extract-entities`: tag entities in caller-supplied chunks. */
export async function extractEntities(
  chunks: EntityChunk[],
  options: ExtractEntitiesOptions = {}
): Promise<ExtractEntitiesResult> {
  const data = await postJson<ExtractEntitiesResult>(apiUrl('/api/extract-entities'), {
    chunks: chunks.map(({ id, start, end, text }) => ({ id, start, end, text })),
    source_label: options.sourceLabel ?? null,
    persist: options.persist !== false,
    backend: options.backend ?? null,
  })
  if (!data.document || !Array.isArray(data.document.entities)) {
    throw new Error('Invalid extract-entities response')
  }
  return data
}

/**
 * `POST /api/enrich-entities`: one source card per unique entity (Wikipedia, map, photo).
 * With `segments`, the server first has an LLM review the entities in context (when it has an
 * OpenAI key) and returns the reviewed list in `entities`.
 */
export function enrichEntityCards(entities: EntityRecord[], options: EnrichOptions = {}): Promise<EnrichEntitiesResponse> {
  return postJson<EnrichEntitiesResponse>(apiUrl('/api/enrich-entities'), {
    entities: entities.map(({ type, text, start_sec, end_sec, chunk_id }) => ({
      type,
      text,
      start_sec,
      end_sec,
      chunk_id,
    })),
    segments: (options.segments ?? []).map(({ id, start, end, text }) => ({ id, start, end, text })),
    source_label: options.sourceLabel ?? null,
  })
}
