/** Entity endpoints: tag existing chunks, and enrich entities into source cards. */

import { apiUrl } from '../config/env'
import type { EnrichedEntityCard, EntityDocument, EntityRecord } from '../types/entities'
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

/** `POST /api/enrich-entities`: one source card per unique entity (Wikipedia, map, photo). */
export function enrichEntityCards(entities: EntityRecord[]): Promise<EnrichEntitiesResponse> {
  return postJson<EnrichEntitiesResponse>(apiUrl('/api/enrich-entities'), {
    entities: entities.map(({ type, text, start_sec, end_sec, chunk_id }) => ({
      type,
      text,
      start_sec,
      end_sec,
      chunk_id,
    })),
  })
}
