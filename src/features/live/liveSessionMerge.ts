// PARKED (future scope): not imported by the app. See ./README.md for how to re-enable live mode.
/** Helpers that grow one session transcript and entity document as live chunks arrive. */

import type { EntityDocument } from '../../types/entities'
import type { TranscriptSegment } from '../../types/transcript'

/** Append newly transcribed segments, keeping the list ordered by start time (then id). */
export function mergeTranscriptSegments(
  prev: TranscriptSegment[],
  incoming: TranscriptSegment[]
): TranscriptSegment[] {
  if (!incoming.length) return prev
  return [...prev, ...incoming].sort((a, b) => a.start - b.start || a.id - b.id)
}

/**
 * Fold one chunk's entity document into the session document.
 * The first chunk becomes the session document; later chunks append chunks and entities.
 */
export function mergeEntityDocuments(
  prev: EntityDocument | null,
  incoming: EntityDocument
): EntityDocument {
  if (!prev) {
    return {
      ...incoming,
      source_label: incoming.source_label ?? 'live microphone',
    }
  }
  return {
    schema_version: prev.schema_version,
    extracted_at: incoming.extracted_at,
    backend: prev.backend,
    source_label: prev.source_label ?? incoming.source_label,
    chunks: [...prev.chunks, ...incoming.chunks],
    entities: [...prev.entities, ...incoming.entities],
  }
}
