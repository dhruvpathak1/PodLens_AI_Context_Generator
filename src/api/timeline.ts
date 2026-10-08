/** `POST /api/timeline`: chronological timeline built by the LLM after transcription + NER. */

import { apiUrl } from '../config/env'
import type { EntityRecord } from '../types/entities'
import type { EpisodeTimelineData } from '../types/timeline'
import type { TranscriptSegment } from '../types/transcript'
import { postJson } from './http'

/** Send the transcript and extracted entities; resolves with the generated timeline. */
export async function createEpisodeTimeline(
  segments: TranscriptSegment[],
  entities: EntityRecord[],
  sourceLabel: string | null,
  signal?: AbortSignal
): Promise<EpisodeTimelineData> {
  const data = await postJson<EpisodeTimelineData>(
    apiUrl('/api/timeline'),
    {
      segments: segments.map(({ id, start, end, text }) => ({ id, start, end, text })),
      entities: entities.map(({ type, text, start_sec }) => ({ type, text, start_sec })),
      source_label: sourceLabel,
    },
    signal
  )
  if (!data || !Array.isArray(data.events)) throw new Error('Timeline response had no events')
  return data
}
