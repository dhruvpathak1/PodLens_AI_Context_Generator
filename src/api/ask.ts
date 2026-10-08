/** `POST /api/ask`: grounded question answering over the episode transcript. */

import { apiUrl } from '../config/env'
import type { AskResponse, ChatTurn } from '../types/ask'
import type { EntityRecord } from '../types/entities'
import type { TranscriptSegment } from '../types/transcript'
import { postJson } from './http'

export type AskParams = {
  question: string
  segments: TranscriptSegment[]
  entities: EntityRecord[]
  /** Earlier turns, so follow-ups like "what happened next?" work. */
  history: ChatTurn[]
  sourceLabel: string | null
  signal?: AbortSignal
}

/** Ask a question; resolves with the answer and its validated timestamp citations. */
export function askEpisode(params: AskParams): Promise<AskResponse> {
  return postJson<AskResponse>(
    apiUrl('/api/ask'),
    {
      question: params.question,
      segments: params.segments.map(({ id, start, end, text }) => ({ id, start, end, text })),
      entities: params.entities.map(({ type, text, start_sec }) => ({ type, text, start_sec })),
      history: params.history,
      source_label: params.sourceLabel,
    },
    params.signal
  )
}
