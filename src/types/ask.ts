/** "Ask the episode" request/response types (`POST /api/ask`). */

/** A transcript line that supports an answer. */
export type AskCitation = { start_sec: number; quote: string }

export type AskResponse = {
  answer: string
  /** False when the transcript does not cover the question. */
  found: boolean
  citations: AskCitation[]
  /** `full` = whole transcript was sent; `retrieval` = top-K embedded chunks. */
  mode: 'full' | 'retrieval'
  model: string
  /** The model claimed an answer but none of its citations matched the transcript. */
  unverified: boolean
}

/** One earlier turn, sent back so follow-up questions have context. */
export type ChatTurn = { role: 'user' | 'assistant'; content: string }
