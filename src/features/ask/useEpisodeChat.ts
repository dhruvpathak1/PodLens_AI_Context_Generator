import { useCallback, useRef, useState } from 'react'
import { askEpisode } from '../../api/ask'
import type { AskCitation, AskResponse, ChatTurn } from '../../types/ask'
import type { EntityRecord } from '../../types/entities'
import type { TranscriptSegment } from '../../types/transcript'

/** A message in the Ask conversation: the user's question, an answer, or an inline error. */
export type ChatMessage =
  | { id: string; role: 'user'; content: string }
  | {
      id: string
      role: 'assistant'
      content: string
      found: boolean
      citations: AskCitation[]
      unverified: boolean
      /** Questions offered as links under the answer (demo mode, when nothing matched). */
      followups: string[]
    }
  | { id: string; role: 'error'; content: string }

/** Episode data sent with every question (read fresh at ask time). */
type Context = {
  segments: TranscriptSegment[]
  entities: EntityRecord[]
  sourceLabel: string | null
}

/** Replaces the API call (demo mode answers from pre-computed results). */
export type AnswerFn = (question: string, history: ChatTurn[], signal: AbortSignal) => Promise<AskResponse>

let nextId = 0
/** Unique React key for a message. */
const mkId = () => `m${++nextId}`

/**
 * Chat state for "Ask the episode".
 * One question in flight at a time; `stop()` aborts it, `reset()` clears the conversation.
 * Pass `answer` to answer without the API (demo mode).
 */
export function useEpisodeChat(getContext: () => Context, answer?: AnswerFn) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const messagesRef = useRef<ChatMessage[]>([])
  const getContextRef = useRef(getContext)
  getContextRef.current = getContext
  const answerRef = useRef(answer)
  answerRef.current = answer

  /** Update messages in state and in a ref (the ref is read synchronously when building history). */
  const commit = (next: ChatMessage[]) => {
    messagesRef.current = next
    setMessages(next)
  }

  /** Send a question with the conversation so far; appends the answer or an error message. */
  const ask = useCallback(async (raw: string) => {
    const question = raw.trim()
    if (!question || abortRef.current) return
    const ctx = getContextRef.current()
    if (!ctx.segments.length) return

    // History = completed user/assistant turns only (errors are not sent).
    const history: ChatTurn[] = messagesRef.current
      .filter((m): m is Extract<ChatMessage, { role: 'user' | 'assistant' }> => m.role !== 'error')
      .map((m) => ({ role: m.role, content: m.content }))

    commit([...messagesRef.current, { id: mkId(), role: 'user', content: question }])
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setBusy(true)
    try {
      const res = answerRef.current
        ? await answerRef.current(question, history, ctrl.signal)
        : await askEpisode({ question, history, signal: ctrl.signal, ...ctx })
      commit([
        ...messagesRef.current,
        {
          id: mkId(),
          role: 'assistant',
          content: res.answer,
          found: res.found,
          citations: res.citations,
          unverified: res.unverified,
          followups: res.followups ?? [],
        },
      ])
    } catch (e) {
      if (!ctrl.signal.aborted) {
        commit([
          ...messagesRef.current,
          { id: mkId(), role: 'error', content: e instanceof Error ? e.message : 'Question failed' },
        ])
      }
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null
      setBusy(false)
    }
  }, [])

  /** Abort the question in flight (no message is added). */
  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setBusy(false)
  }, [])

  /** Abort anything in flight and clear the conversation. */
  const reset = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setBusy(false)
    commit([])
  }, [])

  return { messages, busy, ask, stop, reset }
}

export type EpisodeChat = ReturnType<typeof useEpisodeChat>
