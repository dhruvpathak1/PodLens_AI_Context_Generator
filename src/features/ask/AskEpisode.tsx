import { useEffect, useRef, useState } from 'react'
import { formatTimestamp } from '../../shared/format'
import type { ChatMessage } from './useEpisodeChat'

type Props = {
  messages: ChatMessage[]
  busy: boolean
  suggestions: string[]
  onAsk: (question: string) => void
  onStop: () => void
  onClear: () => void
  /** Jump the player (and transcript) to a cited moment. */
  onCite: (sec: number, quote: string) => void
}

/** Same limit the server enforces. */
const MAX_LEN = 500

/**
 * Ask tab: suggested questions, the conversation (answers with clickable citations),
 * and the question box (Enter sends, Shift+Enter adds a new line).
 */
export function AskEpisode({ messages, busy, suggestions, onAsk, onStop, onClear, onCite }: Props) {
  const [draft, setDraft] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  // Keep the newest message in view.
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length, busy])

  /** Send a trimmed, non-empty question and clear the box. */
  const submit = (q: string) => {
    const question = q.trim()
    if (!question || busy) return
    onAsk(question)
    setDraft('')
  }

  return (
    <section className="ask" aria-label="Ask the episode">
      <div className="ask__list" ref={listRef} aria-live="polite">
        {messages.length === 0 ? (
          <div className="ask__intro">
            <p className="ask__intro-lead">
              Ask anything about this episode. Answers come only from the transcript, with timestamps you can play.
            </p>
            <div className="ask__suggestions">
              {suggestions.map((s) => (
                <button key={s} type="button" className="ask__suggestion" onClick={() => submit(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="ask__msg ask__msg--user">
                {m.content}
              </div>
            ) : m.role === 'error' ? (
              <div key={m.id} className="ask__msg ask__msg--error" role="alert">
                {m.content}
              </div>
            ) : (
              <div key={m.id} className={`ask__msg ask__msg--bot${m.found ? '' : ' ask__msg--notfound'}`}>
                <p className="ask__answer">{m.content}</p>
                {m.citations.length > 0 ? (
                  <ul className="ask__cites" aria-label="Sources in the episode">
                    {m.citations.map((c) => (
                      <li key={c.start_sec}>
                        <button type="button" className="ask__cite" onClick={() => onCite(c.start_sec, c.quote)} title="Play this moment">
                          <span className="ask__cite-time">▶ {formatTimestamp(c.start_sec)}</span>
                          <span className="ask__cite-quote">“{c.quote}”</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
                {m.unverified ? (
                  <p className="ask__warn">Could not match this answer to a moment in the transcript. Treat with care.</p>
                ) : null}
              </div>
            )
          )
        )}
        {busy ? (
          <div className="ask__msg ask__msg--bot ask__msg--pending" aria-label="Answering">
            <span className="ask__dots" aria-hidden>
              <i />
              <i />
              <i />
            </span>
          </div>
        ) : null}
      </div>

      <form
        className="ask__form"
        onSubmit={(e) => {
          e.preventDefault()
          submit(draft)
        }}
      >
        <textarea
          className="ask__input"
          value={draft}
          maxLength={MAX_LEN}
          rows={2}
          placeholder="Ask about this episode…"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit(draft)
            }
          }}
          aria-label="Question about this episode"
        />
        <div className="ask__actions">
          {messages.length > 0 && !busy ? (
            <button type="button" className="ask__clear" onClick={onClear}>
              Clear
            </button>
          ) : (
            <span />
          )}
          {busy ? (
            <button type="button" className="btn btn--ghost btn--sm" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="btn btn--primary btn--sm" disabled={!draft.trim()}>
              Ask
            </button>
          )}
        </div>
      </form>
    </section>
  )
}
