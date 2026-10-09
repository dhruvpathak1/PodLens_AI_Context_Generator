import { useEffect, useRef, useState } from 'react'
import { Panel } from '../../shared/components/Panel'
import { formatClock } from '../../shared/format'
import { ArrowUpIcon, StopIcon } from '../../shared/icons'
import type { ChatMessage } from './useEpisodeChat'

type Props = {
  messages: ChatMessage[]
  busy: boolean
  /** False until an episode is loaded; the box is disabled with an explanation. */
  enabled: boolean
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
 * Right column, bottom: "Ask the episode". Suggested questions, the conversation (answers with
 * clickable citations), and the question box pinned to the bottom (Enter sends, Shift+Enter
 * adds a line).
 */
export function AskEpisode({ messages, busy, enabled, suggestions, onAsk, onStop, onClear, onCite }: Props) {
  const [draft, setDraft] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  // Keep the newest message in view.
  useEffect(() => {
    const el = listRef.current?.closest('.panel__body')
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length, busy])

  /** Send a trimmed, non-empty question and clear the box. */
  const submit = (q: string) => {
    const question = q.trim()
    if (!question || busy || !enabled) return
    onAsk(question)
    setDraft('')
  }

  return (
    <Panel
      className="panel--chat"
      title="Ask the episode"
      actions={
        messages.length > 0 && !busy ? (
          <button type="button" className="link-quiet" onClick={onClear}>
            Clear
          </button>
        ) : null
      }
      footer={
        <form
          className="ask-bar"
          onSubmit={(e) => {
            e.preventDefault()
            submit(draft)
          }}
        >
          <textarea
            className="ask-bar__input"
            value={draft}
            maxLength={MAX_LEN}
            rows={1}
            disabled={!enabled}
            placeholder={enabled ? 'Ask a question' : 'Load an episode to ask questions'}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit(draft)
              }
            }}
            aria-label="Question about this episode"
          />
          {busy ? (
            <button type="button" className="ask-bar__send" onClick={onStop} aria-label="Stop answering">
              <StopIcon />
            </button>
          ) : (
            <button type="submit" className="ask-bar__send" disabled={!draft.trim() || !enabled} aria-label="Send question">
              <ArrowUpIcon />
            </button>
          )}
        </form>
      }
    >
      <div className="ask" ref={listRef} aria-live="polite">
        {messages.length === 0 ? (
          enabled ? (
            <div className="ask__intro">
              <p className="empty">Answers come only from the transcript, with timestamps you can play.</p>
              <div className="ask__suggestions">
                {suggestions.map((s) => (
                  <button key={s} type="button" className="ask__suggestion" onClick={() => submit(s)} disabled={busy}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <p className="empty">Questions about the episode are answered here, with the moments they come from.</p>
          )
        ) : (
          messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.id} className="msg msg--user">
                {m.content}
              </div>
            ) : m.role === 'error' ? (
              <div key={m.id} className="msg msg--error" role="alert">
                {m.content}
              </div>
            ) : (
              <div key={m.id} className={`msg msg--bot${m.found ? '' : ' msg--notfound'}`}>
                <p className="msg__text">{m.content}</p>
                {m.citations.length > 0 ? (
                  <ul className="cites" aria-label="Moments in the episode">
                    {m.citations.map((c) => (
                      <li key={c.start_sec}>
                        <button type="button" className="cite" onClick={() => onCite(c.start_sec, c.quote)} title="Play this moment">
                          <span className="cite__time num">{formatClock(c.start_sec)}</span>
                          <span className="cite__quote">{c.quote}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
                {m.unverified ? (
                  <p className="msg__warn">This answer could not be matched to a moment in the transcript.</p>
                ) : null}
              </div>
            )
          )
        )}
        {busy ? (
          <div className="msg msg--bot msg--pending" aria-label="Answering">
            <span className="dots" aria-hidden>
              <i />
              <i />
              <i />
            </span>
          </div>
        ) : null}
      </div>
    </Panel>
  )
}
