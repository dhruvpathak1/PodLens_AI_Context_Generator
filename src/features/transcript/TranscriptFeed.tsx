import { formatClock } from '../../shared/format'
import { Skeleton } from '../../shared/components/Skeleton'
import type { TranscriptView } from './useTranscriptView'

type Props = {
  view: TranscriptView
  /** Show a loading skeleton instead of sentences. */
  busy: boolean
  hasTranscript: boolean
  hasError: boolean
}

/**
 * Scrollable list of transcript sentences. Each sentence is a button: clicking (or Enter/Space)
 * plays from that point. The sentence at the playhead is highlighted and kept in view.
 */
export function TranscriptFeed({ view, busy, hasTranscript, hasError }: Props) {
  const { feedRef, visibleSentences, selectedSentence, activeSentenceId, hasTimedSentences, searchQuery } = view

  return (
    <div ref={feedRef} className="transcript-feed" role="region" aria-label="Transcript sentences">
      {busy && <Skeleton className="skeleton--in-feed" />}

      {!busy &&
        visibleSentences.map((s) => {
          const isSelected = selectedSentence?.id === s.id
          const isPlayingHere = hasTimedSentences && activeSentenceId === s.id
          return (
            <button
              key={s.id}
              type="button"
              data-sentence-id={s.id}
              className={`transcript-sentence${isSelected ? ' transcript-sentence--selected' : ''}${
                isPlayingHere ? ' transcript-sentence--playing' : ''
              }`}
              onClick={() => view.selectSentence(s)}
              aria-pressed={isSelected}
              aria-current={isPlayingHere ? 'location' : undefined}
            >
              <time
                className="transcript-sentence__time"
                dateTime={hasTimedSentences ? `PT${Math.floor(s.start)}S` : undefined}
              >
                {hasTimedSentences ? formatClock(s.start) : ''}
              </time>
              <span className="transcript-sentence__text">{s.text}</span>
            </button>
          )
        })}

      {!busy && !hasTranscript && !hasError && (
        <p className="empty">The transcript appears here, line by line, and follows the audio as it plays.</p>
      )}

      {!busy && hasTranscript && visibleSentences.length === 0 && searchQuery && (
        <p className="empty">No sentences match this filter.</p>
      )}
    </div>
  )
}
