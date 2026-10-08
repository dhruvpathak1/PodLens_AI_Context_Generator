import { formatTimeRange } from '../../shared/format'
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
    <div ref={feedRef} className="transcript-sidebar__feed" role="region" aria-label="Transcript sentences">
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
              <span className="transcript-sentence__meta">
                <span className="transcript-sentence__speaker">Sentence</span>
                <time
                  className="transcript-sentence__time"
                  dateTime={hasTimedSentences ? `PT${Math.floor(s.start)}S` : undefined}
                >
                  {hasTimedSentences ? formatTimeRange(s.start, s.end) : '–'}
                </time>
              </span>
              <span className="transcript-sentence__text">{s.text}</span>
            </button>
          )
        })}

      {!busy && !hasTranscript && !hasError && (
        <p className="transcript-sidebar__empty">
          Drop an audio file and run <strong>PodLens</strong> to get a timestamped transcript with context cards.
        </p>
      )}

      {!busy && hasTranscript && visibleSentences.length === 0 && searchQuery && (
        <p className="transcript-sidebar__empty">No sentences match this filter.</p>
      )}
    </div>
  )
}
