import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { TimestampedSentence, TranscriptSegment } from '../../types/transcript'
import { buildSentencesFromTranscript, findSentenceForCitation, getActiveSentenceIdAtTime } from './sentences'

type Params = {
  transcript: string | null
  segments: TranscriptSegment[]
  playbackTime: number
  playbackDuration: number
  /** Resets auto-scroll tracking when a different file is loaded. */
  audioUrl: string | null
  seekTo: (sec: number) => void
}

/**
 * Everything the transcript sidebar needs: sentences, search filter, the selected sentence,
 * the sentence currently playing (auto-scrolled into view), and click/citation handlers.
 */
export function useTranscriptView({ transcript, segments, playbackTime, playbackDuration, audioUrl, seekTo }: Params) {
  const feedRef = useRef<HTMLDivElement>(null)
  const lastScrolledIdRef = useRef<string | null>(null)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSentence, setSelectedSentence] = useState<TimestampedSentence | null>(null)

  const allSentences = useMemo(() => buildSentencesFromTranscript(transcript, segments), [transcript, segments])

  /** Sentences matching the search box (all of them when it is empty). */
  const visibleSentences = useMemo(() => {
    const q = searchQuery.trim().toLowerCase()
    return q ? allSentences.filter((s) => s.text.toLowerCase().includes(q)) : allSentences
  }, [allSentences, searchQuery])

  /** False when segments had no timing, so timestamps should be hidden. */
  const hasTimedSentences = allSentences.some((s) => s.start > 0 || s.end > 0)

  const activeSentenceId = useMemo(
    () => (hasTimedSentences ? getActiveSentenceIdAtTime(allSentences, playbackTime, playbackDuration) : null),
    [hasTimedSentences, allSentences, playbackTime, playbackDuration]
  )

  // New transcript or file: allow the first active sentence to scroll into view again.
  useEffect(() => {
    lastScrolledIdRef.current = null
  }, [transcript, audioUrl])

  // Keep the playing sentence visible as playback moves on.
  useEffect(() => {
    if (!activeSentenceId || lastScrolledIdRef.current === activeSentenceId) return
    lastScrolledIdRef.current = activeSentenceId
    feedRef.current
      ?.querySelector<HTMLElement>(`[data-sentence-id="${CSS.escape(activeSentenceId)}"]`)
      ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [activeSentenceId])

  /** Open/close the search box; closing clears the filter. */
  const toggleSearch = useCallback(() => {
    if (searchOpen) setSearchQuery('')
    setSearchOpen(!searchOpen)
  }, [searchOpen])

  /** Click on a sentence: highlight it and play from its start. */
  const selectSentence = useCallback(
    (s: TimestampedSentence) => {
      setSelectedSentence(s)
      seekTo(s.start)
    },
    [seekTo]
  )

  /** Ask citation click: play from the cited moment and highlight the matching sentence. */
  const jumpToCitation = useCallback(
    (sec: number, quote: string) => {
      const hit = findSentenceForCitation(allSentences, sec, quote)
      if (hit) setSelectedSentence(hit)
      seekTo(sec)
    },
    [allSentences, seekTo]
  )

  /** Clear the highlighted sentence (new run). */
  const clearSelection = useCallback(() => setSelectedSentence(null), [])

  return {
    feedRef,
    allSentences,
    visibleSentences,
    hasTimedSentences,
    activeSentenceId,
    selectedSentence,
    searchOpen,
    searchQuery,
    setSearchQuery,
    toggleSearch,
    selectSentence,
    jumpToCitation,
    clearSelection,
  }
}

export type TranscriptView = ReturnType<typeof useTranscriptView>
