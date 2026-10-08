/** Turn Whisper output into complete, timestamped sentences for the transcript sidebar. */

import type { TimestampedSentence, TranscriptSegment } from '../../types/transcript'

/** True if the string ends with a sentence terminator (closing quotes/brackets allowed after it). */
function endsWithSentenceBoundary(s: string): boolean {
  return /[.!?…]["'»”’)\]]*\s*$/.test(s.trimEnd())
}

/**
 * Merge fragments until each one ends at a real sentence boundary.
 * Fixes splitters that cut after abbreviations ("Dr.") or mid-phrase without punctuation.
 */
function mergeToCompleteSentences(parts: string[]): string[] {
  if (parts.length <= 1) return parts
  const out: string[] = []
  for (const p of parts) {
    const t = p.trim()
    if (!t) continue
    const last = out[out.length - 1]
    if (last !== undefined && !endsWithSentenceBoundary(last)) out[out.length - 1] = `${last} ${t}`.trim()
    else out.push(t)
  }
  return out
}

/** Split text into sentences, using `Intl.Segmenter` when available and a regex otherwise. */
export function splitSentencesFromText(text: string): string[] {
  const t = text.trim()
  if (!t) return []

  let parts: string[] = []
  if (typeof Intl !== 'undefined' && 'Segmenter' in Intl) {
    try {
      const segmenter = new Intl.Segmenter(undefined, { granularity: 'sentence' })
      parts = [...segmenter.segment(t)].map((s) => s.segment.trim()).filter(Boolean)
    } catch {
      parts = []
    }
  }
  if (!parts.length) parts = t.split(/(?<=[.!?…])\s+/u).map((s) => s.trim()).filter(Boolean)
  if (!parts.length) return [t]
  return mergeToCompleteSentences(parts)
}

/**
 * Split the full transcript into sentences and give each an approximate time range.
 *
 * Whisper segments often break mid-sentence, so sentences are split from the full text and
 * their times are interpolated by character position across the whole audio span.
 * Known limitation: on long episodes this drifts from the real segment times.
 */
export function buildSentencesFromTranscript(
  transcript: string | null,
  segments: TranscriptSegment[]
): TimestampedSentence[] {
  if (!transcript?.trim()) return []
  const text = transcript.trim()
  const sentences = splitSentencesFromText(text)
  if (!sentences.length) return []
  if (!segments.length) return sentences.map((s, i) => ({ id: `s-${i}`, text: s, start: 0, end: 0 }))

  const t0 = segments[0].start
  const duration = Math.max(segments[segments.length - 1].end - t0, 1e-9)
  const totalLen = Math.max(text.length, 1)

  let searchFrom = 0
  return sentences.map((s, i) => {
    // Locate the sentence in the full text (skipping whitespace the splitter trimmed).
    let startChar = text.indexOf(s, searchFrom)
    if (startChar < 0) {
      while (searchFrom < text.length && /\s/.test(text[searchFrom])) searchFrom++
      startChar = text.indexOf(s, searchFrom)
    }
    if (startChar < 0) startChar = searchFrom
    const endChar = Math.min(startChar + s.length, text.length)
    searchFrom = endChar
    return {
      id: `s-${i}`,
      text: s,
      start: t0 + (startChar / totalLen) * duration,
      end: t0 + (endChar / totalLen) * duration,
    }
  })
}

/** Id of the sentence playing at `currentTime` (each sentence lasts until the next one starts). */
export function getActiveSentenceIdAtTime(
  sentences: TimestampedSentence[],
  currentTime: number,
  trackDuration: number
): string | null {
  if (!sentences.length) return null
  const t = Math.max(0, currentTime)
  const n = sentences.length
  const end =
    Number.isFinite(trackDuration) && trackDuration > 0
      ? trackDuration
      : Math.max(sentences[n - 1].end, sentences[n - 1].start, 0)

  for (let i = 0; i < n; i++) {
    const s = sentences[i]
    if (i + 1 < n) {
      if (t >= s.start && t < sentences[i + 1].start) return s.id
    } else if (t >= s.start && t <= end) {
      return s.id
    }
  }
  return t < sentences[0].start ? sentences[0].id : sentences[n - 1].id
}

/** Lower-case, alphanumeric-only text used to match a quoted citation to a sentence. */
function normalizeForMatch(t: string): string {
  return t.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim()
}

/**
 * Find the sentence a citation points to: match its first words of `quote` first (sentence
 * times are estimates), then fall back to the sentence that starts closest to `sec`.
 */
export function findSentenceForCitation(
  sentences: TimestampedSentence[],
  sec: number,
  quote: string
): TimestampedSentence | null {
  const probe = normalizeForMatch(quote).split(' ').slice(0, 6).join(' ')
  const byQuote = probe.length >= 8 ? sentences.find((s) => normalizeForMatch(s.text).includes(probe)) : undefined
  if (byQuote) return byQuote
  return sentences.reduce<TimestampedSentence | null>(
    (best, s) => (!best || Math.abs(s.start - sec) < Math.abs(best.start - sec) ? s : best),
    null
  )
}
