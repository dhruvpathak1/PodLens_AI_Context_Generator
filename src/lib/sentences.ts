import type { TranscriptSegment } from './transcribeAudio'

export type TimestampedSentence = {
  id: string
  text: string
  start: number
  end: number
}

/** True if the string ends with a sentence terminator (allows quotes after). */
function endsWithSentenceBoundary(s: string): boolean {
  return /[.!?…]["'»”’)\]]*\s*$/.test(s.trimEnd())
}

/**
 * Merge raw segments so each row ends at a full sentence boundary when possible.
 * Handles splitters that cut after "Dr." or mid-phrase without punctuation.
 */
function mergeToCompleteSentences(parts: string[]): string[] {
  if (parts.length <= 1) return parts
  const out: string[] = []
  for (const p of parts) {
    const t = p.trim()
    if (!t) continue
    const last = out[out.length - 1]
    if (last !== undefined && !endsWithSentenceBoundary(last)) {
      out[out.length - 1] = `${last} ${t}`.trim()
    } else {
      out.push(t)
    }
  }
  return out
}

/** Split on sentence boundaries (Intl.Segmenter when available). */
export function splitSentencesFromText(text: string): string[] {
  const t = text.trim()
  if (!t) return []

  let parts: string[] = []

  if (typeof Intl !== 'undefined' && 'Segmenter' in Intl) {
    try {
      const seg = new Intl.Segmenter(undefined, { granularity: 'sentence' })
      parts = [...seg.segment(t)]
        .map((s) => s.segment.trim())
        .filter(Boolean)
    } catch {
      parts = []
    }
  }

  if (!parts.length) {
    parts = t.split(/(?<=[.!?…])\s+/u).map((s) => s.trim()).filter(Boolean)
  }

  if (!parts.length) return [t]

  return mergeToCompleteSentences(parts)
}

/**
 * Split the full transcript into complete sentences, then map each sentence to an
 * approximate time range from the overall audio span (avoids cutting mid-sentence at
 * Whisper segment boundaries).
 */
export function buildSentencesFromTranscript(
  transcript: string | null,
  segments: TranscriptSegment[]
): TimestampedSentence[] {
  if (!transcript?.trim()) return []

  const text = transcript.trim()
  const sentences = splitSentencesFromText(text)
  if (!sentences.length) return []

  if (!segments.length) {
    return sentences.map((s, i) => ({ id: `s-${i}`, text: s, start: 0, end: 0 }))
  }

  const t0 = segments[0].start
  const t1 = segments[segments.length - 1].end
  const dur = Math.max(t1 - t0, 1e-9)
  const totalLen = Math.max(text.length, 1)

  let searchFrom = 0
  const out: TimestampedSentence[] = []

  for (let i = 0; i < sentences.length; i++) {
    const s = sentences[i]
    let startChar = text.indexOf(s, searchFrom)
    if (startChar < 0) {
      while (searchFrom < text.length && /\s/.test(text[searchFrom])) searchFrom++
      startChar = text.indexOf(s, searchFrom)
    }
    if (startChar < 0) {
      startChar = searchFrom
    }
    const endChar = Math.min(startChar + s.length, text.length)
    searchFrom = endChar

    const start = t0 + (startChar / totalLen) * dur
    const end = t0 + (endChar / totalLen) * dur
    out.push({ id: `s-${i}`, text: s, start, end })
  }

  return out
}

/** Sentence active at `currentTime` (seconds), using next sentence start as the end of each block. */
export function getActiveSentenceIdAtTime(
  sentences: TimestampedSentence[],
  currentTime: number,
  trackDuration: number
): string | null {
  if (!sentences.length) return null
  const t = Math.max(0, currentTime)
  const n = sentences.length
  const d =
    Number.isFinite(trackDuration) && trackDuration > 0
      ? trackDuration
      : Math.max(sentences[n - 1].end, sentences[n - 1].start, 0)

  for (let i = 0; i < n; i++) {
    const s = sentences[i]
    if (i + 1 < n) {
      const nextStart = sentences[i + 1].start
      if (t >= s.start && t < nextStart) return s.id
    } else if (t >= s.start && t <= d) {
      return s.id
    }
  }

  if (t < sentences[0].start) return sentences[0].id
  return sentences[n - 1].id
}
