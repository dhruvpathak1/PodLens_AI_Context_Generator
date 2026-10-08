/** Transcript data shared across the app. */

/** One Whisper segment: a short span of speech with start/end times in seconds. */
export type TranscriptSegment = {
  id: number
  start: number
  end: number
  text: string
}

/** A full sentence shown in the transcript sidebar, with an estimated time range. */
export type TimestampedSentence = {
  id: string
  text: string
  start: number
  end: number
}
