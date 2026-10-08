// PARKED (future scope): not imported by the app. See ./README.md for how to re-enable live mode.
/** Placeholder transcript used when a live chunk cannot be transcribed. */

import type { TranscriptSegment } from '../../types/transcript'

/** Shown when a live chunk request fails (network / server error). */
export const UNKNOWN_TRANSCRIPT_SENTENCE = 'Unknown Sentence'

/**
 * Placeholder segment covering a failed live chunk, so the session timeline has no gap.
 * Ids are `chunkIndex * 1000` to stay clear of real segment ids from the same chunk.
 */
export function unknownLiveTranscriptSegments(
  chunkIndex: number,
  windowSec: number
): TranscriptSegment[] {
  const base = chunkIndex * 1000
  const off = chunkIndex * windowSec
  return [{ id: base, start: off, end: off + windowSec, text: UNKNOWN_TRANSCRIPT_SENTENCE }]
}
