// PARKED (future scope): not imported by the app. See ./README.md for how to re-enable live mode.
/** `POST /api/transcribe-chunk`: transcribe one 10-second microphone slice (chunked live mode). */

import { readTranscribeResponse, type TranscribeOptions, type TranscribeResult } from '../../api/transcribe'
import { apiUrl } from '../../config/env'

export type LiveChunkTranscribeOptions = TranscribeOptions & {
  /** Monotonic chunk index; the server uses it for stable segment ids (`chunk_seq`). */
  chunkSeq: number
  /** Seconds added to Whisper segment times so chunks line up on one session timeline. */
  timeOffsetSec: number
  /** When true the server also writes transcript and entity JSON to disk (default false). */
  persistTranscript?: boolean
}

/** Upload one live slice (WebM) and return the same shape as a full-file transcription. */
export async function transcribeLiveAudioChunk(
  blob: Blob,
  options: LiveChunkTranscribeOptions
): Promise<TranscribeResult> {
  const body = new FormData()
  body.append('audio', blob, `live-chunk-${options.chunkSeq}.webm`)
  body.append('chunk_seq', String(options.chunkSeq))
  body.append('time_offset_sec', String(options.timeOffsetSec))
  body.append('extract_entities', options.extractEntities === false ? 'false' : 'true')
  body.append('persist_transcript', options.persistTranscript === true ? 'true' : 'false')
  if (options.language) body.append('language', options.language)
  if (options.backend) body.append('entity_backend', options.backend)

  const res = await fetch(apiUrl('/api/transcribe-chunk'), { method: 'POST', body })
  return readTranscribeResponse(res)
}
