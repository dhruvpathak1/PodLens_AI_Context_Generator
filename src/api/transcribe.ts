/** `POST /api/transcribe`: upload audio, receive transcript segments and tagged entities. */

import { apiUrl } from '../config/env'
import type { EntityDocument } from '../types/entities'
import type { TranscriptSegment } from '../types/transcript'
import { readErrorDetail } from './http'

export type TranscribeOptions = {
  /** ISO language code; skips Whisper's language detection. */
  language?: string
  /** Run entity extraction after Whisper (default true). */
  extractEntities?: boolean
  /** NER backend override, sent as `entity_backend`. */
  backend?: 'spacy' | 'claude'
}

/** Normalized result of a transcription request (also used by the parked live mode). */
export type TranscribeResult = {
  transcript: string
  segments: TranscriptSegment[]
  /** Where the server saved the transcript text file. */
  savedPath?: string
  document?: EntityDocument | null
  entitySavedPath?: string | null
  /** Set when transcription worked but entity tagging failed. */
  entityError?: string | null
}

const isRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object'
const nonEmptyString = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v.trim() : null)

/** Keep only well-formed segments (non-empty text, finite times). */
function parseSegments(raw: unknown): TranscriptSegment[] {
  if (!Array.isArray(raw)) return []
  const out: TranscriptSegment[] = []
  raw.forEach((item, i) => {
    if (!isRecord(item)) return
    const text = typeof item.text === 'string' ? item.text.trim() : ''
    const start = Number(item.start)
    const end = Number(item.end)
    if (!text || !Number.isFinite(start) || !Number.isFinite(end)) return
    const id = typeof item.id === 'number' && Number.isFinite(item.id) ? item.id : i
    out.push({ id, start, end, text })
  })
  return out
}

/** Transcript text: `transcript`, else `text`, else segment texts joined. */
function parseTranscriptText(data: Record<string, unknown>): string | null {
  if (typeof data.transcript === 'string') return data.transcript
  if (typeof data.text === 'string') return data.text
  const joined = parseSegments(data.segments).map((s) => s.text).join(' ')
  return joined || null
}

/**
 * Convert the server's JSON into a `TranscribeResult`.
 * @throws Error when no transcript text can be found.
 */
export function parseTranscribeResponse(data: unknown): TranscribeResult {
  if (!isRecord(data)) throw new Error('Response JSON did not include transcript text')
  const text = parseTranscriptText(data)
  if (text == null) throw new Error('Response JSON did not include transcript text')
  const doc = data.document
  return {
    transcript: text.trim(),
    segments: parseSegments(data.segments),
    savedPath: nonEmptyString(data.saved_path) ?? undefined,
    document: isRecord(doc) && Array.isArray(doc.entities) ? (doc as EntityDocument) : null,
    entitySavedPath: nonEmptyString(data.entity_saved_path),
    entityError: nonEmptyString(data.entity_error),
  }
}

/**
 * Read a transcription response: JSON (normal) or plain text (legacy servers).
 * @throws Error with the server's message for non-2xx responses.
 */
export async function readTranscribeResponse(res: Response): Promise<TranscribeResult> {
  if (!res.ok) throw new Error(await readErrorDetail(res, `Request failed (${res.status})`))
  if ((res.headers.get('content-type') ?? '').includes('application/json')) {
    return parseTranscribeResponse(await res.json())
  }
  const text = (await res.text()).trim()
  if (!text) throw new Error('Empty transcript response')
  return { transcript: text, segments: [] }
}

/** Upload an audio file for transcription and entity tagging. */
export async function transcribeAudio(file: File, options: TranscribeOptions = {}): Promise<TranscribeResult> {
  const body = new FormData()
  body.append('audio', file, file.name)
  if (options.language) body.append('language', options.language)
  body.append('extract_entities', options.extractEntities === false ? 'false' : 'true')
  if (options.backend) body.append('entity_backend', options.backend)

  const res = await fetch(apiUrl('/api/transcribe'), { method: 'POST', body })
  return readTranscribeResponse(res)
}
