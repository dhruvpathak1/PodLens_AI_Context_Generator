/** Shapes of the static demo files written by `npm run demo:build` (server/app/services/demo_bundle.py). */

import type { AskResponse } from './ask'
import type { EnrichedEntityCard, EntityDocument } from './entities'
import type { EpisodeTimelineData } from './timeline'
import type { TranscriptSegment } from './transcript'

/** `episode.json` schema version this build understands. */
export const DEMO_SCHEMA_VERSION = 1

/** One entry of `public/demo/index.json` (shown in the episode picker). */
export type DemoEpisodeSummary = {
  slug: string
  title: string
  description: string
  /** Attribution for the audio, e.g. "Audio: NASA (public domain)". */
  credit: string
  duration_sec: number
  entity_count: number
  has_timeline: boolean
  question_count: number
}

export type DemoIndex = {
  schema_version: number
  episodes: DemoEpisodeSummary[]
}

/** A pre-computed answer for one suggested question. */
export type DemoQA = { question: string; response: AskResponse }

/** `public/demo/<slug>/episode.json`: everything the API would have returned for the episode. */
export type DemoEpisode = {
  schema_version: number
  slug: string
  title: string
  description: string
  credit: string
  /** Audio file name, relative to the episode folder. */
  audio: string
  duration_sec: number
  generated_at: string
  generated_with: {
    whisper_model: string
    entity_backend: string | null
    timeline_model: string | null
    ask_model: string | null
  }
  transcript: string
  segments: TranscriptSegment[]
  document: EntityDocument
  cards: EnrichedEntityCard[]
  unsplash_enabled: boolean
  /** Null when the demo was built without an OpenAI key. */
  timeline: EpisodeTimelineData | null
  qa: DemoQA[]
}
