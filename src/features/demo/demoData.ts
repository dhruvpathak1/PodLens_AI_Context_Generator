/**
 * Read-only data access for demo mode: loads the static files in `public/demo/` and answers
 * timeline / Ask requests from the pre-computed results instead of calling the LLM.
 */

import { demoUrl } from '../../config/env'
import type { AskResponse } from '../../types/ask'
import { DEMO_SCHEMA_VERSION, type DemoEpisode, type DemoIndex } from '../../types/demo'
import type { EpisodeTimelineData } from '../../types/timeline'

/** Fetch and parse a JSON file under `public/demo/`, with a readable error on failure. */
async function fetchDemoJson<T>(path: string): Promise<T> {
  const res = await fetch(demoUrl(path))
  if (!res.ok) throw new Error(`Could not load demo data (${res.status} for ${path})`)
  return (await res.json()) as T
}

/** List of demo episodes. An empty list means no demo has been built yet. */
export async function loadDemoIndex(): Promise<DemoIndex> {
  try {
    const index = await fetchDemoJson<DemoIndex>('index.json')
    return { schema_version: index.schema_version, episodes: Array.isArray(index.episodes) ? index.episodes : [] }
  } catch {
    return { schema_version: DEMO_SCHEMA_VERSION, episodes: [] }
  }
}

/** Load one episode bundle; rejects bundles written by an incompatible generator version. */
export async function loadDemoEpisode(slug: string): Promise<DemoEpisode> {
  const episode = await fetchDemoJson<DemoEpisode>(`${encodeURIComponent(slug)}/episode.json`)
  if (episode.schema_version !== DEMO_SCHEMA_VERSION) {
    throw new Error(`Demo "${slug}" uses schema v${episode.schema_version}; rebuild it with npm run demo:build.`)
  }
  return episode
}

/** Public URL of an episode's audio file. */
export function demoAudioUrl(episode: DemoEpisode): string {
  return demoUrl(`${encodeURIComponent(episode.slug)}/${episode.audio}`)
}

/** Short pause so pre-computed results still show the loading state (feels like the real app). */
const settle = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** Timeline request in demo mode: returns the pre-computed timeline. */
export async function demoTimeline(episode: DemoEpisode): Promise<EpisodeTimelineData> {
  await settle(700)
  if (!episode.timeline) throw new Error('This demo episode was built without a timeline.')
  return episode.timeline
}

/** Lowercase, drop punctuation and extra spaces, so "Who is X?" matches "who is x". */
export function normalizeQuestion(q: string): string {
  return q
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Answer shown for questions that were not pre-computed. */
const NOT_PRECOMPUTED: Omit<AskResponse, 'model'> = {
  answer:
    'This public demo only has answers for the suggested questions. Run PodLens locally with your own OpenAI key to ask anything about any episode.',
  found: false,
  citations: [],
  mode: 'full',
  unverified: false,
}

/** Ask request in demo mode: the stored answer for a suggested question, or an explanation. */
export async function demoAnswer(episode: DemoEpisode, question: string): Promise<AskResponse> {
  await settle(600)
  const key = normalizeQuestion(question)
  const hit = episode.qa.find((qa) => normalizeQuestion(qa.question) === key)
  return hit ? hit.response : { ...NOT_PRECOMPUTED, model: 'demo' }
}
