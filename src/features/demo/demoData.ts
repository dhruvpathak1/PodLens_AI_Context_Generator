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

/** Words too common to tell questions apart. */
const STOP_WORDS = new Set(
  'a an and are about did does do episode for from how in is it mentioned of on said say says the they this to was were what when where which who why with'.split(' ')
)

/** Meaningful words of a question, for loose matching. */
function keywords(q: string): Set<string> {
  return new Set(normalizeQuestion(q).split(' ').filter((w) => w.length > 1 && !STOP_WORDS.has(w)))
}

/** Share of the stored question's keywords that the visitor's question also uses (0..1). */
function overlap(asked: Set<string>, stored: Set<string>): number {
  if (!stored.size) return 0
  let hits = 0
  for (const w of stored) if (asked.has(w)) hits++
  return hits / stored.size
}

/** Minimum keyword overlap for a typed question to count as a stored one. */
const MATCH_THRESHOLD = 0.6

/**
 * Ask request in demo mode. Exact or loose matches return the stored answer
 * ("who was joseph strauss" finds "Who is Joseph Strauss and why are they mentioned?").
 * Otherwise the reply explains the limit and offers the stored questions as follow-ups.
 */
export async function demoAnswer(episode: DemoEpisode, question: string): Promise<AskResponse> {
  await settle(600)
  const key = normalizeQuestion(question)
  const exact = episode.qa.find((qa) => normalizeQuestion(qa.question) === key)
  if (exact) return exact.response

  const asked = keywords(question)
  let best: { score: number; response: AskResponse } | null = null
  for (const qa of episode.qa) {
    const score = overlap(asked, keywords(qa.question))
    if (score >= MATCH_THRESHOLD && (!best || score > best.score)) best = { score, response: qa.response }
  }
  if (best) return best.response

  return {
    answer: episode.qa.length
      ? 'This public demo has answers for a few questions only. Try one of these, or run PodLens locally with your own OpenAI key to ask anything.'
      : 'This sample was published without answers. Run PodLens locally with your own OpenAI key to ask questions.',
    found: false,
    citations: [],
    mode: 'full',
    unverified: false,
    model: 'demo',
    followups: episode.qa.map((qa) => qa.question),
  }
}
