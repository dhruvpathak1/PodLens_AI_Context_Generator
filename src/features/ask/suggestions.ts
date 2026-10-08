/** Starter questions for the Ask tab, built from the episode's most-mentioned entities. */

import type { EntityRecord } from '../../types/entities'

/**
 * Up to four suggestions: a summary prompt, plus one each for the top person, the top
 * company/technology and the top place/event (ranked by number of mentions).
 */
export function suggestQuestions(entities: EntityRecord[]): string[] {
  const freq = new Map<string, { type: string; text: string; n: number }>()
  for (const e of entities) {
    const key = `${e.type}\0${e.text.trim().toLowerCase()}`
    const cur = freq.get(key)
    if (cur) cur.n++
    else freq.set(key, { type: e.type, text: e.text.trim(), n: 1 })
  }
  const ranked = [...freq.values()].sort((a, b) => b.n - a.n)
  const pick = (types: string[]) => ranked.find((r) => types.includes(r.type))
  const out: string[] = ['Summarize this episode in 3 points.']
  const person = pick(['PERSON'])
  if (person) out.push(`Who is ${person.text} and why are they mentioned?`)
  const org = pick(['COMPANY', 'TECHNOLOGY'])
  if (org) out.push(`What is said about ${org.text}?`)
  const place = pick(['PLACE', 'EVENT'])
  if (place && out.length < 4) out.push(`What happens around ${place.text}?`)
  return out.slice(0, 4)
}
