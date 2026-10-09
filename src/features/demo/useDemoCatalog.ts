import { useCallback, useEffect, useRef, useState } from 'react'
import type { DemoEpisode, DemoEpisodeSummary } from '../../types/demo'
import { loadDemoEpisode, loadDemoIndex } from './demoData'

/**
 * Demo episode picker state: loads `index.json` once (when `enabled`), then loads an episode
 * bundle on `select(slug)` and hands it to `onLoaded`. The first episode is opened automatically
 * so visitors land on a populated dashboard.
 */
export function useDemoCatalog(enabled: boolean, onLoaded: (episode: DemoEpisode) => void) {
  const [episodes, setEpisodes] = useState<DemoEpisodeSummary[]>([])
  const [indexLoaded, setIndexLoaded] = useState(!enabled)
  const [activeSlug, setActiveSlug] = useState<string | null>(null)
  const [loadingSlug, setLoadingSlug] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const onLoadedRef = useRef(onLoaded)
  useEffect(() => {
    onLoadedRef.current = onLoaded
  }, [onLoaded])
  /** Slug of the most recent request; older responses are ignored if the user clicks quickly. */
  const latestRef = useRef<string | null>(null)

  /** Load one episode bundle and pass it to `onLoaded`. */
  const select = useCallback(async (slug: string) => {
    latestRef.current = slug
    setLoadingSlug(slug)
    setError(null)
    try {
      const episode = await loadDemoEpisode(slug)
      if (latestRef.current !== slug) return
      setActiveSlug(slug)
      onLoadedRef.current(episode)
    } catch (e) {
      if (latestRef.current === slug) setError(e instanceof Error ? e.message : 'Could not load the demo episode')
    } finally {
      if (latestRef.current === slug) setLoadingSlug(null)
    }
  }, [])

  // Load the catalog once, then open the first episode.
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    void loadDemoIndex().then((index) => {
      if (cancelled) return
      setEpisodes(index.episodes)
      setIndexLoaded(true)
      if (index.episodes[0]) void select(index.episodes[0].slug)
    })
    return () => {
      cancelled = true
    }
  }, [enabled, select])

  return { episodes, indexLoaded, activeSlug, loadingSlug, error, select }
}

export type DemoCatalog = ReturnType<typeof useDemoCatalog>
