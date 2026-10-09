import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

/** localStorage key; also read by the inline script in index.html before first paint. */
const STORAGE_KEY = 'podlens-theme'

/** Starting theme: whatever index.html already applied (saved choice), else dark. */
function initialTheme(): Theme {
  const fromDom = document.documentElement.dataset.theme
  return fromDom === 'light' ? 'light' : 'dark'
}

/**
 * Light/dark theme state. Writes `data-theme` on `<html>` (CSS tokens switch on it) and
 * remembers the user's choice in localStorage.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  const toggleTheme = useCallback(() => {
    // Briefly enable color transitions so the switch cross-fades (see .theme-anim in styles/base.css).
    const root = document.documentElement
    root.classList.add('theme-anim')
    window.setTimeout(() => root.classList.remove('theme-anim'), 300)
    setTheme((current) => {
      const next: Theme = current === 'dark' ? 'light' : 'dark'
      try {
        localStorage.setItem(STORAGE_KEY, next)
      } catch {
        /* private mode / storage blocked: theme still applies for this session */
      }
      return next
    })
  }, [])

  return { theme, toggleTheme }
}
