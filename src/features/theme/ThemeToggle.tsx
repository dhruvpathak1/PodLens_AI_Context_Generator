import { MoonIcon, SunIcon } from '../../shared/icons'
import { useTheme } from './useTheme'

/** Icon button that switches between light and dark themes. */
export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const toLight = theme === 'dark'
  return (
    <button
      type="button"
      className="icon-btn theme-toggle"
      onClick={toggleTheme}
      aria-label={toLight ? 'Switch to light theme' : 'Switch to dark theme'}
      title={toLight ? 'Light theme' : 'Dark theme'}
    >
      {toLight ? <SunIcon /> : <MoonIcon />}
    </button>
  )
}
