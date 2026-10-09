import { useMemo, useRef, type KeyboardEvent, type PointerEvent } from 'react'
import { formatClock } from '../../shared/format'

type Props = {
  /** Loudness per bar (0..1); null while loading, empty when unavailable. */
  peaks: number[] | null
  /** Number of bars to draw when `peaks` is not ready. */
  bars: number
  current: number
  duration: number
  /** Move playback to `sec` (does not start playing). */
  onSeek: (sec: number) => void
}

/** Gentle fixed pattern shown while the real waveform loads or if it cannot be decoded. */
function placeholder(bars: number): number[] {
  return Array.from({ length: bars }, (_, i) => 0.25 + 0.2 * Math.abs(Math.sin(i * 0.7)) + 0.1 * Math.abs(Math.sin(i * 2.3)))
}

/**
 * Waveform scrubber ("sound bar"): bars before the playhead are lit. Click or drag to seek;
 * keyboard: arrows move 5 s, Page Up/Down 30 s, Home/End jump to the ends.
 */
export function Waveform({ peaks, bars, current, duration, onSeek }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  const heights = useMemo(() => (peaks && peaks.length ? peaks : placeholder(bars)), [peaks, bars])
  const progress = duration > 0 ? Math.min(1, current / duration) : 0
  const ready = duration > 0

  /** Convert a pointer x position to a time and seek there. */
  const seekFromPointer = (e: PointerEvent<HTMLDivElement>) => {
    const el = ref.current
    if (!el || !ready) return
    const rect = el.getBoundingClientRect()
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    onSeek(ratio * duration)
  }

  /** Keyboard seeking for the slider. */
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!ready) return
    const steps: Record<string, number> = { ArrowRight: 5, ArrowUp: 5, ArrowLeft: -5, ArrowDown: -5, PageUp: 30, PageDown: -30 }
    if (e.key in steps) onSeek(Math.min(duration, Math.max(0, current + steps[e.key])))
    else if (e.key === 'Home') onSeek(0)
    else if (e.key === 'End') onSeek(duration)
    else return
    e.preventDefault()
  }

  return (
    <div
      ref={ref}
      className={`wave${peaks === null ? ' wave--loading' : ''}`}
      role="slider"
      tabIndex={ready ? 0 : -1}
      aria-label="Seek in audio"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(current)}
      aria-valuetext={`${formatClock(current)} of ${formatClock(duration)}`}
      aria-disabled={!ready}
      onPointerDown={(e) => {
        dragging.current = true
        e.currentTarget.setPointerCapture(e.pointerId)
        seekFromPointer(e)
      }}
      onPointerMove={(e) => {
        if (dragging.current) seekFromPointer(e)
      }}
      onPointerUp={() => {
        dragging.current = false
      }}
      onKeyDown={onKeyDown}
    >
      {heights.map((h, i) => (
        <span
          key={i}
          className={`wave__bar${(i + 0.5) / heights.length <= progress ? ' wave__bar--on' : ''}`}
          style={{ height: `${Math.round(h * 100)}%` }}
        />
      ))}
    </div>
  )
}
