import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { formatClock } from '../../shared/format'
import { PauseIcon, PlayIcon } from '../../shared/icons'
import { useWaveformPeaks } from './useWaveformPeaks'
import { Waveform } from './Waveform'

/** Bars in the waveform scrubber. */
const WAVE_BARS = 64

/** Imperative API exposed to parents through a ref. */
export type MiniAudioPlayerHandle = {
  /** Jump to `seconds` and start playing (waits for metadata if the file is still loading). */
  seekTo: (seconds: number) => void
}

type Props = {
  /** Object URL of the current file, or null when no file is loaded. */
  src: string | null
  /** Called on every time update, seek, and when the duration becomes known. */
  onPlaybackTick?: (currentTime: number, duration: number) => void
  /** Text shown instead of the player when there is no `src`. */
  emptyHint?: string | null
}

/** Play/pause button, waveform scrubber and clock for the current episode. */
export const MiniAudioPlayer = forwardRef<MiniAudioPlayerHandle, Props>(function MiniAudioPlayer(
  { src, onPlaybackTick, emptyHint },
  ref
) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const tickCbRef = useRef(onPlaybackTick) // latest callback without re-binding audio events
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)
  const peaks = useWaveformPeaks(src, WAVE_BARS)

  useEffect(() => {
    tickCbRef.current = onPlaybackTick
  }, [onPlaybackTick])

  /** Report the element's current time and duration to the parent. */
  const reportPlayback = useCallback(() => {
    const el = audioRef.current
    if (!el) return
    const d = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : 0
    tickCbRef.current?.(el.currentTime, d)
  }, [])

  /** Set `currentTime`, clamped to the track length when it is known. */
  const applySeek = useCallback((el: HTMLAudioElement, sec: number) => {
    const d = el.duration
    el.currentTime = Number.isFinite(d) && d > 0 ? Math.min(Math.max(0, sec), d) : Math.max(0, sec)
    setCurrent(el.currentTime)
  }, [])

  useImperativeHandle(
    ref,
    () => ({
      seekTo(seconds: number) {
        const el = audioRef.current
        if (!el || !src) return
        const run = () => {
          applySeek(el, Math.max(0, seconds))
          reportPlayback()
          void el.play().catch(() => {
            /* autoplay policy may block play() without a user gesture */
          })
        }
        if (el.readyState >= HTMLMediaElement.HAVE_METADATA) run()
        else el.addEventListener('loadedmetadata', run, { once: true })
      },
    }),
    [src, applySeek, reportPlayback]
  )

  const togglePlay = useCallback(() => {
    const el = audioRef.current
    if (!el || !src) return
    if (el.paused) void el.play().catch(() => {})
    else el.pause()
  }, [src])

  /** Scrubber drag: move playback without auto-playing. */
  const onScrub = useCallback(
    (t: number) => {
      const el = audioRef.current
      if (!el) return
      el.currentTime = t
      setCurrent(t)
      reportPlayback()
    },
    [reportPlayback]
  )

  /** Read the duration once the browser knows it (metadata loaded / duration changed). */
  const syncDuration = () => {
    const el = audioRef.current
    if (el && Number.isFinite(el.duration)) setDuration(el.duration)
    reportPlayback()
  }

  if (!src) {
    return (
      <div className="mini-player mini-player--empty" aria-live="polite">
        <span className="mini-player__hint">{emptyHint?.trim() || 'Pick a sample or add an audio file to play it here'}</span>
      </div>
    )
  }

  return (
    <div className="mini-player">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onTimeUpdate={() => {
          const el = audioRef.current
          if (el) setCurrent(el.currentTime)
          reportPlayback()
        }}
        onLoadedMetadata={syncDuration}
        onDurationChange={syncDuration}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />

      <button type="button" className="mini-player__play" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>

      <div className="mini-player__track-wrap">
        <Waveform peaks={peaks} bars={WAVE_BARS} current={current} duration={duration} onSeek={onScrub} />
        <div className="mini-player__times">
          <span className="mini-player__t">{formatClock(current)}</span>
          <span className="mini-player__t mini-player__t--muted">{formatClock(duration)}</span>
        </div>
      </div>
    </div>
  )
})
