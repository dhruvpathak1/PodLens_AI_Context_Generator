import { useCallback, useEffect, useRef, useState } from 'react'
import type { MiniAudioPlayerHandle } from './MiniAudioPlayer'

/**
 * Playback state: an audio URL for `<audio>`, the current time and duration (reported by
 * `MiniAudioPlayer`), and a `seekTo` that any panel can call.
 *
 * `source` is either an uploaded `File` (played through an object URL) or a plain URL string
 * (demo episodes served from `public/demo/`).
 */
export function useAudioPlayback(source: File | string | null) {
  const file = source instanceof File ? source : null
  const playerRef = useRef<MiniAudioPlayerHandle>(null)
  /** Object URL paired with the file it was made for, so a stale URL is never returned. */
  const [objectUrl, setObjectUrl] = useState<{ file: File; url: string } | null>(null)
  const [playbackTime, setPlaybackTime] = useState(0)
  const [playbackDuration, setPlaybackDuration] = useState(0)

  // Create an object URL for the selected file and revoke it when the file changes.
  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    // The URL is an external resource that must be created and revoked in an effect, so state
    // is set here on purpose (once per file).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setObjectUrl({ file, url })
    return () => URL.revokeObjectURL(url)
  }, [file])

  const audioUrl =
    typeof source === 'string' ? source : objectUrl && objectUrl.file === file ? objectUrl.url : null

  // A new file starts at 0:00 (state adjusted during render, React's recommended pattern).
  const [timedUrl, setTimedUrl] = useState(audioUrl)
  if (timedUrl !== audioUrl) {
    setTimedUrl(audioUrl)
    setPlaybackTime(0)
    setPlaybackDuration(0)
  }

  /** Receives `(currentTime, duration)` from the player on every update. */
  const handlePlaybackTick = useCallback((t: number, d: number) => {
    setPlaybackTime(t)
    setPlaybackDuration(d)
  }, [])

  /** Jump to a moment and play (used by timeline, Ask citations and transcript clicks). */
  const seekTo = useCallback((sec: number) => {
    playerRef.current?.seekTo(sec)
  }, [])

  return { playerRef, audioUrl, playbackTime, playbackDuration, handlePlaybackTick, seekTo }
}
