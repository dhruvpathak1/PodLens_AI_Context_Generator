import { useEffect, useState } from 'react'

/** Files larger than this are not decoded for the waveform (too slow and memory hungry). */
const MAX_DECODE_BYTES = 60 * 1024 * 1024

/**
 * Loudness envelope of an audio file for the waveform scrubber: `bars` values in 0..1.
 * Decodes the file once with Web Audio (works for object URLs and same-origin demo files).
 *
 * Returns null while decoding, and an empty array when the file cannot be decoded (the
 * scrubber then draws a neutral placeholder).
 */
export function useWaveformPeaks(src: string | null, bars: number): number[] | null {
  const [result, setResult] = useState<{ src: string; peaks: number[] } | null>(null)

  useEffect(() => {
    if (!src) return
    let cancelled = false
    const ctrl = new AbortController()
    void (async () => {
      let peaks: number[] = []
      try {
        const buf = await (await fetch(src, { signal: ctrl.signal })).arrayBuffer()
        if (buf.byteLength <= MAX_DECODE_BYTES) {
          // A tiny offline context is enough: we only need decodeAudioData.
          const ctx = new OfflineAudioContext(1, 1, 44100)
          const audio = await ctx.decodeAudioData(buf)
          peaks = computePeaks(audio.getChannelData(0), bars)
        }
      } catch {
        peaks = [] // network error, abort or undecodable file: fall back to the placeholder
      }
      if (!cancelled) setResult({ src, peaks })
    })()
    return () => {
      cancelled = true
      ctrl.abort()
    }
  }, [src, bars])

  return result && result.src === src ? result.peaks : null
}

/** Root-mean-square loudness per bar, normalized so the loudest bar is 1 (with a small floor). */
export function computePeaks(samples: Float32Array, bars: number): number[] {
  if (!samples.length || bars <= 0) return []
  const block = Math.max(1, Math.floor(samples.length / bars))
  const stride = Math.max(1, Math.floor(block / 400)) // sample at most ~400 points per bar
  const out: number[] = []
  for (let b = 0; b < bars; b++) {
    let sum = 0
    let n = 0
    for (let i = b * block; i < Math.min((b + 1) * block, samples.length); i += stride) {
      sum += samples[i] * samples[i]
      n++
    }
    out.push(n ? Math.sqrt(sum / n) : 0)
  }
  const max = Math.max(...out) || 1
  return out.map((v) => Math.max(0.08, v / max))
}
