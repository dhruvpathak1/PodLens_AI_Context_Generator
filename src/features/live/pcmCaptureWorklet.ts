// PARKED (future scope): not imported by the app. See ./README.md for how to re-enable live mode.
/**
 * AudioWorklet that downmixes the mic to mono, resamples to 24 kHz and emits PCM16 LE frames
 * (~100 ms each), the format OpenAI Realtime transcription expects (`audio/pcm`, rate 24000).
 * Shipped as a Blob URL so no extra static asset / Vite config is needed.
 */
const WORKLET_SOURCE = `
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 24000;
    this.frameSamples = Math.round(this.targetRate * 0.1);
    this.ratio = sampleRate / this.targetRate;
    this.pos = 0;            // fractional read position into the input stream
    this.prev = 0;           // last sample of the previous block (for interpolation)
    this.out = new Int16Array(this.frameSamples);
    this.outLen = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const n = input[0].length;
    const mono = new Float32Array(n);
    for (let c = 0; c < input.length; c++) {
      const ch = input[c];
      for (let i = 0; i < n; i++) mono[i] += ch[i] / input.length;
    }
    // Linear-interpolation resampler; pos is relative to the start of this block (-1 = prev sample).
    while (this.pos < n - 1) {
      const i0 = Math.floor(this.pos);
      const frac = this.pos - i0;
      const a = i0 < 0 ? this.prev : mono[i0];
      const b = mono[i0 + 1];
      let s = a + (b - a) * frac;
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.outLen === this.frameSamples) {
        const buf = this.out.buffer.slice(0);
        this.port.postMessage(buf, [buf]);
        this.outLen = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= n;
    this.prev = mono[n - 1];
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
`

/** Blob URL is created once and reused for every session. */
let cachedUrl: string | null = null

/** URL to pass to `audioContext.audioWorklet.addModule()`; registers the `pcm-capture` processor. */
export function pcmCaptureWorkletUrl(): string {
  if (!cachedUrl) {
    cachedUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }))
  }
  return cachedUrl
}
