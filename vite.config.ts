import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.API_PORT ?? '8000'

/**
 * Base URL the site is served from (`VITE_BASE_PATH`):
 * - unset or "/"  -> served from the domain root (local dev, Docker)
 * - "./"          -> relative asset URLs: the same build works at any path, e.g. both
 *                    podlens.dhruvpathak.com/ and <user>.github.io/<repo>/ (used by the Pages workflow)
 * - "/repo"       -> normalised to "/repo/"
 */
function normalizeBase(raw: string | undefined): string {
  const v = raw == null ? '' : String(raw).trim()
  if (v === '' || v === '/') return '/'
  if (v === '.' || v === './') return './'
  let b = String(raw).trim()
  if (!b.startsWith('/')) b = `/${b}`
  if (!b.endsWith('/')) b = `${b}/`
  return b
}

// https://vite.dev/config/
export default defineConfig({
  base: normalizeBase(process.env.VITE_BASE_PATH),
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
        // /api/live is a WebSocket (OpenAI Realtime transcription relay)
        ws: true,
      },
    },
  },
})
