/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_TRANSCRIBE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
