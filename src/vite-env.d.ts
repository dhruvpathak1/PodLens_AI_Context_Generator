/// <reference types="vite/client" />

/** Typed `import.meta.env` for the variables this app reads. */
interface ImportMetaEnv {
  /** Full URL of POST /api/transcribe on the API host, e.g. https://api.example.com/api/transcribe.
   *  Required for static hosting (GitHub Pages); leave empty in local dev to use the Vite proxy. */
  readonly VITE_TRANSCRIBE_URL?: string
  /** Force the NER backend from the UI: "spacy" | "claude". */
  readonly VITE_ENTITY_BACKEND?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
