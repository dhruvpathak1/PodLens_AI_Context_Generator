import { useCallback, useId, useRef, useState } from 'react'

const ACCEPT = 'audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/webm,audio/ogg,audio/mp4,audio/x-m4a,audio/flac,audio/*'

type Props = {
  file: File | null
  onFileChange: (file: File | null) => void
  disabled?: boolean
  /** Shorter drop zone for sidebar layouts */
  compact?: boolean
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

export function AudioDropZone({ file, onFileChange, disabled, compact }: Props) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const setFile = useCallback(
    (f: File | null) => {
      onFileChange(f)
    },
    [onFileChange]
  )

  const onPick = useCallback(
    (list: FileList | null) => {
      const f = list?.[0]
      if (!f) return
      if (!f.type.startsWith('audio/') && !/\.(mp3|wav|m4a|ogg|webm|flac)$/i.test(f.name)) {
        return
      }
      setFile(f)
    },
    [setFile]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      if (disabled) return
      onPick(e.dataTransfer.files)
    },
    [disabled, onPick]
  )

  return (
    <div className="drop-wrap">
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={ACCEPT}
        className="sr-only"
        disabled={disabled}
        onChange={(e) => {
          onPick(e.target.files)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        className={`drop-zone${compact ? ' drop-zone--compact' : ''}${dragOver ? ' drop-zone--active' : ''}${file ? ' drop-zone--has-file' : ''}`}
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(e) => {
          e.preventDefault()
          if (!disabled) setDragOver(true)
        }}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        aria-labelledby={`${inputId}-label`}
      >
        <span className="drop-zone__icon" aria-hidden>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 18V6m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4" strokeLinecap="round" />
          </svg>
        </span>
        <span id={`${inputId}-label`} className="drop-zone__text">
          {file ? (
            <>
              <strong className="drop-zone__name">{file.name}</strong>
              <span className="drop-zone__meta">{formatBytes(file.size)}</span>
            </>
          ) : (
            <>
              <strong>Drop an audio file here</strong>
              <span className="drop-zone__hint">or click to browse — MP3, WAV, M4A, WebM, OGG, FLAC</span>
            </>
          )}
        </span>
      </button>
      {file && !disabled && (
        <button type="button" className="btn-text" onClick={() => setFile(null)}>
          Remove file
        </button>
      )}
    </div>
  )
}
