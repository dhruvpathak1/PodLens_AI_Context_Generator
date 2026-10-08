import { useCallback, useId, useRef, useState } from 'react'
import { formatBytes } from '../../shared/format'
import { UploadIcon } from '../../shared/icons'

/** MIME types offered in the file picker (the server accepts anything ffmpeg can decode). */
const ACCEPT = 'audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/webm,audio/ogg,audio/mp4,audio/x-m4a,audio/flac,audio/*'

/** Fallback check for files whose MIME type the browser leaves empty. */
const AUDIO_EXTENSIONS = /\.(mp3|wav|m4a|ogg|webm|flac)$/i

type Props = {
  file: File | null
  onFileChange: (file: File | null) => void
  disabled?: boolean
  /** Shorter drop zone for the sidebar layout. */
  compact?: boolean
}

/** Click-or-drop audio picker showing the chosen file's name and size, with a "Remove" link. */
export function AudioDropZone({ file, onFileChange, disabled, compact }: Props) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [dragOver, setDragOver] = useState(false)

  /** Accept the first file if it looks like audio; silently ignore anything else. */
  const pick = useCallback(
    (list: FileList | null) => {
      const f = list?.[0]
      if (!f) return
      if (!f.type.startsWith('audio/') && !AUDIO_EXTENSIONS.test(f.name)) return
      onFileChange(f)
    },
    [onFileChange]
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      if (!disabled) pick(e.dataTransfer.files)
    },
    [disabled, pick]
  )

  /** Drag-over must call preventDefault for the drop event to fire. */
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled) setDragOver(true)
  }

  const zoneClass = [
    'drop-zone',
    compact && 'drop-zone--compact',
    dragOver && 'drop-zone--active',
    file && 'drop-zone--has-file',
  ]
    .filter(Boolean)
    .join(' ')

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
          pick(e.target.files)
          e.target.value = '' // allow picking the same file again
        }}
      />
      <button
        type="button"
        className={zoneClass}
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragEnter={onDragOver}
        onDragOver={onDragOver}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        aria-labelledby={`${inputId}-label`}
      >
        <span className="drop-zone__icon" aria-hidden>
          <UploadIcon />
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
              <span className="drop-zone__hint">or click to browse: MP3, WAV, M4A, WebM, OGG, FLAC</span>
            </>
          )}
        </span>
      </button>
      {file && !disabled && (
        <button type="button" className="btn-text" onClick={() => onFileChange(null)}>
          Remove file
        </button>
      )}
    </div>
  )
}
