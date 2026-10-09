import { useId } from 'react'

type Props = {
  /** Rendered width and height in pixels. */
  size?: number
  /** Accessible name; omit when a visible "PodLens" label sits next to the mark. */
  title?: string
}

/**
 * The PodLens mark: a lens ring whose stroke breaks into a sound wave, around a four-point
 * spark. Vector redraw of the original logo, with colours lifted to read on dark backgrounds.
 */
export function Logo({ size = 96, title }: Props) {
  const id = useId()
  const ring = `${id}-ring`
  const star = `${id}-star`
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 120 120"
      fill="none"
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <defs>
        <linearGradient id={ring} x1="10" y1="110" x2="110" y2="10" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#5b8cff" />
          <stop offset="1" stopColor="#3fd6a0" />
        </linearGradient>
        <linearGradient id={star} x1="34" y1="92" x2="78" y2="38" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#4aa8ff" />
          <stop offset="1" stopColor="#3fd6a0" />
        </linearGradient>
      </defs>
      {/* Ring that opens at the top right and continues as a sound wave */}
      <path
        d="M92.3 47.1A40 40 0 1 1 69.7 26.4C72.5 24 72.6 6 75.4 6S78.4 40 81.2 40 84.2 9 87 9s3.1 27 5.8 27 3.1-15 5.8-15 3 6 6.4 6h6"
        stroke={`url(#${ring})`}
        strokeWidth="6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Four-point spark */}
      <path d="M56 38 61.2 58.8 78 64 61.2 69.2 56 92 50.8 69.2 34 64 50.8 58.8Z" fill={`url(#${star})`} />
      <path d="M56 38 61.2 58.8 78 64 56 64Z" fill="#fff" fillOpacity="0.18" />
    </svg>
  )
}
