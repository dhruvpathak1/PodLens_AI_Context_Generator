import type { ReactNode } from 'react'

type Props = {
  children: ReactNode
  /** Extra class names (e.g. layout helpers). */
  className?: string
}

/** Red error banner announced to screen readers. Renders nothing when `children` is empty. */
export function ErrorAlert({ children, className }: Props) {
  if (children == null || children === '' || children === false) return null
  return (
    <div className={`alert alert--error${className ? ` ${className}` : ''}`} role="alert">
      {children}
    </div>
  )
}
