type Props = {
  /** Extra class names, e.g. `skeleton--in-feed` inside the transcript feed. */
  className?: string
}

/** Three shimmering placeholder lines shown while content loads. */
export function Skeleton({ className }: Props) {
  return (
    <div className={`skeleton${className ? ` ${className}` : ''}`} aria-live="polite">
      <div className="skeleton__line" />
      <div className="skeleton__line skeleton__line--short" />
      <div className="skeleton__line" />
    </div>
  )
}
