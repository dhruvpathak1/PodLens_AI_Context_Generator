import type { ReactNode } from 'react'

type Props = {
  /** Visible heading; also labels the region for screen readers. */
  title: ReactNode
  /** Short secondary text next to the title (status, counts). */
  meta?: ReactNode
  /** Controls on the right of the header (buttons, links). */
  actions?: ReactNode
  /** Modifier for layout sizing, e.g. `panel--timeline`. */
  className?: string
  /** Extra class for the scrolling body (e.g. to disable padding). */
  bodyClassName?: string
  /** Content rendered between the header and the scrolling body (e.g. the audio player). */
  toolbar?: ReactNode
  /** Content pinned below the scrolling body (e.g. the question box). */
  footer?: ReactNode
  children: ReactNode
}

/**
 * The dashboard's building block: a raised surface with a header and a body that scrolls on
 * its own, so the page itself never scrolls on desktop.
 */
export function Panel({ title, meta, actions, className, bodyClassName, toolbar, footer, children }: Props) {
  return (
    <section className={`panel${className ? ` ${className}` : ''}`}>
      <header className="panel__head">
        <h2 className="panel__title">{title}</h2>
        {meta ? <span className="panel__meta">{meta}</span> : null}
        {actions ? <div className="panel__actions">{actions}</div> : null}
      </header>
      {toolbar}
      <div className={`panel__body${bodyClassName ? ` ${bodyClassName}` : ''}`}>{children}</div>
      {footer}
    </section>
  )
}
