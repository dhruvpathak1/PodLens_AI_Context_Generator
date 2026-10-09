import type { ReactNode } from 'react'
import { REPO_URL } from '../../config/env'
import { Panel } from '../../shared/components/Panel'
import { ExternalIcon } from '../../shared/icons'

type Props = {
  /** Sample episode buttons (demo) or the upload box and run button (local). */
  children: ReactNode
  /** Note under the controls, e.g. the audio credit of the current sample. */
  note?: ReactNode
}

/** Left column, bottom: what PodLens is, plus the way to load an episode. */
export function AboutPanel({ children, note }: Props) {
  return (
    <Panel
      className="panel--about"
      title="PodLens"
      actions={
        <a className="link-quiet" href={REPO_URL} target="_blank" rel="noreferrer">
          Source <ExternalIcon />
        </a>
      }
    >
      <p className="about__lead">
        Listens to a podcast and researches it while you listen. Every person, place and organisation becomes a source
        card the moment it is spoken, with a dated timeline and answers you can check against the audio.
      </p>
      <div className="about__controls">{children}</div>
      {note ? <p className="about__note">{note}</p> : null}
    </Panel>
  )
}
