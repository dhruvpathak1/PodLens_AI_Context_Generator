import { Logo } from '../../shared/components/Logo'

/**
 * The four cards shown in the live grid before the first name is mentioned:
 * welcome, what PodLens does, a disclaimer, and the logo.
 */
export function IntroCards() {
  return (
    <>
      <article className="intro-card">
        <p className="intro-card__kicker">Welcome</p>
        <h3 className="intro-card__title">Welcome to PodLens</h3>
        <p className="intro-card__text">
          Press play on a sample, or add your own episode. As people, places and organisations are mentioned, their
          source cards replace these four.
        </p>
      </article>

      <article className="intro-card">
        <p className="intro-card__kicker">What it does</p>
        <h3 className="intro-card__title">Context while you listen</h3>
        <ul className="intro-card__list">
          <li>Transcribes the episode and keeps the text in sync with the audio</li>
          <li>Finds every person, place and organisation mentioned</li>
          <li>Pulls a summary, photo and map for each one</li>
          <li>Builds a dated timeline and answers questions with timestamps you can play</li>
        </ul>
      </article>

      <article className="intro-card intro-card--note">
        <p className="intro-card__kicker">Disclaimer</p>
        <h3 className="intro-card__title">Check before you rely on it</h3>
        <p className="intro-card__text intro-card__text--small">
          PodLens is an experimental project. Transcripts, names, summaries, timelines and answers are generated
          automatically by AI models and third-party sources, and may be incomplete, inaccurate or out of date. Nothing
          here is professional, legal, medical or financial advice. Verify important facts against the original audio
          and primary sources. Summaries, photos and maps belong to their owners (Wikipedia, Unsplash, OpenStreetMap)
          and are shown with attribution. Provided as is, without warranty of any kind.
        </p>
      </article>

      <article className="intro-card intro-card--logo" aria-label="PodLens">
        <Logo size={104} />
        <p className="intro-card__wordmark">PodLens</p>
        <p className="intro-card__tagline">Context for every episode</p>
      </article>
    </>
  )
}
