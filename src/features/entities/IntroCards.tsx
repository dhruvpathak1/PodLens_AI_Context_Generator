import { Logo } from '../../shared/components/Logo'

/**
 * The four cards shown in the live grid before the first name is mentioned:
 * welcome, what PodLens does, a disclaimer, and the logo.
 */
export function IntroCards() {
  return (
    <>
      <article className="intro-card">
        <h3 className="intro-card__title">Welcome to PodLens</h3>
        <p className="intro-card__text">
          AI context engine for podcasts. Transcribes episodes, tags people, places and companies, 
          and shows source cards, a timeline and cited Q&A synced to playback.
        </p>
      </article>

      <article className="intro-card">
        <h3 className="intro-card__title">What it does</h3>
        <ul className="intro-card__list">
          <li>Transcribes episodes and syncs audio with text.</li>
          <li>Finds every person, place and event mentioned. Generates a summary, photo and map.</li>
          <li>Builds a dated timeline and answers questions with timestamps you can play.</li>
        </ul>
      </article>

      <article className="intro-card intro-card--note">
        <h3 className="intro-card__title">Disclaimer</h3>
        <p className="intro-card__text intro-card__text--small">
          PodLens is an experimental project. Content is generated automatically by AI models and third-party APIs, and may be incomplete, inaccurate or out of date. Nothing
          here is professional, legal, medical or financial advice.
        </p>
      </article>

      <article className="intro-card intro-card--logo" aria-label="PodLens">
        <Logo size={104} />
        <p className="intro-card__wordmark">PodLens</p>
        <p className="intro-card__tagline">Zoom into every podcast</p>
      </article>
    </>
  )
}
