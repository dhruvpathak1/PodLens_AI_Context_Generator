/**
 * Banner for production builds with no API URL configured.
 * GitHub Pages only serves static files, so the API must run elsewhere and be configured at build time.
 */
export function DeployHint() {
  return (
    <div className="dashboard__deploy-hint alert alert--error" role="status">
      This build has no <code className="app-inline-code">VITE_TRANSCRIBE_URL</code>. Add the GitHub Actions secret{' '}
      <code className="app-inline-code">VITE_TRANSCRIBE_URL</code> pointing at your API (for example{' '}
      <code className="app-inline-code">https://your-host/api/transcribe</code>), rebuild, and set{' '}
      <code className="app-inline-code">CORS_EXTRA_ORIGINS</code> on the server to this site&apos;s origin (for example{' '}
      <code className="app-inline-code">https://your-username.github.io</code>).
    </div>
  )
}
