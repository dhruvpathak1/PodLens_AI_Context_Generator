/** Small fetch helpers shared by every API module. */

/**
 * Extract a human-readable error from a failed response.
 * FastAPI returns `{detail: string}` or, for validation errors, `{detail: [{msg}]}`.
 */
export async function readErrorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const ct = res.headers.get('content-type') ?? ''
    if (ct.includes('application/json')) {
      const body = (await res.json()) as Record<string, unknown>
      const detail = body.detail ?? body.error ?? body.message
      if (typeof detail === 'string') return detail
      if (Array.isArray(detail)) {
        const msgs = detail
          .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : null))
          .filter(Boolean)
        if (msgs.length) return msgs.join('; ')
      }
    } else {
      const text = await res.text()
      if (text) return text.slice(0, 500)
    }
  } catch {
    /* body unreadable: use the fallback */
  }
  return res.statusText || fallback
}

/** POST a JSON body and parse the JSON response; throws `Error(detail)` on non-2xx. */
export async function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok) throw new Error(await readErrorDetail(res, `Request failed (${res.status})`))
  return (await res.json()) as T
}
