/* Shared fetch wrapper — vanilla ES module, NO build step.
 *
 * Extracted from gui/static/app.js so every surface normalizes the typed
 * {error, detail} body + non-2xx status into a thrown ApiError identically.
 * Keep this byte-compatible with app.js's api()/ApiError so a later app.js
 * port is a no-op swap.
 */

export class ApiError extends Error {
  constructor(status, error, detail) {
    super(detail || error || `HTTP ${status}`);
    this.status = status;
    this.error = error || 'error';
    this.detail = detail || '';
  }
}

/* Opt-in error hooks. A surface (e.g. the /dev/ console's ApiError drawer)
 * registers a hook to observe every ApiError api() throws WITHOUT changing the
 * throw behavior — studio keeps its toast-on-catch flow untouched. Hooks are
 * called best-effort just before the throw with {path, method, status, error,
 * detail}; a throwing hook never masks the original ApiError. */
const _errorHooks = new Set();
export function onApiError(hook) {
  if (typeof hook === 'function') _errorHooks.add(hook);
  return () => _errorHooks.delete(hook);
}
function _notifyError(path, opts, err) {
  if (!_errorHooks.size) return;
  const info = {
    path,
    method: (opts && opts.method) || 'GET',
    status: err.status,
    error: err.error,
    detail: err.detail,
  };
  for (const hook of _errorHooks) {
    try { hook(info); } catch (_) { /* a broken hook must not mask the error */ }
  }
}

/** fetch + typed-error normalization. Throws ApiError on network / non-2xx.
 * Every ApiError is reported to any registered error hooks (opt-in) before the
 * throw, so an observer surface (the /dev/ ApiError drawer) sees it without
 * altering the throw flow. */
export async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (networkErr) {
    const e = new ApiError(0, 'network_error', String((networkErr && networkErr.message) || networkErr));
    _notifyError(path, opts, e);
    throw e;
  }
  if (res.status === 204) return null;
  const ctype = res.headers.get('content-type') || '';
  let body = null;
  if (ctype.includes('application/json')) {
    try { body = await res.json(); } catch (_) { body = null; }
  } else {
    const txt = await res.text();
    body = txt ? { error: 'non_json_response', detail: txt.slice(0, 500) } : null;
  }
  if (!res.ok) {
    let e;
    if (body && Array.isArray(body.detail)) {
      const detail = body.detail
        .map((entry) => {
          const loc = Array.isArray(entry?.loc) ? entry.loc.slice(1).join('.') : '';
          const msg = entry?.msg ?? entry?.type ?? 'invalid';
          return loc ? `${loc}: ${msg}` : String(msg);
        })
        .filter(Boolean)
        .join('; ') || res.statusText;
      e = new ApiError(res.status, 'validation_error', detail);
    } else {
      const nested = body && typeof body.detail === 'object' && body.detail !== null ? body.detail : null;
      const error = (nested?.error ?? body?.error ?? `http_${res.status}`);
      const detail = (nested?.detail ?? (typeof body?.detail === 'string' ? body.detail : null) ?? body?.error ?? res.statusText);
      e = new ApiError(res.status, error, detail);
    }
    _notifyError(path, opts, e);
    throw e;
  }
  return body;
}

/** JSON body convenience for POST/PUT/PATCH. */
export const apiJSON = (path, method, body) =>
  api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
