/* Anonymous per-browser learner identity (W10 §10 stateful learner features).
 *
 * IDENTITY MODEL (the pinned default): there are NO accounts and NO auth. A
 * learner is identified ONLY by an opaque uuid minted ONCE per browser and kept
 * in localStorage. Every stateful request sends it as the `X-Learner-Id` header.
 * The server treats it as an OWNER KEY ONLY — "this learner reads/writes their
 * own attempts/submissions" — never as an auth principal. This preserves the
 * learner-OPEN posture (spec §1.5): the assessment surface stays account-less.
 *
 * Vanilla ES module, no framework, no build step.
 */

const STORAGE_KEY = "ed4all.learnerId";

// Mint a uuid. Prefer crypto.randomUUID (every modern browser); fall back to a
// crypto.getRandomValues-seeded v4 shape so an old/locked-down browser still
// gets a stable opaque id. The id is NOT a secret (it's an owner key), so a
// non-cryptographic fallback is acceptable, but we use getRandomValues anyway.
function mintId() {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
  } catch (_e) {
    /* fall through */
  }
  const bytes = new Uint8Array(16);
  try {
    window.crypto.getRandomValues(bytes);
  } catch (_e) {
    for (let i = 0; i < 16; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  }
  // RFC-4122-ish v4 layout.
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.prototype.map
    .call(bytes, function (b) {
      return ("0" + b.toString(16)).slice(-2);
    })
    .join("");
  return (
    hex.slice(0, 8) +
    "-" +
    hex.slice(8, 12) +
    "-" +
    hex.slice(12, 16) +
    "-" +
    hex.slice(16, 20) +
    "-" +
    hex.slice(20)
  );
}

// Return the stable learner id for this browser, minting + persisting it once.
// localStorage may be unavailable (private mode / disabled) — fall back to a
// per-session id held in a module variable so the surface still works.
let _sessionId = null;
export function getLearnerId() {
  try {
    let id = window.localStorage.getItem(STORAGE_KEY);
    if (!id) {
      id = mintId();
      window.localStorage.setItem(STORAGE_KEY, id);
    }
    return id;
  } catch (_e) {
    if (!_sessionId) _sessionId = mintId();
    return _sessionId;
  }
}

// fetch() wrapper that attaches the X-Learner-Id header (merging any caller
// headers). Use this for every stateful learner request.
export function learnerFetch(url, options) {
  const opts = options || {};
  const headers = Object.assign({}, opts.headers || {}, {
    "X-Learner-Id": getLearnerId(),
  });
  return fetch(url, Object.assign({}, opts, { headers: headers }));
}
