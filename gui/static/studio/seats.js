/* Ed4All Studio — vLLM seat monitor (shared by the Dashboard + the build page).
 *
 * WHY: two concurrent runs once fought over one seat — run B's seat schedule
 * stopped the seat run A was mid-dispatch to, and a 2.5-hour build failed with
 * nothing in the GUI showing that the seat the current phase depended on had
 * gone away. This module is the always-on answer to "which seat is up", and on
 * a build page the EXPECTED-vs-ACTUAL answer: "Phase X needs <seat> — <seat> is
 * not serving".
 *
 * Two mounts over ONE backend (gui/services/seat_service.py, one shared ~10s
 * server cache — a dashboard and a build page open at once do NOT double the
 * probe rate):
 *   - mountSeatMonitor({ runId })  → the compact strip for the build page
 *     (GET /api/seats/run/<id>: seats + current phase + mismatch),
 *   - renderSeatSection(shell)     → the Dashboard's always-visible section
 *     (GET /api/seats: seats only; no run needed).
 *
 * REGISTRY-DRIVEN: every seat rendered comes from the payload, in payload
 * (registry) order. NO seat name appears in this file — a new seat is a
 * registry entry (ED4ALL_SEAT_BASE_URLS), never code.
 *
 * A11y: each chip is a statusPill (glyph + text) PLUS the seat name and the
 * state word in plain text — never color alone (WCAG 1.4.1). The mismatch line
 * carries a "⚠" glyph + full sentence. No new live region is introduced (the
 * monitor polls; announcing every 15s would be noise) — the pages keep their
 * single polite region.
 *
 * HONESTY: `loading` (container up, /v1/models not answering yet) is a ~9-minute
 * cold start, NOT an alarm; `unknown` means we truly cannot tell. The age
 * ("live 12m") is how long THIS SERVER PROCESS has observed the current state,
 * and is omitted entirely when the server reports none.
 */

import { api } from '/shared/api.js';
import { el, clear } from '/shared/dom.js';
import { statusPill } from '/shared/components/pill.js';

/** Poll cadence while a monitor is on screen (the server caches ~10s). */
export const SEAT_POLL_MS = 15000;

/** Run states after which the build-page strip stops polling. */
const SEAT_TERMINAL = ['completed', 'failed', 'cancelled', 'canceled', 'interrupted', 'paused', 'timeout', 'error', 'incomplete'];

/* state → { pill kind, plain-language word }. The word is rendered as TEXT, so
 * the distinction never depends on the pill's color. A seat that is down "at
 * rest" is NOT a failure (the GPU-lifecycle design hands the card between
 * stages), so `down` is only a warning here — it escalates to a failure pill
 * exactly when the CURRENT PHASE declares it as needed (see seatChip). */
const STATE_VIEW = {
  live: { pill: 'severity:ok', word: 'serving' },
  loading: { pill: 'severity:info', word: 'starting up' },
  down: { pill: 'severity:warn', word: 'down' },
  unknown: { pill: 'severity:info', word: 'unknown' },
};

/** States that satisfy a phase's declared seat expectation (mirrors the service). */
const SATISFYING = ['live', 'loading'];

function stateView(state) {
  return STATE_VIEW[String(state || '').toLowerCase()] || STATE_VIEW.unknown;
}

/** Compact relative age: 45s / 12m / 2h 5m. */
export function fmtAge(ms) {
  const sec = Math.max(0, Math.round(Number(ms) / 1000));
  if (!Number.isFinite(sec)) return '';
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `${h}h ${rem}m` : `${h}h`;
}

/** Age of a seat's current state, preferring the authoritative ISO marker. */
function seatAge(seat) {
  const iso = seat && seat.since;
  if (iso) {
    const t = Date.parse(iso);
    if (!Number.isNaN(t)) return fmtAge(Date.now() - t);
  }
  if (Number.isFinite(seat && seat.since_ms)) return fmtAge(seat.since_ms);
  return '';
}

/** One seat chip: pill + seat name + state word (+ age, + "needed now"). */
function seatChip(seat) {
  const view = stateView(seat.state);
  const age = seatAge(seat);
  const missing = !!seat.expected && !SATISFYING.includes(String(seat.state || '').toLowerCase());
  const li = el('li', {
    class: missing ? 'seat-chip seat-chip-missing' : 'seat-chip',
    'data-state': String(seat.state || 'unknown'),
    'data-seat': String(seat.name || ''),
  });
  li.appendChild(statusPill(missing ? 'severity:fail' : view.pill));
  li.appendChild(el('span', { class: 'seat-chip-name', text: String(seat.name || '') }));
  li.appendChild(el('span', {
    class: 'seat-chip-state',
    text: age ? `${view.word} · ${age}` : view.word,
  }));
  if (seat.expected) {
    li.appendChild(el('span', { class: 'seat-chip-flag', text: 'needed now' }));
  }
  const bits = [];
  if (seat.base_url) bits.push(String(seat.base_url));
  if (seat.model) bits.push(`model ${seat.model}`);
  if (seat.container) bits.push(`container ${seat.container}`);
  if (bits.length) li.title = bits.join(' · ');
  return li;
}

/** The mismatch warning line: names the phase and every missing seat. */
function mismatchLine(payload) {
  const rows = Array.isArray(payload.mismatch) ? payload.mismatch : [];
  if (!rows.length) return null;
  const phase = String(payload.current_phase || 'this phase');
  const names = rows.map((m) => String(m.seat || '')).filter(Boolean).join(', ');
  const states = rows.map((m) => `${m.seat}: ${stateView(m.state).word}`).join('; ');
  return el('p', { class: 'seat-mismatch' }, [
    el('span', { class: 'seat-mismatch-glyph', 'aria-hidden': 'true', text: '⚠ ' }),
    el('span', {
      text: `Phase ${phase} needs ${names} — not serving (${states}). A dispatch to a stopped seat fails the build.`,
    }),
  ]);
}

/** Render a payload into `box`; returns nothing (pure presentation). */
function renderInto(box, payload, { showPhase }) {
  clear(box);
  if (!payload) return;
  const seats = Array.isArray(payload.seats) ? payload.seats : [];
  if (showPhase) {
    const warn = mismatchLine(payload);
    if (warn) box.appendChild(warn);
  }
  if (!seats.length) {
    box.appendChild(el('p', {
      class: 'seat-empty muted',
      text: payload.registry_configured
        ? 'No model seats reported.'
        : 'No model seats registered (ED4ALL_SEAT_BASE_URLS is unset), so there is nothing to monitor.',
    }));
    return;
  }
  const list = el('ul', { class: 'seat-strip', 'aria-label': 'Model seat status' });
  seats.forEach((s) => list.appendChild(seatChip(s)));
  box.appendChild(list);

  const notes = [];
  if (showPhase && payload.current_phase) {
    const expected = payload.expected_seats;
    if (Array.isArray(expected) && expected.length) {
      notes.push(`Phase ${payload.current_phase} needs ${expected.join(', ')}.`);
    } else if (Array.isArray(expected)) {
      notes.push(`Phase ${payload.current_phase} needs no seat.`);
    } else {
      notes.push(`Phase ${payload.current_phase} declares no seat requirement.`);
    }
  }
  if (payload.docker_available === null || payload.docker_available === undefined) {
    notes.push('Container state unavailable, so a stopped seat cannot be told apart from one still starting.');
  }
  notes.push('Ages are since this server first saw the current state.');
  box.appendChild(el('p', { class: 'seat-note muted', text: notes.join(' ') }));
}

/**
 * Build-page seat strip. `runId` gives the phase-aware payload (expected seats
 * + mismatch); without one it falls back to the global list. Callers append
 * `.el`, call `start()`, feed `setRunState(effective_status)` from the stage
 * rail, and `stop()` on teardown. A terminal run takes one last snapshot and
 * stops polling.
 */
export function mountSeatMonitor({ runId } = {}) {
  const box = el('section', { class: 'seat-monitor', 'aria-label': 'Model seat status' });
  const url = runId ? `/api/seats/run/${encodeURIComponent(runId)}` : '/api/seats';
  let timer = null;
  let latched = false;

  async function pollOnce() {
    let payload = null;
    try {
      payload = await api(url);
    } catch (_) {
      // Best-effort: an unknown run (404) or a transient failure leaves the
      // last good strip in place rather than blanking it.
      return;
    }
    renderInto(box, payload, { showPhase: !!runId });
  }
  function stop() {
    if (timer != null) { clearInterval(timer); timer = null; }
  }
  function start() {
    if (latched) return;
    pollOnce();
    timer = setInterval(pollOnce, SEAT_POLL_MS);
  }
  /** Stop polling once the run is over (one final snapshot first). */
  function setRunState(eff) {
    const state = String(eff || '').toLowerCase();
    if (!latched && SEAT_TERMINAL.includes(state)) {
      latched = true;
      stop();
      pollOnce();
    }
  }
  return { el: box, start, stop, pollOnce, setRunState };
}

/**
 * Dashboard "Model seats" section — the ALWAYS-ON monitor. Renders with no
 * build running (the run-scoped phase/mismatch fields are simply absent) and
 * self-polls while mounted. Returns `{ el, stop }`; the dashboard route hands
 * `stop` back to the router so a route change tears the timer down.
 */
export function renderSeatSection() {
  const sec = el('section', { class: 'dash-zone seat-zone', 'aria-labelledby': 'dash-seats-h' });
  sec.appendChild(el('h2', { id: 'dash-seats-h', class: 'dash-zone-h', text: 'Model seats' }));
  sec.appendChild(el('p', {
    class: 'dash-zone-intro muted',
    text: 'Live status of every registered model seat on this server — a build fails if the seat its phase needs is not serving.',
  }));
  const ctl = mountSeatMonitor({});
  sec.appendChild(ctl.el);
  ctl.start();
  return { el: sec, stop: ctl.stop };
}

export default mountSeatMonitor;
