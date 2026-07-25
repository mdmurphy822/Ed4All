/* statusPill — vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * ONE source of truth for every status badge across the three surfaces:
 *   - the 5 run statuses (queued|running|completed|failed|cancelled),
 *   - the 6 learner answer statuses (answered|refused|blocked|error|asking|
 *     no_index),
 *   - the 3 gate severities (pass|warn|fail / critical mapped to fail).
 *
 * Non-color-only by construction (WCAG 1.4.1): every pill renders a GLYPH + a
 * plain-language TEXT label, never color alone. Takes a kind string only; calls
 * no API.
 */

import { el } from '../dom.js';

/* kind → { glyph, label, cls }. cls drives the token-backed color in the CSS.
 * The labels are plain language (the "what a non-developer reads" copy map). */
const _RUN = {
  queued: { glyph: '◷', label: 'Queued', cls: 'pending' },
  running: { glyph: '◐', label: 'Building', cls: 'running' },
  building: { glyph: '◐', label: 'Building', cls: 'running' },
  completed: { glyph: '●', label: 'Ready', cls: 'done' },
  failed: { glyph: '✕', label: 'Failed', cls: 'failed' },
  cancelled: { glyph: '–', label: 'Cancelled', cls: 'skipped' },
  cancel_requested: { glyph: '◌', label: 'Cancelling…', cls: 'running' },
  interrupted: { glyph: '–', label: 'Interrupted', cls: 'skipped' },
  // Honest non-terminal states derived server-side from process liveness +
  // the graceful-stop sentinel (gui/services/liveness.py::effective_status).
  // Each is a glyph + plain-language label (never color-only, WCAG 1.4.1).
  paused: { glyph: '❚❚', label: 'Paused', cls: 'warn' },
  stopping: { glyph: '◔', label: 'Stopping…', cls: 'warn' },
  incomplete: { glyph: '■', label: 'Incomplete', cls: 'skipped' },
  'stalled?': { glyph: '◍', label: 'Stalled?', cls: 'warn' },
};

const _ANSWER = {
  answered: { glyph: '●', label: 'Answered', cls: 'done' },
  asking: { glyph: '◐', label: 'Thinking…', cls: 'running' },
  refused: { glyph: '–', label: 'No clear answer', cls: 'skipped' },
  blocked: { glyph: '△', label: "Couldn't verify", cls: 'warn' },
  error: { glyph: '✕', label: 'Error', cls: 'failed' },
  no_index: { glyph: '○', label: 'Keyword search only', cls: 'pending' },
};

const _SEVERITY = {
  pass: { glyph: '✓', label: 'Passing', cls: 'pass' },
  ok: { glyph: '✓', label: 'OK', cls: 'pass' },
  info: { glyph: 'ℹ', label: 'Info', cls: 'info' },
  warn: { glyph: '△', label: 'Warning', cls: 'warn' },
  warning: { glyph: '△', label: 'Warning', cls: 'warn' },
  fail: { glyph: '✗', label: 'Failed', cls: 'fail' },
  critical: { glyph: '✗', label: 'Failed', cls: 'fail' },
};

const _REGISTRY = { ...Object.fromEntries(Object.entries(_RUN).map(([k, v]) => [`run:${k}`, v])) };

/** The full kind table, exported so the gallery can enumerate every variant. */
export const PILL_KINDS = {
  run: _RUN,
  answer: _ANSWER,
  severity: _SEVERITY,
};

function _resolve(kind) {
  // Prefixed lookups (`run:running`, `answer:blocked`, `severity:fail`) win; a
  // bare key resolves run → answer → severity, in that order.
  if (typeof kind === 'string' && kind.includes(':')) {
    const [group, key] = kind.split(':', 2);
    const table = { run: _RUN, answer: _ANSWER, severity: _SEVERITY }[group];
    if (table && table[key]) return { ...table[key], group };
  }
  if (_RUN[kind]) return { ..._RUN[kind], group: 'run' };
  if (_ANSWER[kind]) return { ..._ANSWER[kind], group: 'answer' };
  if (_SEVERITY[kind]) return { ..._SEVERITY[kind], group: 'severity' };
  return { glyph: '·', label: String(kind || 'Unknown'), cls: 'pending', group: 'unknown' };
}

/**
 * Build a status pill.
 * @param {string} kind  one of the run / answer / severity keys, optionally
 *                       prefixed `run:` / `answer:` / `severity:`.
 * @returns {HTMLElement} a <span class="pill pill-<cls>"> with glyph + text.
 */
export function statusPill(kind) {
  const r = _resolve(kind);
  return el('span', { class: `pill pill-${r.cls}`, 'data-kind': String(kind), 'data-group': r.group }, [
    el('span', { class: 'pill-glyph', 'aria-hidden': 'true', text: r.glyph }),
    el('span', { class: 'pill-label', text: r.label }),
  ]);
}

export default statusPill;
