/* gateChip — vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * A compact icon + visually-hidden text label for a single validation gate
 * result. NEVER color-only (WCAG 1.4.1): the colored glyph is paired with a
 * visually-hidden label so a screen reader reads "Passed: <gate>" while a
 * sighted user reads the ✓/△/✗ glyph. Takes data as args, calls no API.
 *
 * Result vocabulary (mapped onto the gate tokens in tokens.css):
 *   pass     → ✓  --gate-pass
 *   warn     → △  --gate-warn   (also: "warning")
 *   fail     → ✗  --gate-fail   (also: "critical")
 */

import { el } from '../dom.js';

const _RESULT = {
  pass: { glyph: '✓', word: 'Passed', cls: 'pass' },
  warn: { glyph: '△', word: 'Warning', cls: 'warn' },
  warning: { glyph: '△', word: 'Warning', cls: 'warn' },
  fail: { glyph: '✗', word: 'Failed', cls: 'fail' },
  critical: { glyph: '✗', word: 'Failed', cls: 'fail' },
};

/** The result vocabulary, exported so the gallery enumerates every variant. */
export const GATE_RESULTS = _RESULT;

function _resolve(result) {
  return _RESULT[String(result || '').toLowerCase()] || _RESULT.warn;
}

/**
 * Build one gate chip.
 * @param {Object} gate
 * @param {string} gate.result  pass|warn|warning|fail|critical.
 * @param {string} [gate.gate_id]  the gate id (e.g. "curie_anchoring"); shown
 *                                  in the visually-hidden label.
 * @param {string} [gate.label]  optional friendlier label overriding gate_id.
 * @returns {HTMLElement} a <span class="gate-chip gate-chip-<cls>">.
 */
export function gateChip(gate = {}) {
  const r = _resolve(gate.result);
  const name = gate.label || gate.gate_id || 'check';
  const span = el('span', {
    class: `gate-chip gate-chip-${r.cls}`,
    'data-result': r.cls,
    'data-gate': gate.gate_id || '',
    title: `${r.word}: ${name}`,
  }, [
    el('span', { class: 'gate-chip-glyph', 'aria-hidden': 'true', text: r.glyph }),
    // The visually-hidden label is the a11y truth (never color-only).
    el('span', { class: 'visually-hidden', text: `${r.word}: ${name}` }),
  ]);
  return span;
}

export default gateChip;
