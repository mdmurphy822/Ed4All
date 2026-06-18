/* phaseRow — vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * A single phase row in the build checklist. Faithfully extracts the working
 * create.js::renderProgress behavior (the 5 states + the glyph→state mapping)
 * WITHOUT editing create.js, swapping the static Unicode glyph for a real SVG
 * progressRing (the Phase-3 swap is what wires it INTO create.js; this is the
 * reusable primitive).
 *
 * The five states + their create.js glyph mapping, preserved verbatim:
 *   pending  ○  → ring(pending)   "Pending"
 *   running  ◐  → ring(running)   "Running…"
 *   done     ●  → ring(done)      "Done"
 *   skipped  –  → ring(skipped)   "Skipped"
 *   failed   ✕  → ring(failed)    "Failed"
 *
 * The ring is aria-hidden DECORATION; the textual ".phase-state" carries the
 * a11y truth (so a screen reader reads "Pending"/"Running…"/… not a glyph).
 *
 * Methods on the returned controller:
 *   setState(state)              swap the ring + state text + row class.
 *   setTaskProgress(done,total)  show "· 23/50 tasks" + drive the ring fill.
 *   addGate(gate)                append a gateChip to the row's gate strip.
 *   setGateSummary(passed,total,worst)  render/update the "N of M checks
 *                                passing" chip (green when all pass; calm amber
 *                                when warnings; red when a critical failed).
 *   expand()                     reveal the row's detail drawer.
 *
 * PURE PRESENTATION: takes data as args, calls no API.
 */

import { el, clear } from '../dom.js';
import { progressRing } from './ring.js';
import { gateChip } from './gate-chip.js';

const _STATE_TEXT = {
  pending: 'Pending',
  running: 'Running…',
  done: 'Done',
  skipped: 'Skipped',
  failed: 'Failed',
};

/**
 * Build a phase row.
 * @param {Object} phase
 * @param {string} phase.name     phase id (data-phase).
 * @param {string} [phase.label]  friendly label (falls back to the name).
 * @param {boolean} [phase.optional]
 * @param {string} [phase.state="pending"]
 * @param {number} [phase.size=28]  ring size in px.
 * @returns {{el: HTMLElement, setState, setTaskProgress, addGate, expand,
 *            state: ()=>string}}
 */
export function phaseRow(phase = {}) {
  const name = phase.name || '';
  const label = phase.label || name;
  let state = _STATE_TEXT[phase.state] ? phase.state : 'pending';
  const size = phase.size || 28;
  let percent = 0;

  const ringHolder = el('span', { class: 'phase-ring' });
  const stateEl = el('span', { class: 'phase-state', text: _STATE_TEXT[state] });
  const taskEl = el('span', { class: 'phase-tasks tabular-nums', hidden: true });
  const gateStrip = el('span', { class: 'phase-gates', role: 'group', 'aria-label': `Checks for ${label}` });
  // The per-phase "N of M checks passing" summary chip. Decorative-status (a
  // glyph paired with visible text); it is NOT a live region — the polite
  // announcement (when one happens) routes through the console's single
  // role=status line, never per-gate chatter. Hidden until a __summary__ line.
  const summaryChip = el('span', { class: 'phase-gate-summary', hidden: true, 'data-result': '' });
  gateStrip.appendChild(summaryChip);
  const detail = el('div', { class: 'phase-detail', hidden: true });

  function paintRing() {
    clear(ringHolder).appendChild(
      progressRing({ percent, state, size, label: _STATE_TEXT[state] }),
    );
  }
  paintRing();

  const li = el('li', { class: `phase-row is-${state}`, 'data-phase': name }, [
    ringHolder,
    el('span', { class: 'phase-label', text: label }),
    stateEl,
    taskEl,
    phase.optional ? el('span', { class: 'phase-opt', text: '(optional)' }) : null,
    gateStrip,
    detail,
  ]);

  function setState(next) {
    state = _STATE_TEXT[next] ? next : 'pending';
    li.className = `phase-row is-${state}`;
    stateEl.textContent = _STATE_TEXT[state];
    if (state === 'done') percent = 100;
    if (state === 'pending' || state === 'skipped') percent = 0;
    paintRing();
  }

  function setTaskProgress(done, total) {
    const d = Math.max(0, Number(done) || 0);
    const t = Math.max(0, Number(total) || 0);
    if (t > 0) {
      percent = Math.min(100, Math.round((d / t) * 100));
      taskEl.hidden = false;
      taskEl.textContent = `· ${d}/${t} tasks`;
      if (state === 'running') paintRing();
    } else {
      // Validator-only phases synthesize a virtual task with no meaningful
      // count → keep the indeterminate running spin (verdict-flagged caveat).
      taskEl.hidden = true;
      taskEl.textContent = '';
    }
  }

  function addGate(gate) {
    gateStrip.appendChild(gateChip(gate));
  }

  /**
   * Render/update the per-phase "N of M checks passing" summary chip.
   *
   * Non-color-only (WCAG 1.4.1): a glyph + visible text, NOT a bare colored dot.
   * The TONE follows the worst gate seen so warnings read CALM, not alarming:
   *   all pass         → ✓  green  "N of N checks passing"
   *   warnings present → △  amber  "N of M checks passing · K warnings"  (calm)
   *   a critical fail  → ✗  red    "N of M checks passing · K failed"
   *
   * @param {number} passed  gates that passed.
   * @param {number} total   total resolved gates.
   * @param {string} [worst="pass"]  the worst result seen on this phase
   *        (pass|warn|fail) — drives the tone/glyph, NOT color-only.
   */
  function setGateSummary(passed, total, worst) {
    const p = Math.max(0, Number(passed) || 0);
    const t = Math.max(0, Number(total) || 0);
    const failed = Math.max(0, t - p);
    const w = worst === 'fail' ? 'fail' : worst === 'warn' ? 'warn' : 'pass';
    const glyph = w === 'fail' ? '✗' : w === 'warn' ? '△' : '✓';
    // The visible suffix names the non-passing kind CALMLY (warnings, not alarm).
    let tail = '';
    if (w === 'warn' && failed > 0) tail = ` · ${failed} ${failed === 1 ? 'warning' : 'warnings'}`;
    else if (w === 'fail' && failed > 0) tail = ` · ${failed} failed`;
    const text = `${p} of ${t} checks passing${tail}`;
    summaryChip.setAttribute('data-result', w);
    summaryChip.className = `phase-gate-summary is-${w}`;
    clear(summaryChip);
    summaryChip.appendChild(el('span', { class: 'phase-gate-summary-glyph', 'aria-hidden': 'true', text: glyph }));
    summaryChip.appendChild(el('span', { class: 'phase-gate-summary-text', text }));
    summaryChip.hidden = false;
  }

  function expand() {
    detail.hidden = false;
  }

  return {
    el: li,
    detail,
    setState,
    setTaskProgress,
    addGate,
    setGateSummary,
    expand,
    state: () => state,
  };
}

export default phaseRow;
