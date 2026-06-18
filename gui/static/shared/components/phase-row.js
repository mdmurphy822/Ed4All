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

  function expand() {
    detail.hidden = false;
  }

  return {
    el: li,
    detail,
    setState,
    setTaskProgress,
    addGate,
    expand,
    state: () => state,
  };
}

export default phaseRow;
