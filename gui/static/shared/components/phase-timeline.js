/* phaseTimeline — vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * The checklist container: an ordered <ol class="phase-checklist"> of phaseRow
 * components, preserving the create.js semantic-list + accessible-name pattern
 * (aria-label="Course build steps"). Owns the sequential running-phase
 * inference (the create.js "first not-done becomes running" heuristic) so the
 * Build Console can drive it from the [phase] WS line stream unchanged.
 *
 * PURE PRESENTATION: takes the phase list as args, calls no API.
 */

import { el } from '../dom.js';
import { phaseRow } from './phase-row.js';

/**
 * Build a phase timeline.
 * @param {Array<{name,label,optional}>} phases  ordered phase descriptors.
 * @param {Object} [opts]
 * @param {string} [opts.ariaLabel="Course build steps"]
 * @param {number} [opts.ringSize=28]
 * @returns {{el, setState, setTaskProgress, addGate, setGateSummary,
 *            markCompleted, startRunning, get}}
 */
export function phaseTimeline(phases = [], opts = {}) {
  const ariaLabel = opts.ariaLabel || 'Course build steps';
  const ringSize = opts.ringSize || 28;

  const ol = el('ol', { class: 'phase-checklist', 'aria-label': ariaLabel });
  const rows = new Map();
  const order = [];
  const done = new Set();

  phases.forEach((p) => {
    const row = phaseRow({ name: p.name, label: p.label, optional: p.optional, size: ringSize });
    rows.set(p.name, row);
    order.push(p.name);
    ol.appendChild(row.el);
  });

  function get(name) { return rows.get(name) || null; }

  function setState(name, state) {
    const row = rows.get(name);
    if (row) row.setState(state);
  }

  function setTaskProgress(name, doneN, total) {
    const row = rows.get(name);
    if (row) row.setTaskProgress(doneN, total);
  }

  function addGate(name, gate) {
    const row = rows.get(name);
    if (row) row.addGate(gate);
  }

  function setGateSummary(name, passed, total, worst) {
    const row = rows.get(name);
    if (row) row.setGateSummary(passed, total, worst);
  }

  /** Mark a phase done|skipped, then promote the first not-done to running.
   * Mirrors create.js::markCompleted (best-effort sequential inference). */
  function markCompleted(name, marker) {
    setState(name, marker === 'skipped' ? 'skipped' : 'done');
    done.add(name);
    const next = order.find((n) => !done.has(n));
    if (next) { setState(next, 'running'); return next; }
    return null;
  }

  /** Seed the first phase as running (create.js does this when no WS line has
   * arrived yet). Returns the running phase name (or null when empty). */
  function startRunning() {
    if (!order.length) return null;
    const first = order.find((n) => !done.has(n));
    if (first) { setState(first, 'running'); return first; }
    return null;
  }

  return { el: ol, setState, setTaskProgress, addGate, setGateSummary, markCompleted, startRunning, get, order };
}

export default phaseTimeline;
