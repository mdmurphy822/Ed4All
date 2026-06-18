/* elapsed — the single canonical elapsed timer. Vanilla ES module, NO build.
 *
 * Extracted VERBATIM from the two divergent setInterval copies
 * (create.js:491 + learn.js): a 1s-tick counter rendered with tabular-nums so
 * digits don't jitter.
 *
 * A11y decision preserved verbatim: the timer is DECORATION (aria-hidden) — a
 * per-second tick must NOT chatter to a screen reader. Truth (phase
 * transitions, terminal verdict) lives in the host page's single
 * role="status" aria-live region, never here.
 *
 * PURE PRESENTATION: takes a start time, calls no API.
 */

import { el } from '../dom.js';

/** Format a ms duration the way create.js::fmtElapsed does (kept identical). */
export function fmtElapsed(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/**
 * Build a live elapsed timer.
 * @param {Object} [opts]
 * @param {number} [opts.startMs=Date.now()]  epoch-ms the build started.
 * @param {string} [opts.prefix="elapsed "]   text prepended to the duration.
 * @param {boolean} [opts.autostart=true]     start ticking immediately.
 * @returns {{el: HTMLElement, start: ()=>void, stop: ()=>void,
 *            freeze: (ms:number)=>void, setStart: (ms:number)=>void}}
 *   `el` is a <span class="elapsed tabular-nums" aria-hidden="true">.
 *   `freeze` renders a fixed final duration and stops ticking (the terminal
 *   "finished" state). The caller MUST call stop()/freeze() on teardown.
 */
export function elapsed({ startMs = Date.now(), prefix = 'elapsed ', autostart = true } = {}) {
  let start = startMs;
  let timer = null;
  // aria-hidden: the verified no-screen-reader-chatter decision, verbatim.
  const node = el('span', { class: 'elapsed tabular-nums', 'aria-hidden': 'true', text: `${prefix}0s` });

  function render() {
    node.textContent = `${prefix}${fmtElapsed(Date.now() - start)}`;
  }
  function stop() {
    if (timer != null) { clearInterval(timer); timer = null; }
  }
  function startTick() {
    stop();
    render();
    timer = setInterval(render, 1000);
  }
  function freeze(ms) {
    stop();
    node.textContent = typeof ms === 'number' ? `${prefix}${fmtElapsed(ms)}` : 'finished';
  }
  function setStart(ms) {
    start = ms;
    render();
  }

  if (autostart) startTick();
  return { el: node, start: startTick, stop, freeze, setStart };
}

export default elapsed;
