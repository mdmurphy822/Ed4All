/* timelineBar — a stacked horizontal bar of run-history phase durations.
 * Vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * Renders "this build normally takes 28 min; phase X is the slow one" as one
 * proportional bar of phase segments. The bar is DECORATION (aria-hidden); the
 * a11y truth is a semantic <ul> of "<phase> — <duration>" rows that a screen
 * reader reads (the segments alone would be color-only, WCAG 1.4.1).
 *
 * PURE PRESENTATION: takes the duration vector as args, calls no API.
 */

import { el } from '../dom.js';

function _fmtDur(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return rem ? `${m}m ${rem}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/**
 * Build a stacked timeline bar.
 * @param {Array<{name,label,duration_ms,state}>} phases  per-phase durations.
 * @param {Object} [opts]
 * @param {string} [opts.ariaLabel="Phase durations"]
 * @returns {HTMLElement} a <figure class="timeline-bar">.
 */
export function timelineBar(phases = [], opts = {}) {
  const ariaLabel = opts.ariaLabel || 'Phase durations';
  const total = phases.reduce((sum, p) => sum + Math.max(0, Number(p.duration_ms) || 0), 0);

  // The visual bar: aria-hidden segments sized proportionally.
  const bar = el('div', { class: 'tl-bar-track', 'aria-hidden': 'true' });
  phases.forEach((p) => {
    const dur = Math.max(0, Number(p.duration_ms) || 0);
    const pct = total > 0 ? (dur / total) * 100 : 0;
    const state = p.state || 'done';
    bar.appendChild(el('span', {
      class: `tl-seg tl-seg-${state}`,
      'data-phase': p.name || '',
      style: `width:${pct.toFixed(2)}%`,
      title: `${p.label || p.name || 'phase'} — ${_fmtDur(dur)}`,
    }));
  });

  // The a11y truth: a semantic list with text durations (never color-only).
  const list = el('ul', { class: 'tl-legend', 'aria-label': ariaLabel });
  phases.forEach((p) => {
    list.appendChild(el('li', { class: 'tl-legend-item' }, [
      el('span', { class: `tl-legend-swatch tl-seg-${p.state || 'done'}`, 'aria-hidden': 'true' }),
      el('span', { class: 'tl-legend-name', text: p.label || p.name || 'phase' }),
      el('span', { class: 'tl-legend-dur tabular-nums', text: _fmtDur(Math.max(0, Number(p.duration_ms) || 0)) }),
    ]));
  });

  return el('figure', { class: 'timeline-bar' }, [
    bar,
    el('figcaption', { class: 'visually-hidden', text: `${ariaLabel}: total ${_fmtDur(total)}` }),
    list,
  ]);
}

export default timelineBar;
