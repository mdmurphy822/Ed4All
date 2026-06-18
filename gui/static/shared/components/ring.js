/* progressRing — vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * Takes data as args, calls NO API/fetch, wires no endpoints. This is a hard
 * security contract: the locked-down learner bundle can import this kit and it
 * structurally cannot pull in a run-control endpoint.
 *
 * Two concentric SVG <circle>s drive the arc via stroke-dasharray /
 * stroke-dashoffset. The ring is DECORATION: it carries aria-hidden="true" so a
 * screen reader never sees it — the truth is surfaced by the host page in a
 * single role="status" aria-live region (see the gallery / phase-row).
 *
 * States (color via tokens.css, never hardcoded):
 *   pending  → empty ring (--phase-pending)
 *   running  → animated fill (@keyframes spin), --phase-running
 *   done     → filled + check glyph, --phase-done
 *   skipped  → dashed ring at 0%, --phase-skipped
 *   failed   → --phase-failed + ✕ glyph
 */

import { el } from '../dom.js';

const _STATE_COLOR = {
  pending: 'var(--phase-pending)',
  running: 'var(--phase-running)',
  done: 'var(--phase-done)',
  skipped: 'var(--phase-skipped)',
  failed: 'var(--phase-failed)',
};

const SVG_NS = 'http://www.w3.org/2000/svg';

function _svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    node.setAttribute(k, String(v));
  }
  return node;
}

/**
 * Build a progress ring.
 * @param {Object} opts
 * @param {number} [opts.percent=0]   0..100 fill (clamped). Ignored for the
 *                                    indeterminate running spin.
 * @param {string} [opts.state="pending"]  pending|running|done|skipped|failed.
 * @param {number} [opts.size=28]     px diameter of the SVG.
 * @param {string} [opts.label=""]    optional visually-hidden text label (the
 *                                    a11y truth, mirrored into the wrapper for
 *                                    callers that don't supply their own live
 *                                    region). The SVG itself stays aria-hidden.
 * @returns {HTMLElement} a <span class="ring ring-<state>"> wrapper.
 */
export function progressRing({ percent = 0, state = 'pending', size = 28, label = '' } = {}) {
  const st = _STATE_COLOR[state] ? state : 'pending';
  const color = _STATE_COLOR[st];
  const stroke = Math.max(2, Math.round(size * 0.12));
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const circ = 2 * Math.PI * r;

  // Determinate fill fraction. running = indeterminate (CSS spin) so we paint a
  // fixed 25% arc and let the wrapper rotate; done = always full.
  let frac;
  if (st === 'done') frac = 1;
  else if (st === 'skipped') frac = 0;
  else if (st === 'running') frac = 0.25;
  else frac = Math.min(1, Math.max(0, percent / 100));

  const svg = _svgEl('svg', {
    class: 'ring-svg',
    width: size,
    height: size,
    viewBox: `0 0 ${size} ${size}`,
    'aria-hidden': 'true',
    focusable: 'false',
  });

  // Track (faint full circle).
  svg.appendChild(_svgEl('circle', {
    class: 'ring-track',
    cx, cy: cx, r,
    fill: 'none',
    stroke: 'var(--border)',
    'stroke-width': stroke,
    opacity: '0.35',
  }));

  // Progress arc. Skipped renders a dashed ring (a 0% fill would be invisible).
  const arc = _svgEl('circle', {
    class: 'ring-arc',
    cx, cy: cx, r,
    fill: 'none',
    stroke: color,
    'stroke-width': stroke,
    'stroke-linecap': 'round',
    'stroke-dasharray': st === 'skipped' ? `${(circ / 16).toFixed(2)} ${(circ / 16).toFixed(2)}` : circ.toFixed(2),
    'stroke-dashoffset': st === 'skipped' ? '0' : (circ * (1 - frac)).toFixed(2),
    transform: `rotate(-90 ${cx} ${cx})`,
  });
  svg.appendChild(arc);

  // done/failed glyphs are drawn as a centered text node so they recolor with
  // the state token and stay aria-hidden (the wrapper label carries meaning).
  if (st === 'done' || st === 'failed') {
    const glyph = _svgEl('text', {
      class: 'ring-glyph',
      x: cx, y: cx,
      'text-anchor': 'middle',
      'dominant-baseline': 'central',
      'font-size': Math.round(size * 0.5),
      fill: color,
      'aria-hidden': 'true',
    });
    glyph.textContent = st === 'done' ? '✓' : '✕';
    svg.appendChild(glyph);
  }

  const wrap = el('span', {
    class: `ring ring-${st}`,
    'data-state': st,
    'data-percent': String(Math.round(frac * 100)),
    style: `--ring-size:${size}px`,
  }, [svg]);
  // The wrapper carries the optional visually-hidden label. It is NOT a live
  // region — the host page owns the single role=status region.
  if (label) wrap.appendChild(el('span', { class: 'visually-hidden', text: label }));
  return wrap;
}

export default progressRing;
