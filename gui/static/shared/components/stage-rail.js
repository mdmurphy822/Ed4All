/* stageRail — the animated pipeline stage-tracker rail + live stats band.
 * Vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * Renders GET /api/runs/{run_id}/progress as:
 *   - a horizontal connected rail of phase nodes (wraps on narrow widths),
 *     visually grouped by the server-derived `group` (conversion / planning /
 *     generation / validation / packaging / archive) — the phase LIST and the
 *     grouping both come from the payload (config-driven), never hardcoded;
 *   - node states: done ✓ · current (CSS pulse; static highlight under
 *     prefers-reduced-motion) · pending ○ · failed ✗ · skipped – (dimmed);
 *   - per-node wall-clock for completed phases (small text under the node);
 *   - a stats band: tok/s, LLM calls, prompt+completion tokens, TTFT p50,
 *     elapsed-in-phase, and the serving model seat.
 *
 * PURE PRESENTATION + host-owns-fetch: the host page owns the poll loop and
 * calls update(payload); this module never fetches (the kit contract that lets
 * the locked-down learner bundle import kit files safely).
 *
 * A11y: the rail is a semantic <ol>; every node pairs its glyph with a
 * visually-hidden state word (never color-only); the whole section is
 * aria-labelled; live digits use .tabular-nums. No live region here — the Build
 * Console's single role=status line stays the only announcer on the page.
 */

import { el, clear } from '../dom.js';

const STATE_GLYPHS = { done: '✓', current: '●', pending: '○', failed: '✗', skipped: '–' };
const STATE_WORDS = { done: 'done', current: 'in progress', pending: 'pending', failed: 'failed', skipped: 'skipped' };

function fmtDur(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return s % 60 ? `${m}m ${s % 60}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function fmtTokens(n) {
  if (!Number.isFinite(n)) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  return String(n);
}

/** Prettify a phase id when the payload carries no friendly label. */
function titleWords(name) {
  return String(name || '').replace(/_/g, ' ');
}

/**
 * Build the stage rail + stats band.
 * @param {Object} [opts]
 * @param {string} [opts.ariaLabel="Pipeline progress"]
 * @returns {{el: HTMLElement, update: (payload:Object)=>void}}
 */
export function stageRail(opts = {}) {
  const ariaLabel = opts.ariaLabel || 'Pipeline progress';
  const railHost = el('div', { class: 'stage-rail-host' });
  const statsHost = el('dl', { class: 'stage-stats' });
  const root = el('section', { class: 'stage-rail kit', 'aria-label': ariaLabel }, [
    railHost,
    statsHost,
  ]);

  function renderRail(payload) {
    clear(railHost);
    const phases = Array.isArray(payload.phases) ? payload.phases : [];
    if (!phases.length) return;

    // Group CONSECUTIVE phases sharing `group` into labelled segments so the
    // rail reads conversion → planning → generation → … (server-derived).
    const groups = [];
    phases.forEach((p) => {
      const g = p.group || 'other';
      const last = groups[groups.length - 1];
      if (last && last.group === g) last.phases.push(p);
      else groups.push({ group: g, phases: [p] });
    });

    const ol = el('ol', { class: 'stage-rail-groups' });
    groups.forEach((g) => {
      const nodes = el('ol', { class: 'stage-group-list' });
      g.phases.forEach((p) => {
        const state = STATE_GLYPHS[p.state] ? p.state : 'pending';
        const label = p.label && p.label !== p.name ? p.label : titleWords(p.name);
        const wall = typeof p.wallclock_s === 'number' ? fmtDur(p.wallclock_s) : '';
        // The current node's dot carries the `stage-pulse` motion utility —
        // its animation (and reduced-motion static-highlight swap) lives in
        // tokens.css, the single motion layer.
        const li = el('li', {
          class: `stage-node is-${state}`,
          title: wall ? `${label} — ${wall}` : label,
        }, [
          el('span', {
            class: state === 'current' ? 'stage-node-dot stage-pulse' : 'stage-node-dot',
            'aria-hidden': 'true',
            text: STATE_GLYPHS[state],
          }),
          el('span', { class: 'stage-node-name', text: titleWords(p.name) }),
          el('span', { class: 'visually-hidden', text: ` — ${STATE_WORDS[state]}` }),
          wall
            ? el('span', { class: 'stage-node-time tabular-nums', text: wall })
            : el('span', { class: 'stage-node-time', 'aria-hidden': 'true', text: ' ' }),
        ]);
        nodes.appendChild(li);
      });
      ol.appendChild(el('li', { class: 'stage-group' }, [
        el('span', { class: 'stage-group-label', text: g.group }),
        nodes,
      ]));
    });
    railHost.appendChild(ol);
  }

  function statRow(term, valueNode) {
    return el('div', { class: 'stage-stat' }, [
      el('dt', { text: term }),
      valueNode,
    ]);
  }

  function renderStats(payload) {
    clear(statsHost);
    const s = payload.stats || {};
    const rows = [];
    rows.push(statRow('tok/s', el('dd', {
      class: 'tabular-nums',
      text: typeof s.tok_s === 'number' ? String(s.tok_s) : '—',
    })));
    rows.push(statRow('LLM calls', el('dd', {
      class: 'tabular-nums',
      text: Number.isFinite(s.calls) ? String(s.calls) : '—',
    })));
    rows.push(statRow('tokens', el('dd', {
      class: 'tabular-nums',
      text: `${fmtTokens(s.prompt_tokens)} in · ${fmtTokens(s.completion_tokens)} out`,
    })));
    if (typeof s.ttft_p50_ms === 'number') {
      rows.push(statRow('TTFT p50', el('dd', {
        class: 'tabular-nums',
        text: s.ttft_p50_ms >= 1000 ? `${(s.ttft_p50_ms / 1000).toFixed(1)}s` : `${Math.round(s.ttft_p50_ms)}ms`,
      })));
    }
    if (typeof s.phase_elapsed_s === 'number') {
      rows.push(statRow('in this step', el('dd', {
        class: 'tabular-nums',
        text: fmtDur(s.phase_elapsed_s),
      })));
    }
    if (s.seat && (s.seat.model || s.seat.name)) {
      const seatText = [s.seat.name, s.seat.model].filter(Boolean).join(' · ');
      rows.push(statRow('serving', el('dd', { text: seatText })));
    }
    rows.forEach((r) => statsHost.appendChild(r));
  }

  function update(payload) {
    if (!payload || typeof payload !== 'object') return;
    renderRail(payload);
    renderStats(payload);
  }

  return { el: root, update };
}

export default stageRail;
