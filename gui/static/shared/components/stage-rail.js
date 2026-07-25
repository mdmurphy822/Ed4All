/* stageRail — the animated pipeline stage-tracker rail + live stats band.
 * Vanilla ES module, NO build step, PURE PRESENTATION.
 *
 * Renders GET /api/runs/{run_id}/progress as:
 *   - a horizontal connected rail of phase nodes (wraps on narrow widths),
 *     visually grouped by the server-derived `group` (conversion / planning /
 *     generation / validation / packaging / archive) — the phase LIST and the
 *     grouping both come from the payload (config-driven), never hardcoded;
 *     phases are BUCKETED by group (first-occurrence order, within-group
 *     phase order preserved) so each section header renders exactly once;
 *   - node states: done ✓ · current (CSS pulse; static highlight under
 *     prefers-reduced-motion) · pending ○ · failed ✗ · skipped – (dimmed);
 *   - per-node time (small text under the node): completed phases show their
 *     final wall-clock (sub-second phases render "<1s", never blank); the
 *     in-progress phase shows a LIVE elapsed (payload `elapsed_s` = now − start,
 *     repainted every poll) so the active phase never sits time-less;
 *   - a stats band: tok/s (AGGREGATE seat throughput — window tokens over the
 *     wall span, what the seat produces overall), streams ("N in flight" —
 *     the server's estimated concurrent request count, queued+decoding),
 *     LLM calls, prompt+completion tokens, TTFT p50, elapsed-in-phase, real
 *     in-phase unit progress (stats.phase_units — the server-counted rows of
 *     the phase's own resume sidecar; rendered only when the server sent it,
 *     never a fabricated 0), and the serving model seat;
 *   - a collapsible "Detailed stats" disclosure (native <details>/<summary>,
 *     the project's established pattern) fed by stats.detail: run totals,
 *     the throughput trio (window aggregate / per-stream / cumulative avg
 *     tok/s), TTFT p50/p95, call-duration mean/median, the
 *     truncation + missing-usage health tripwires, a per-(provider · model)
 *     table, a per-phase token-attribution table (incl. the server's explicit
 *     "unattributed" bucket), and the latest VRAM sample. Hidden entirely
 *     when the server omitted stats.detail (the honesty contract — sections
 *     render only from real data). Open/closed state survives the 3s poll
 *     (only the body is rebuilt). Tables sit in an overflow-x wrapper so wide
 *     content scrolls locally, never the page. No live region — silent DOM
 *     refreshes, same as the band;
 *   - a "Live output" panel below the disclosure (updateTail, fed by GET
 *     /api/runs/{id}/output-tail): the current phase's real sidecar tail as
 *     bounded monospace rows, local vertical scroll auto-stuck to newest
 *     unless the user scrolled up; hidden when the server sent no rows.
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
// Friendly HEADER label per server-derived `group`. The pipeline groups
// (conversion / planning / generation / validation / packaging / archive)
// render their raw key; the trailing post-build LoRA-training group carries a
// proper header so it reads consistently with the others.
const GROUP_LABELS = { training: 'Training' };

function fmtDur(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  // Honest sub-second rendering: a fast deterministic phase (staging /
  // source_mapping complete in ~0.1-0.3s) must show "<1s", never a blank.
  if (seconds < 1) return '<1s';
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
  const detailBody = el('div', { class: 'stage-detail-body' });
  // Created ONCE and kept across updates so the native open/closed state the
  // user chose survives the poll; only detailBody is rebuilt.
  const detailHost = el('details', { class: 'stage-detail' }, [
    el('summary', { text: 'Detailed stats' }),
    detailBody,
  ]);
  detailHost.hidden = true;
  // "Live output" panel — REAL pipeline output (the current phase's resume
  // sidecar tail from GET /api/runs/{id}/output-tail) as scrolling monospace
  // text, newest at the bottom. Sits BELOW the collapsed "Detailed stats"
  // disclosure (owner: actual output at the page bottom; stats stay
  // reachable above). Created ONCE; rows rebuild silently per poll and the
  // view auto-sticks to the newest row unless the user scrolled up (standard
  // log-tail UX). Hidden entirely when the server sent no rows (honesty
  // contract — nothing fabricated, no empty shell). No live region — the
  // page's single announcer stays elsewhere.
  const tailHeading = el('p', { class: 'stage-output-heading', text: 'Live output' });
  const tailList = el('ol', { class: 'stage-output-list' });
  const tailScroll = el('div', {
    class: 'stage-output-scroll',
    tabindex: '0',
    role: 'group',
    'aria-label': 'Live output',
  }, [tailList]);
  const tailHost = el('section', { class: 'stage-output' }, [tailHeading, tailScroll]);
  tailHost.hidden = true;
  const root = el('section', { class: 'stage-rail kit', 'aria-label': ariaLabel }, [
    railHost,
    statsHost,
    detailHost,
    tailHost,
  ]);

  function renderRail(payload) {
    clear(railHost);
    const phases = Array.isArray(payload.phases) ? payload.phases : [];
    if (!phases.length) return;

    // Bucket phases by the server-derived `group` (first-occurrence order,
    // within-group phase order preserved) so each section header renders
    // exactly ONCE — conversion → planning → generation → validation →
    // packaging → archive — even when the pipeline's phase order interleaves
    // groups (owner design: one consolidated "generation" section).
    const groups = [];
    const byGroup = new Map();
    phases.forEach((p) => {
      const g = p.group || 'other';
      let bucket = byGroup.get(g);
      if (!bucket) {
        bucket = { group: g, phases: [] };
        byGroup.set(g, bucket);
        groups.push(bucket);
      }
      bucket.phases.push(p);
    });

    const ol = el('ol', { class: 'stage-rail-groups' });
    groups.forEach((g) => {
      const nodes = el('ol', { class: 'stage-group-list' });
      g.phases.forEach((p) => {
        const state = STATE_GLYPHS[p.state] ? p.state : 'pending';
        const label = p.label && p.label !== p.name ? p.label : titleWords(p.name);
        // Completed phases show their final wall-clock; the IN-PROGRESS phase
        // shows a LIVE elapsed (elapsed_s = now − start, recomputed server-side
        // and repainted on every poll). Both run through the same
        // sub-second-honest formatter, so a fast deterministic phase is never
        // blank and the active phase never sits time-less until it completes.
        const nodeSeconds = state === 'current' && typeof p.elapsed_s === 'number'
          ? p.elapsed_s
          : (typeof p.wallclock_s === 'number' ? p.wallclock_s : null);
        const wall = nodeSeconds !== null ? fmtDur(nodeSeconds) : '';
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
        el('span', { class: 'stage-group-label', text: GROUP_LABELS[g.group] || g.group }),
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
    // Estimated concurrent requests on the seat (queued + decoding). The
    // server OMITS the field when it isn't computable — absence renders
    // nothing (no fabricated 0).
    if (typeof s.streams === 'number') {
      rows.push(statRow('streams', el('dd', {
        class: 'tabular-nums',
        text: `${s.streams} in flight`,
      })));
    }
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
    // Real per-unit progress inside the current phase (e.g. "blocks done"),
    // counted server-side from the pipeline's resume-checkpoint sidecar. The
    // server OMITS the field when no sidecar exists, so absence renders
    // nothing here (the honesty contract: no fabricated counts, no estimated
    // totals). Not a live region — the band is polled presentation.
    if (s.phase_units && Number.isFinite(s.phase_units.count)) {
      rows.push(statRow(s.phase_units.label || 'units done', el('dd', {
        class: 'tabular-nums',
        text: String(s.phase_units.count),
      })));
    }
    if (s.seat && (s.seat.model || s.seat.name)) {
      const seatText = [s.seat.name, s.seat.model].filter(Boolean).join(' · ');
      rows.push(statRow('serving', el('dd', { text: seatText })));
    }
    rows.forEach((r) => statsHost.appendChild(r));
  }

  /** Format a millisecond figure like the band's TTFT (1.2s / 340ms). */
  function fmtMs(ms) {
    if (!Number.isFinite(ms)) return '—';
    return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
  }

  /** A compact term/value line inside the detail disclosure. */
  function detailLine(term, text) {
    return el('div', { class: 'stage-stat' }, [
      el('dt', { text: term }),
      el('dd', { class: 'tabular-nums', text }),
    ]);
  }

  /** A per-model / per-phase breakdown table in a local-scroll wrapper. */
  function detailTable(heading, headers, rows) {
    const thead = el('thead', {}, [
      el('tr', {}, headers.map((h) => el('th', { scope: 'col', text: h }))),
    ]);
    const tbody = el('tbody', {}, rows.map((cells) => el('tr', {}, cells.map((c, i) => el('td', {
      class: i >= headers.length - 3 ? 'tabular-nums' : '',
      text: c,
    })))));
    return el('div', { class: 'stage-detail-section' }, [
      el('p', { class: 'stage-detail-heading', text: heading }),
      el('div', { class: 'stage-detail-scroll' }, [
        el('table', { class: 'stage-detail-table' }, [thead, tbody]),
      ]),
    ]);
  }

  function renderDetail(payload) {
    const d = (payload.stats || {}).detail;
    clear(detailBody);
    // The server OMITS stats.detail when no source file exists — hide the
    // whole disclosure rather than rendering an empty shell (honesty
    // contract: nothing fabricated, nothing implied).
    if (!d || typeof d !== 'object') {
      detailHost.hidden = true;
      return;
    }
    detailHost.hidden = false;

    const lines = el('dl', { class: 'stage-stats stage-detail-lines' });
    const t = d.totals;
    if (t) {
      lines.appendChild(detailLine('calls', Number.isFinite(t.calls) ? String(t.calls) : '—'));
      lines.appendChild(detailLine(
        'tokens',
        `${fmtTokens(t.prompt_tokens)} in · ${fmtTokens(t.completion_tokens)} out · ${fmtTokens(t.total_tokens)} total`,
      ));
    }
    if (d.throughput) {
      // Three clearly-named figures: window aggregate (matches the band's
      // tok/s), per-stream (what one request experiences under concurrency),
      // and the cumulative run average.
      if (typeof d.throughput.window_tok_s === 'number') {
        lines.appendChild(detailLine('window tok/s (aggregate)', String(d.throughput.window_tok_s)));
      }
      if (typeof d.throughput.per_stream_tok_s === 'number') {
        lines.appendChild(detailLine('per-stream tok/s', String(d.throughput.per_stream_tok_s)));
      }
      if (typeof d.throughput.avg_tok_s === 'number') {
        lines.appendChild(detailLine('avg tok/s (cumulative)', String(d.throughput.avg_tok_s)));
      }
    }
    const lat = d.latency;
    if (lat && (typeof lat.ttft_p50_ms === 'number' || typeof lat.ttft_p95_ms === 'number')) {
      lines.appendChild(detailLine('TTFT p50 / p95', `${fmtMs(lat.ttft_p50_ms)} / ${fmtMs(lat.ttft_p95_ms)}`));
    }
    if (lat && (typeof lat.duration_mean_ms === 'number' || typeof lat.duration_median_ms === 'number')) {
      lines.appendChild(detailLine('call time mean / median', `${fmtMs(lat.duration_mean_ms)} / ${fmtMs(lat.duration_median_ms)}`));
    }
    if (d.health) {
      // Tripwires: a real 0 is worth showing (it means "no truncation seen").
      lines.appendChild(detailLine('truncated (finish=length)', String(d.health.truncated_calls ?? '—')));
      lines.appendChild(detailLine('usage missing', String(d.health.usage_missing_calls ?? '—')));
    }
    if (d.vram && d.vram.latest) {
      const v = d.vram.latest;
      const free = Number.isFinite(v.free_mib) ? fmtTokens(v.free_mib) : '—';
      const total = Number.isFinite(v.total_mib) ? fmtTokens(v.total_mib) : '—';
      const where = [v.phase, v.when].filter(Boolean).join(' · ');
      lines.appendChild(detailLine(
        'VRAM free/total (MiB)',
        `${free} / ${total}${where ? ` (${where})` : ''} · ${d.vram.samples} samples`,
      ));
    }
    if (lines.childNodes.length) detailBody.appendChild(lines);

    if (Array.isArray(d.by_model) && d.by_model.length) {
      detailBody.appendChild(detailTable(
        'By model',
        ['provider', 'model', 'calls', 'in', 'out'],
        d.by_model.map((m) => [
          m.provider || '—', m.model || '—',
          String(m.calls), fmtTokens(m.prompt_tokens), fmtTokens(m.completion_tokens),
        ]),
      ));
    }
    if (Array.isArray(d.by_phase) && d.by_phase.length) {
      detailBody.appendChild(detailTable(
        'By phase',
        ['phase', 'calls', 'in', 'out'],
        // The server includes an "unattributed" row only when it is nonzero
        // (rows outside every checkpoint window — never guessed into a phase).
        d.by_phase.map((p) => [
          titleWords(p.phase), String(p.calls),
          fmtTokens(p.prompt_tokens), fmtTokens(p.completion_tokens),
        ]),
      ));
    }
  }

  /**
   * Render the live-output tail (GET /api/runs/{id}/output-tail payload).
   * Separate from update() because the host polls the two endpoints
   * independently; absence of rows hides the whole panel.
   */
  function updateTail(payload) {
    if (!payload || typeof payload !== 'object') return;
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    if (!rows.length) {
      tailHost.hidden = true;
      clear(tailList);
      return;
    }
    tailHost.hidden = false;
    // Self-label from the server: unit count + sidecar basename (real data
    // only — both come straight off the payload).
    const bits = ['Live output'];
    if (Number.isFinite(payload.row_count) && payload.label) {
      bits.push(`${payload.row_count} ${payload.label}`);
    }
    if (payload.source) bits.push(payload.source);
    tailHeading.textContent = bits.join(' · ');
    // Auto-scroll to newest unless the user scrolled up (log-tail UX).
    const nearBottom = tailScroll.scrollHeight - tailScroll.scrollTop - tailScroll.clientHeight < 8;
    clear(tailList);
    rows.forEach((r) => {
      const li = el('li', { class: 'stage-output-row' });
      const label = [
        typeof r.label === 'string' && r.label ? r.label : '',
        Number.isFinite(r.seq) ? `#${r.seq}` : '',
      ].filter(Boolean).join(' ');
      if (label) li.appendChild(el('span', { class: 'stage-output-label', text: label }));
      li.appendChild(el('span', { class: 'stage-output-text', text: String(r.text || '') }));
      tailList.appendChild(li);
    });
    if (nearBottom) tailScroll.scrollTop = tailScroll.scrollHeight;
  }

  function update(payload) {
    if (!payload || typeof payload !== 'object') return;
    renderRail(payload);
    renderStats(payload);
    renderDetail(payload);
  }

  return { el: root, update, updateTail };
}

export default stageRail;
