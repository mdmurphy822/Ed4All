/* Ed4All Studio — Run History (GUI redesign Phase 4 §5.1(F)).
 *
 * A self-contained run-history list reachable via the studio shell's existing
 * hash router at #/runs — WITHOUT the Phase-2 auth remount and WITHOUT touching
 * routes/auth. It renders terminal (and live) runs from GET /api/runs as the
 * shared runCard()s, each with a timelineBar() of the persisted phase_durations,
 * plus two no-dead-end affordances:
 *
 *   - Re-attach  — open a still-running build's console (#/create/<run_id>), so a
 *                  page reload never strands a live build.
 *   - Run again  — Retry: re-POST /api/runs with the run's OWN params (no
 *                  re-upload), then navigate to the new run's progress view.
 *
 * PURE PRESENTATION + host-owns-fetch: this view owns its REST calls; the kit
 * (runCard / timelineBar) is data-fed and structurally fetch-free.
 *
 * A11y: a single <h1>; cards carry the shared statusPill (glyph + text, never
 * color-only); timelineBar's bar is aria-hidden with a semantic legend; all
 * actions are real <button>s. Status announcements route through the shell's
 * single #status role=status region (no new live region).
 */

import { api, apiJSON } from '/shared/api.js';
import { el, clear } from '/shared/dom.js';
import { toast, toastErr } from '/shared/toast.js';
import { runCard } from '/shared/components/card.js';
import { timelineBar } from '/shared/components/timeline-bar.js';

const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted', 'canceled', 'error']);

export async function renderRunHistory(shell) {
  shell.crumbs([{ text: 'Library', href: '#/library' }, { text: 'Run history' }]);
  const v = clear(shell.view());
  shell.setBusy(true);
  shell.setStatus('Loading run history.');

  v.appendChild(el('h1', { text: 'Run history' }));
  v.appendChild(el('p', { class: 'muted', text: 'Every course build, newest first. Re-open a running build, or run a finished one again from the same textbook.' }));

  let runs;
  try {
    const data = await api('/api/runs');
    runs = Array.isArray(data) ? data : (data && data.runs) || [];
  } catch (e) {
    shell.setBusy(false);
    v.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    toastErr(e, 'Could not load run history');
    return;
  }

  if (!runs.length) {
    v.appendChild(el('p', { class: 'muted', text: 'No builds yet. Create a course to see it here.' }));
    v.appendChild(el('a', { class: 'btn primary', href: '#/create', text: 'Create a course' }));
    shell.setBusy(false);
    shell.setStatus('No builds yet.');
    return;
  }

  const list = el('ul', { class: 'run-history-list cards', 'aria-label': 'Course builds' });
  runs.forEach((r) => list.appendChild(el('li', { class: 'run-history-li' }, [runHistoryEntry(shell, r)])));
  v.appendChild(list);

  shell.setBusy(false);
  shell.setStatus(`${runs.length} build${runs.length === 1 ? '' : 's'} in history.`);
}

/** One run entry: a runCard + (when present) a timelineBar + actions. */
function runHistoryEntry(shell, r) {
  const runId = r.run_id || r.runId || r.id || '';
  const status = normalizeStatus(r.status || '');
  const workflow = r.workflow || r.workflow_name || '';
  const isTerminal = TERMINAL.has(r.status || '');

  // Meta line: workflow + total duration (from persisted phase_durations).
  const durs = Array.isArray(r.phase_durations) ? r.phase_durations : [];
  const totalMs = durs.reduce((s, p) => s + Math.max(0, Number(p.duration_ms) || 0), 0);
  const metaParts = [workflow].filter(Boolean);
  if (totalMs > 0) metaParts.push(fmtDur(totalMs));
  else if (!isTerminal) metaParts.push('building');

  const wrap = el('div', { class: 'run-history-entry' });

  // The card (re-attach link is the card href for a still-running build; a
  // terminal run links to its frozen progress view too).
  const card = runCard({
    title: r.course_name || runId,
    status,
    href: runId ? `#/create/${encodeURIComponent(runId)}` : undefined,
    meta: metaParts.join(' · '),
  });
  wrap.appendChild(card);

  // The persisted timeline bar (run-history phase durations) when we have them.
  if (durs.length) {
    wrap.appendChild(timelineBar(
      durs.map((p) => ({
        name: p.name || p.phase || 'phase',
        label: p.label || p.name || p.phase || 'phase',
        duration_ms: Number(p.duration_ms) || 0,
        state: p.state || (p.skipped ? 'skipped' : 'done'),
      })),
      { ariaLabel: `Phase durations for ${r.course_name || runId}` },
    ));
  }

  // Actions: Re-attach (live) / Run again (terminal). Real <button>s.
  const actions = el('div', { class: 'run-history-actions' });
  if (!isTerminal && runId) {
    const reattach = el('button', { type: 'button', class: 'btn', text: 'Re-open build' });
    reattach.addEventListener('click', () => { location.hash = `#/create/${encodeURIComponent(runId)}`; });
    actions.appendChild(reattach);
  }
  // Run again (Retry) — needs the run's recorded params (corpus / upload).
  const params = r.params && typeof r.params === 'object' ? r.params : null;
  if (params && (params.corpus || params.upload_id)) {
    const again = el('button', { type: 'button', class: 'btn', text: 'Run again' });
    again.addEventListener('click', () => runAgain(shell, r));
    actions.appendChild(again);
  }
  if (actions.childElementCount) wrap.appendChild(actions);

  return wrap;
}

/** Re-POST /api/runs from a history run's OWN params (no re-upload). */
async function runAgain(shell, r) {
  const p = r.params || {};
  const body = {
    workflow: r.workflow || 'textbook_to_course',
    course_name: r.course_name || p.course_name || '',
    corpus: p.corpus || p.upload_id,
    options: p.options || {},
  };
  if (p.weeks) body.weeks = Number(p.weeks);
  if (p.mode) body.mode = p.mode;
  if (p.provider) body.provider = p.provider;
  if (p.model) body.model = p.model;
  try {
    const resp = await apiJSON('/api/runs', 'POST', body);
    toast('Started.', '', 'success');
    location.hash = `#/create/${encodeURIComponent(resp.run_id)}`;
  } catch (e) {
    toastErr(e, 'Could not start');
    shell.setStatus(`Could not start: ${shell.errText(e)}`);
  }
}

/** Map backend status strings onto the statusPill run keys. */
function normalizeStatus(s) {
  if (s === 'canceled') return 'cancelled';
  if (s === 'error') return 'failed';
  if (s === 'requested') return 'queued';
  return s || 'queued';
}

function fmtDur(ms) {
  const sec = Math.max(0, Math.round(ms / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const rem = sec % 60;
  if (m < 60) return rem ? `${m}m ${rem}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export default renderRunHistory;
