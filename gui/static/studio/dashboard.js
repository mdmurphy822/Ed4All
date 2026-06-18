/* Ed4All Studio — Author Dashboard (GUI redesign Phase 2 follow-up §4.3/§8).
 *
 * The home route (#/) "state-at-a-glance" view. Three semantic <section> zones,
 * each with its own heading, plus an onboarding empty state:
 *
 *   - "Building now"  → live runs (queued|running|cancel_requested) as runCard()s,
 *                       each re-opening its #/build/<run_id> console.
 *   - "Your courses"  → the GET /api/library grid as courseCard()s → the viewer.
 *   - "Recent builds" → terminal runs as runCard() + a timelineBar() of the
 *                       persisted phase_durations, with Re-open (build console)
 *                       and Run again (Retry: re-POST /api/runs with the run's
 *                       OWN params, no re-upload). Links to the full #/runs view.
 *
 * Onboarding: when there are NO courses AND NO runs, a single primary path
 * "Build your first course" → #/create (via emptyState, which requires a CTA so
 * no zone ever ships a dead end). Each populated-but-other-zone-empty case still
 * offers a sensible next step.
 *
 * PURE PRESENTATION + host-owns-fetch: this view owns its two REST calls
 * (GET /api/runs, GET /api/library); the shared kit (courseCard / runCard /
 * timelineBar / emptyState) is data-fed and structurally fetch-free.
 *
 * A11y: each zone is a <section aria-labelledby> with a heading (the view keeps
 * its single <h1>; zone headings are <h2>). Cards are keyboard-operable links
 * (runCard/courseCard render <a class="card"> when given an href). Status
 * announcements route through the shell's single #status role=status region —
 * no new live region. Reduced-motion is handled in CSS.
 */

import { api, apiJSON } from '/shared/api.js';
import { el, clear } from '/shared/dom.js';
import { toast, toastErr } from '/shared/toast.js';
import { courseCard, runCard } from '/shared/components/card.js';
import { timelineBar } from '/shared/components/timeline-bar.js';
import { emptyState } from '/shared/components/empty-state.js';

const LIVE = new Set(['queued', 'running', 'cancel_requested', 'requested']);
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted', 'canceled', 'error']);

export async function renderDashboard(shell) {
  shell.crumbs([{ text: 'Dashboard' }]);
  const v = clear(shell.view());
  shell.setBusy(true);
  shell.setStatus('Loading your dashboard.');

  v.appendChild(el('h1', { text: 'Dashboard' }));

  // Fetch both feeds in parallel; each degrades independently (a failed feed
  // renders its own inline error rather than blanking the whole dashboard).
  const [runsRes, libRes] = await Promise.allSettled([
    api('/api/runs'),
    api('/api/library'),
  ]);

  const runs = runsRes.status === 'fulfilled' ? normalizeRuns(runsRes.value) : [];
  const courses = libRes.status === 'fulfilled' && Array.isArray(libRes.value) ? libRes.value : [];
  const runsErr = runsRes.status === 'rejected' ? runsRes.reason : null;
  const libErr = libRes.status === 'rejected' ? libRes.reason : null;

  const live = runs.filter((r) => LIVE.has(r.status));
  const recent = runs.filter((r) => TERMINAL.has(r.status));

  // Onboarding: nothing built and nothing in flight, and both feeds healthy.
  if (!courses.length && !runs.length && !runsErr && !libErr) {
    v.appendChild(emptyState({
      title: 'Build your first course',
      message: 'Turn a textbook PDF into an accessible, ask-ready course. Upload a PDF and we do the rest.',
      cta: { label: 'Build your first course', href: '#/create' },
    }));
    shell.setBusy(false);
    shell.setStatus('No courses yet. Build your first course.');
    return;
  }

  // Zone 1 — Building now (omitted entirely when nothing is live).
  if (live.length) {
    v.appendChild(buildingNowSection(live));
  }

  // Zone 2 — Your courses.
  v.appendChild(coursesSection(shell, courses, libErr));

  // Zone 3 — Recent builds.
  v.appendChild(recentSection(shell, recent, runsErr));

  shell.setBusy(false);
  const parts = [];
  if (live.length) parts.push(`${live.length} building`);
  parts.push(`${courses.length} course${courses.length === 1 ? '' : 's'}`);
  parts.push(`${recent.length} past build${recent.length === 1 ? '' : 's'}`);
  shell.setStatus(`Dashboard ready: ${parts.join(', ')}.`);
}

/* ----------------------------------------------------------- Building now */

function buildingNowSection(live) {
  const sec = section('dash-building-h', 'Building now',
    'Course builds in progress. Re-open one to watch its console.');
  const list = el('ul', { class: 'dash-cards cards', 'aria-label': 'Builds in progress' });
  live.forEach((r) => list.appendChild(el('li', { class: 'dash-card-li' }, [liveRunCard(r)])));
  sec.appendChild(list);
  return sec;
}

function liveRunCard(r) {
  const runId = r.run_id;
  const metaParts = [r.workflow].filter(Boolean);
  metaParts.push('building');
  return runCard({
    title: r.course_name || runId,
    status: r.status,
    href: runId ? `#/build/${encodeURIComponent(runId)}` : undefined,
    meta: metaParts.join(' · '),
  });
}

/* ------------------------------------------------------------ Your courses */

function coursesSection(shell, courses, libErr) {
  const sec = section('dash-courses-h', 'Your courses',
    'Open a course to read it or ask questions about it.');
  if (libErr) {
    sec.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(libErr) }));
    return sec;
  }
  if (!courses.length) {
    // Empty Library still offers a next step (never a dead end).
    sec.appendChild(emptyState({
      title: 'No courses yet',
      message: 'Once a build finishes, your course shows up here.',
      cta: { label: 'Create a course', href: '#/create' },
    }));
    return sec;
  }
  const list = el('ul', { class: 'dash-cards cards', 'aria-label': 'Your courses' });
  courses.forEach((c) => {
    const metaParts = [`${c.page_count} page${c.page_count === 1 ? '' : 's'}`];
    list.appendChild(el('li', { class: 'dash-card-li' }, [
      courseCard({
        title: c.title || c.slug,
        href: `#/viewer/${encodeURIComponent(c.slug)}`,
        meta: metaParts.join(' · '),
        askReady: !!c.has_vector_index,
      }),
    ]));
  });
  sec.appendChild(list);
  return sec;
}

/* ------------------------------------------------------------ Recent builds */

function recentSection(shell, recent, runsErr) {
  const sec = section('dash-recent-h', 'Recent builds',
    'Finished builds. Re-open one, or run it again from the same textbook.');
  if (runsErr) {
    sec.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(runsErr) }));
    return sec;
  }
  if (!recent.length) {
    sec.appendChild(emptyState({
      title: 'No finished builds yet',
      message: 'Your completed and failed builds will be listed here.',
      cta: { label: 'Create a course', href: '#/create' },
    }));
    return sec;
  }
  const list = el('ul', { class: 'dash-recent-list cards', 'aria-label': 'Finished builds' });
  recent.forEach((r) => list.appendChild(el('li', { class: 'dash-recent-li' }, [recentEntry(shell, r)])));
  sec.appendChild(list);
  // Link to the full run history (no dead end — Recent builds is a preview).
  sec.appendChild(el('p', { class: 'dash-more' }, [
    el('a', { class: 'dash-more-link', href: '#/runs', text: 'See all builds →' }),
  ]));
  return sec;
}

/** One recent-build entry: runCard + timelineBar + Re-open / Run again. */
function recentEntry(shell, r) {
  const runId = r.run_id;
  const durs = Array.isArray(r.phase_durations) ? r.phase_durations : [];
  const totalMs = durs.reduce((s, p) => s + Math.max(0, Number(p.duration_ms) || 0), 0);
  const metaParts = [r.workflow].filter(Boolean);
  if (totalMs > 0) metaParts.push(fmtDur(totalMs));

  const wrap = el('div', { class: 'dash-recent-entry' });
  wrap.appendChild(runCard({
    title: r.course_name || runId,
    status: r.status,
    href: runId ? `#/build/${encodeURIComponent(runId)}` : undefined,
    meta: metaParts.join(' · '),
  }));

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

  const actions = el('div', { class: 'dash-recent-actions' });
  if (runId) {
    const reopen = el('button', { type: 'button', class: 'btn', text: 'Re-open' });
    reopen.addEventListener('click', () => { location.hash = `#/build/${encodeURIComponent(runId)}`; });
    actions.appendChild(reopen);
  }
  const params = r.params && typeof r.params === 'object' ? r.params : null;
  if (params && (params.corpus || params.upload_id)) {
    const again = el('button', { type: 'button', class: 'btn', text: 'Run again' });
    again.addEventListener('click', () => runAgain(shell, r));
    actions.appendChild(again);
  }
  if (actions.childElementCount) wrap.appendChild(actions);
  return wrap;
}

/** Re-POST /api/runs from a recent run's OWN params (no re-upload). */
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
    location.hash = `#/build/${encodeURIComponent(resp.run_id)}`;
  } catch (e) {
    toastErr(e, 'Could not start');
    shell.setStatus(`Could not start: ${shell.errText(e)}`);
  }
}

/* ------------------------------------------------------------------ helpers */

/** A semantic <section> with an id'd <h2> + intro, labelled by the heading. */
function section(headingId, title, intro) {
  const sec = el('section', { class: 'dash-zone', 'aria-labelledby': headingId });
  sec.appendChild(el('h2', { id: headingId, class: 'dash-zone-h', text: title }));
  if (intro) sec.appendChild(el('p', { class: 'dash-zone-intro muted', text: intro }));
  return sec;
}

/** Normalize the GET /api/runs payload into a flat, status-normalized list. */
function normalizeRuns(data) {
  const arr = Array.isArray(data) ? data : (data && data.runs) || [];
  return arr.map((r) => ({
    run_id: r.run_id || r.runId || r.id || '',
    course_name: r.course_name || '',
    workflow: r.workflow || r.workflow_name || '',
    status: normalizeStatus(r.status || ''),
    phase_durations: Array.isArray(r.phase_durations) ? r.phase_durations : [],
    params: r.params && typeof r.params === 'object' ? r.params : null,
  }));
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

export default renderDashboard;
