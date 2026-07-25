/* Ed4All Studio — Dashboard "System health" section (in-process ed4all doctor).
 *
 * A pure front-end over GET /api/health/doctor (cached) + POST
 * /api/health/doctor/refresh (force re-run). Renders:
 *   - an overall verdict BANNER (healthy | degraded | critical, from the worst
 *     check severity) with a last-updated timestamp + a Refresh button,
 *   - one CARD per check group, each listing its checks as rows: a severity
 *     PILL (glyph + text, never color-only — the shared statusPill kit) + the
 *     check summary, with an expandable <details> carrying remediation + detail.
 *
 * GROUP-AGNOSTIC: the groups + checks are whatever the backend reports — a
 * newly-registered doctor group renders with no change here. Health is
 * on-demand: fetched once when the section mounts (dashboard render) and again
 * only when Refresh is pressed — NO polling loop.
 *
 * PURE PRESENTATION + host-owns-fetch boundary is relaxed for this self-
 * contained section: it owns its two REST calls (mirroring the assistant panel
 * pattern), announces through the shell's single #status live region, and
 * degrades to an inline error on a failed fetch (never blanks the dashboard).
 */

import { api, apiJSON } from '/shared/api.js';
import { el, clear } from '/shared/dom.js';
import { statusPill } from '/shared/components/pill.js';

const BANNER = {
  healthy: { label: 'All systems healthy', cls: 'ok' },
  degraded: { label: 'Degraded — safe but check the warnings', cls: 'warn' },
  critical: { label: 'Critical — resolve before building', cls: 'fail' },
};

/** Build the "System health" <section>; self-fetches + populates on mount. */
export function renderHealthSection(shell) {
  const sec = el('section', { class: 'dash-zone health-zone', 'aria-labelledby': 'dash-health-h' });
  sec.appendChild(el('h2', { id: 'dash-health-h', class: 'dash-zone-h', text: 'System health' }));
  sec.appendChild(el('p', {
    class: 'dash-zone-intro muted',
    text: 'Preflight diagnostics for this server — GPU fit, serving windows, and environment.',
  }));

  const body = el('div', { class: 'health-body', 'aria-live': 'polite', 'aria-busy': 'true' }, [
    el('p', { class: 'loading', text: 'Running health checks…' }),
  ]);
  sec.appendChild(body);

  load(shell, body, { force: false });
  return sec;
}

/** Fetch (or force-refresh) the doctor payload and render it into `body`. */
async function load(shell, body, { force }) {
  body.setAttribute('aria-busy', 'true');
  let payload;
  try {
    payload = force
      ? await apiJSON('/api/health/doctor/refresh', 'POST', {})
      : await api('/api/health/doctor');
  } catch (e) {
    clear(body).appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    body.setAttribute('aria-busy', 'false');
    shell.setStatus('Health checks failed to load.');
    return;
  }
  renderPayload(shell, body, payload);
  body.setAttribute('aria-busy', 'false');
}

function renderPayload(shell, body, payload) {
  clear(body);
  const state = String(payload.exit_state || 'healthy');
  const banner = BANNER[state] || BANNER.healthy;

  // --- verdict banner + refresh + last-updated -----------------------------
  const refreshBtn = el('button', { type: 'button', class: 'btn health-refresh', text: 'Refresh' });
  refreshBtn.addEventListener('click', async () => {
    refreshBtn.disabled = true;
    shell.setStatus('Re-running health checks.');
    await load(shell, body, { force: true });
  });

  const summary = payload.summary || {};
  const counts = [];
  if (summary.fail) counts.push(`${summary.fail} failing`);
  if (summary.warn) counts.push(`${summary.warn} warning${summary.warn === 1 ? '' : 's'}`);
  if (summary.ok) counts.push(`${summary.ok} OK`);
  if (summary.info) counts.push(`${summary.info} info`);

  const bannerEl = el('div', { class: `health-banner health-${banner.cls}`, role: 'status' }, [
    statusPill(`severity:${banner.cls === 'ok' ? 'ok' : banner.cls}`),
    el('span', { class: 'health-banner-label', text: banner.label }),
    counts.length ? el('span', { class: 'health-banner-counts muted', text: `(${counts.join(' · ')})` }) : null,
  ]);
  const head = el('div', { class: 'health-head' }, [bannerEl, refreshBtn]);
  body.appendChild(head);

  const updated = fmtUpdated(payload.generated_at);
  if (updated) {
    body.appendChild(el('p', { class: 'health-updated muted', text: `Last checked ${updated}.` }));
  }

  // --- per-group cards ------------------------------------------------------
  const groups = Array.isArray(payload.groups) ? payload.groups : [];
  if (!groups.length) {
    body.appendChild(el('p', { class: 'muted', text: 'No health checks reported.' }));
  } else {
    const grid = el('div', { class: 'health-cards cards' });
    groups.forEach((g) => grid.appendChild(groupCard(g)));
    body.appendChild(grid);
  }

  shell.setStatus(`Health: ${banner.label}.`);
}

/** One group card: heading + a worst-severity pill + a list of check rows. */
function groupCard(group) {
  const name = String(group.group || 'checks');
  const checks = Array.isArray(group.checks) ? group.checks : [];
  const worst = worstSeverity(checks);

  const card = el('div', { class: 'health-card', 'aria-label': `${name} checks` }, [
    el('div', { class: 'health-card-head' }, [
      el('h3', { class: 'health-card-title', text: name }),
      statusPill(`severity:${worst}`),
    ]),
  ]);

  const list = el('ul', { class: 'health-checks' });
  checks.forEach((c) => list.appendChild(checkRow(c)));
  card.appendChild(list);
  return card;
}

/** One check row: severity pill + summary, with expandable remediation/detail. */
function checkRow(check) {
  const sev = normSeverity(check.severity);
  const li = el('li', { class: 'health-check' });
  const row = el('div', { class: 'health-check-row' }, [
    statusPill(`severity:${sev}`),
    el('span', { class: 'health-check-summary', text: String(check.summary || check.name || '') }),
  ]);
  li.appendChild(row);

  const remediation = String(check.remediation || '').trim();
  const detail = String(check.detail || '').trim();
  if (remediation || detail) {
    const details = el('details', { class: 'health-check-more' }, [
      el('summary', { text: 'Details' }),
    ]);
    if (remediation) {
      details.appendChild(el('p', { class: 'health-remediation' }, [
        el('strong', { text: 'Fix: ' }),
        remediation,
      ]));
    }
    if (detail) {
      details.appendChild(el('p', { class: 'health-detail muted', text: detail }));
    }
    li.appendChild(details);
  }
  return li;
}

/* ------------------------------------------------------------------ helpers */

/** Normalize a backend severity onto the statusPill severity keys. */
function normSeverity(sev) {
  const s = String(sev || '').toLowerCase();
  if (s === 'ok' || s === 'info' || s === 'warn' || s === 'fail') return s;
  if (s === 'warning') return 'warn';
  if (s === 'critical' || s === 'error') return 'fail';
  return 'info';
}

/** Worst severity across a group's checks (fail > warn > info > ok). */
function worstSeverity(checks) {
  const rank = { ok: 0, info: 1, warn: 2, fail: 3 };
  let worst = 'ok';
  checks.forEach((c) => {
    const s = normSeverity(c.severity);
    if ((rank[s] || 0) > (rank[worst] || 0)) worst = s;
  });
  return worst;
}

/** Format an ISO timestamp for the "last checked" line (local, best-effort). */
function fmtUpdated(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
  } catch {
    return String(iso);
  }
}

export default renderHealthSection;
