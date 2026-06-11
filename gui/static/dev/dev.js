/* Ed4All Developer Console (Marketable-v1 C4) — pane-based power-user surface.
 *
 * A vanilla ES module that ports the six classic operator tabs into a /dev/
 * pane shell and adds power-user tooling (Run Inspector, API Explorer, Env
 * Catalog editor, Events feed, console-wide ApiError drawer).
 *
 * Per-pane port strategy:
 *   - upload / settings / routing / courses / retrieval  → EMBEDDED. The legacy
 *     classic-script /app.js is loaded once into THIS document (after DOMContent-
 *     Loaded, so its auto-boot listener never fires and it only registers its
 *     globals); each pane calls the matching window.render<Tab>(container)
 *     directly. This reuses 100% of the working tab logic verbatim without a
 *     rewrite — the disproportionate alternative (porting 1.3k lines of tab
 *     render code to ES modules) is explicitly out of scope this wave.
 *   - runs (Run Inspector)  → NATIVE. A purpose-built inspector (run list →
 *     live WS tail → A6 gate table + validation report → download log).
 *   - env (Env Catalog)     → NATIVE, reuses the settings API (masked secrets,
 *     dotted-path PATCH writes).
 *   - events (Events feed)  → NATIVE, live-polls /api/activity/events.
 *   - api (API Explorer)    → NATIVE raw-request panel + /docs link-out.
 *
 * The shared api() error hook feeds the console-wide ApiError drawer.
 */

import { api, ApiError, onApiError } from '/shared/api.js';
import { $, $$, el, clear, uid } from '/shared/dom.js';
import { toast, toastErr } from '/shared/toast.js';

/* ===================================================================== */
/* Operator token (same sessionStorage key as the legacy SPA + studio).  */
/* The shared api() does not attach the token itself, so the dev console  */
/* re-implements the D2 attach/401-prompt flow (mirrors app.js) and the   */
/* legacy app.js — once embedded — uses the same key, so a token entered  */
/* in either surface unlocks both.                                        */
/* ===================================================================== */
const TOKEN_KEY = 'ed4all_gui_token';
export function getToken() { try { return sessionStorage.getItem(TOKEN_KEY) || ''; } catch (_) { return ''; } }
function setToken(t) { try { if (t) sessionStorage.setItem(TOKEN_KEY, t); else sessionStorage.removeItem(TOKEN_KEY); } catch (_) {} }

/* A token-aware fetch wrapper: attaches the bearer token + on a 401 prompts
 * once and retries. Delegates the body normalization to the shared api(), so
 * the ApiError drawer hook fires for every failure exactly once. */
let _tokenPromptInFlight = null;
function promptForToken() {
  if (_tokenPromptInFlight) return _tokenPromptInFlight;
  _tokenPromptInFlight = new Promise((resolve) => {
    const input = el('input', { type: 'password', placeholder: 'Operator token', 'aria-label': 'Operator token', autocomplete: 'off' });
    const id = uid('tok'); input.id = id;
    const label = el('label', { text: 'Token' }); label.htmlFor = id;
    const submit = el('button', { class: 'primary', text: 'Unlock' });
    const form = el('form', { class: 'token-form' }, [
      el('h2', { text: 'Operator token required' }),
      el('p', { class: 'help', text: 'This developer console is protected. Enter the operator token (ED4ALL_GUI_TOKEN) to continue.' }),
      el('div', { class: 'field' }, [label, input]),
      submit,
    ]);
    const overlay = el('div', { class: 'token-overlay', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Operator token required' }, [form]);
    function close(t) { setToken(t); overlay.remove(); _tokenPromptInFlight = null; resolve(t); }
    form.addEventListener('submit', (e) => { e.preventDefault(); const t = input.value.trim(); if (t) close(t); });
    document.body.appendChild(overlay);
    input.focus();
  });
  return _tokenPromptInFlight;
}

export async function tapi(path, opts = {}) {
  const token = getToken();
  let o = opts;
  if (token) o = Object.assign({}, opts, { headers: Object.assign({}, opts.headers, { Authorization: `Bearer ${token}` }) });
  try {
    return await api(path, o);
  } catch (e) {
    if (e instanceof ApiError && e.status === 401 && !opts._tokenRetry) {
      setToken('');
      const t = await promptForToken();
      if (t) return tapi(path, Object.assign({}, opts, { _tokenRetry: true }));
    }
    throw e;
  }
}
const tapiJSON = (path, method, body) =>
  tapi(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

/* ===================================================================== */
/* Console-wide ApiError drawer                                          */
/* ===================================================================== */
const MAX_DRAWER_ROWS = 50;
let _drawerRows = [];

function initErrorDrawer() {
  const drawer = $('#dev-error-drawer');
  const toggle = $('#dev-drawer-toggle');
  const body = $('#dev-drawer-body');
  const clearBtn = $('#dev-drawer-clear');
  const count = $('#dev-drawer-count');
  const rows = $('#dev-err-rows');
  if (!drawer || !toggle) return;

  toggle.addEventListener('click', () => {
    const open = drawer.classList.toggle('collapsed') === false;
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) { body.hidden = false; } else { body.hidden = true; }
  });
  clearBtn.addEventListener('click', () => {
    _drawerRows = [];
    clear(rows);
    count.textContent = '0';
    clearBtn.hidden = true;
  });

  // Register the opt-in error hook on the shared api(). Studio/learner keep
  // their toast-only behavior (they don't register a hook); this is additive.
  onApiError((info) => {
    const ts = new Date();
    _drawerRows.push(Object.assign({ ts }, info));
    if (_drawerRows.length > MAX_DRAWER_ROWS) _drawerRows = _drawerRows.slice(-MAX_DRAWER_ROWS);
    const tr = el('tr', {}, [
      el('td', { text: ts.toLocaleTimeString() }),
      el('td', { class: 'dev-err-status', text: String(info.status) }),
      el('td', {}, el('code', { text: `${info.method} ${info.path}` })),
      el('td', { class: 'dev-err-detail', text: (info.detail || info.error || '').slice(0, 200) }),
    ]);
    rows.appendChild(tr);
    count.textContent = String(_drawerRows.length);
    clearBtn.hidden = false;
  });
}

/* ===================================================================== */
/* Embedded legacy /app.js loader                                        */
/* ===================================================================== */
let _legacyLoad = null;
function loadLegacyApp() {
  if (_legacyLoad) return _legacyLoad;
  _legacyLoad = new Promise((resolve, reject) => {
    // app.js is a classic script. Injected AFTER our DOMContentLoaded, its
    // own `addEventListener('DOMContentLoaded', boot)` never fires, so it
    // registers its globals (render functions, api, el, toast) WITHOUT
    // auto-booting into the (absent) #view/#navlist of the classic SPA.
    if (window.renderUpload) { resolve(); return; }
    const s = document.createElement('script');
    s.src = '/app.js';
    s.onload = () => {
      // The legacy renderers re-render via the legacy global `route()` (e.g.
      // renderSettings after a save). That route() targets the classic SPA's
      // #view / #navlist, absent here. Override the global so a legacy
      // re-render re-dispatches THIS console's pane router against #dev-main.
      // Classic-script top-level `function route()` is a window property, so
      // the in-closure `route()` calls resolve to this override via the scope
      // chain → no #view null-deref.
      try { window.route = route; } catch (_) { /* ignore */ }
      resolve();
    };
    s.onerror = () => reject(new Error('failed to load /app.js'));
    document.head.appendChild(s);
  });
  return _legacyLoad;
}

/* A pane backed by a legacy render function. The legacy render takes a `view`
 * container element + returns an optional teardown. We give it our pane root. */
function legacyPane(fnName, heading) {
  return async (root) => {
    await loadLegacyApp();
    const fn = window[fnName];
    if (typeof fn !== 'function') {
      root.appendChild(el('p', { class: 'error', text: `Legacy renderer ${fnName} unavailable.` }));
      return;
    }
    // The legacy renderers clear + populate the passed container directly.
    return await fn(root);
  };
}

/* ===================================================================== */
/* Run Inspector pane (NATIVE)                                            */
/* ===================================================================== */
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'canceled', 'interrupted', 'error']);

function statusChip(status) {
  const st = status || '';
  const cls = st === 'completed' ? 'completed'
    : (st === 'failed' || st === 'cancelled' || st === 'canceled' || st === 'error' || st === 'interrupted') ? 'failed'
      : (st === 'running' || st === 'requested' || st === 'queued') ? 'running' : '';
  return el('span', { class: 'run-status' }, [el('span', { class: `dot ${cls}` }), el('span', { text: st || '—' })]);
}

async function renderRunInspector(root) {
  clear(root);
  root.appendChild(el('h1', { text: 'Run Inspector' }));
  root.appendChild(el('p', { class: 'subtitle', text: 'Inspect runs: live log tail, validation gates, failure cause, and log download. Re-attaches to a live run on refresh.' }));

  const layout = el('div', { class: 'dev-inspector' });
  const listCol = el('div', { class: 'dev-run-list card' });
  const detailCol = el('div', { class: 'dev-run-detail card' });
  layout.append(listCol, detailCol);
  root.appendChild(layout);

  // ---- run list (newest first) ----
  const refreshBtn = el('button', { class: 'ghost sm', text: 'Refresh' });
  const listBody = el('div', { class: 'recent-runs' }, el('div', { class: 'loading', text: 'Loading runs…' }));
  listCol.append(
    el('div', { class: 'flex wrap' }, [el('h2', { text: 'Runs', style: 'flex:1;margin:0;border:none' }), refreshBtn]),
    listBody,
  );

  // ---- detail pane state ----
  let ws = null;
  let currentRunId = null;
  const detailHead = el('div', { class: 'dev-detail-head' });
  const consoleEl = el('pre', { class: 'console', tabindex: '0', 'aria-label': 'Run log', role: 'log', 'aria-live': 'polite' });
  const gateWrap = el('div', { class: 'dev-gate-wrap' });
  const reportWrap = el('div', { class: 'dev-report-wrap' });
  detailCol.append(
    el('h2', { text: 'Detail' }),
    detailHead,
    el('div', { class: 'flex wrap', style: 'gap:8px;margin:8px 0' }, [
      el('button', { class: 'ghost sm', text: 'Download log', onclick: downloadLog }),
    ]),
    consoleEl, gateWrap, reportWrap,
  );
  detailHead.appendChild(el('p', { class: 'muted', text: 'Select a run to inspect.' }));

  let lineCount = 0;
  function logLine(text, kind) {
    const line = el('div', { class: kind ? `ln ${kind}` : 'ln', text: String(text) });
    consoleEl.appendChild(line);
    lineCount++;
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
  function downloadLog() {
    const blob = new Blob([consoleEl.textContent || ''], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: `${currentRunId || 'run'}.log` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function renderGates(gates) {
    clear(gateWrap);
    const list = Array.isArray(gates) ? gates : [];
    if (!list.length) return;
    gateWrap.appendChild(el('h3', { text: 'Gate results' }));
    const rows = list.map((g) => {
      const passed = g.passed === true || g.status === 'pass';
      const cls = passed ? 'gate-pass' : (g.severity === 'warning' ? 'gate-warn' : 'gate-fail');
      return el('tr', {}, [
        el('td', { text: g.gate_id || g.gate || g.name || '?' }),
        el('td', { class: cls, text: passed ? 'pass' : (g.status || 'fail') }),
        el('td', { text: g.severity || '' }),
        el('td', { text: g.detail || g.message || (g.issues ? `${g.issues.length} issues` : '') }),
      ]);
    });
    gateWrap.appendChild(el('table', {}, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'Gate' }), el('th', { text: 'Result' }), el('th', { text: 'Severity' }), el('th', { text: 'Detail' })])),
      el('tbody', {}, rows),
    ]));
  }

  async function loadValidationReport(runId) {
    clear(reportWrap);
    let rep;
    try {
      rep = await tapi(`/api/runs/${encodeURIComponent(runId)}/validation-report`);
    } catch (e) {
      reportWrap.appendChild(el('p', { class: 'muted', text: `validation report unavailable: ${e instanceof ApiError ? e.error : e}` }));
      return;
    }
    if (!rep) return;
    if (rep.failed_phase || rep.failure_reason) {
      reportWrap.appendChild(el('div', { class: 'dev-failbox' }, [
        el('h3', { text: 'Failure' }),
        rep.failed_phase ? el('p', {}, [el('strong', { text: 'failed_phase: ' }), el('code', { text: String(rep.failed_phase) })]) : null,
        rep.failure_reason ? el('p', {}, [el('strong', { text: 'failure_reason: ' }), el('span', { text: String(rep.failure_reason) })]) : null,
      ]));
    }
    const fg = Array.isArray(rep.failed_gates) ? rep.failed_gates : [];
    if (fg.length) {
      reportWrap.appendChild(el('h3', { text: 'Failed gates' }));
      const rows = fg.map((g) => el('tr', {}, [
        el('td', { text: g.phase || '' }),
        el('td', { text: g.gate_id || '' }),
        el('td', { class: g.severity === 'warning' ? 'gate-warn' : 'gate-fail', text: g.severity || '' }),
        el('td', { text: g.message || '' }),
        el('td', { text: g.issues_count != null ? String(g.issues_count) : '' }),
      ]));
      reportWrap.appendChild(el('table', {}, [
        el('thead', {}, el('tr', {}, [el('th', { text: 'Phase' }), el('th', { text: 'Gate' }), el('th', { text: 'Severity' }), el('th', { text: 'Message' }), el('th', { text: 'Issues' })])),
        el('tbody', {}, rows),
      ]));
    }
    if (rep.report) {
      const details = el('details', { class: 'dev-report-json' }, [
        el('summary', { text: 'Validation report (courseforge_validation_report.json)' }),
        el('pre', { class: 'json', text: JSON.stringify(rep.report, null, 2) }),
      ]);
      reportWrap.appendChild(details);
    } else if (rep.note) {
      reportWrap.appendChild(el('p', { class: 'muted', text: rep.note }));
    }
  }

  function openWs(runId) {
    if (ws) { try { ws.close(); } catch (_) {} ws = null; }
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = getToken();
    const q = token ? `?token=${encodeURIComponent(token)}` : '';
    const url = `${proto}//${location.host}/api/ws/runs/${encodeURIComponent(runId)}${q}`;
    logLine(`[gui] connecting ${url.replace(/\?token=[^&]*/, '?token=***')}`, 'sys');
    try { ws = new WebSocket(url); } catch (e) { logLine(`[gui] websocket failed: ${e}`, 'err'); return; }
    ws.onopen = () => logLine('[gui] log stream open', 'sys');
    ws.onmessage = (ev) => {
      let msg = ev.data;
      if (typeof msg === 'string' && msg.startsWith('{')) {
        try {
          const obj = JSON.parse(msg);
          if (obj.type === 'status' || obj.status) setStatusChip(obj.status || obj.type);
          if (obj.line != null) logLine(obj.line);
          if (obj.gates || obj.gate_results) renderGates(obj.gates || obj.gate_results);
          if (obj.error) logLine(`[error] ${obj.error}: ${obj.detail || ''}`, 'err');
          if (obj.type === 'status' && TERMINAL.has(obj.status)) loadValidationReport(runId);
          return;
        } catch (_) { /* raw line */ }
      }
      logLine(String(msg));
    };
    ws.onerror = () => logLine('[gui] websocket error', 'err');
    ws.onclose = () => logLine('[gui] log stream closed', 'sys');
  }

  let chipHolder = null;
  function setStatusChip(status) {
    if (!chipHolder) return;
    clear(chipHolder).appendChild(statusChip(status));
  }

  function selectRun(rid, status) {
    currentRunId = rid;
    clear(consoleEl); clear(gateWrap); clear(reportWrap);
    lineCount = 0;
    clear(detailHead);
    chipHolder = el('span', { class: 'dev-chip-holder' });
    detailHead.append(
      el('code', { class: 'kv', text: String(rid) }),
      chipHolder,
    );
    setStatusChip(status);
    // Mark the active row.
    $$('.recent-runs tr.clickable').forEach((tr) => tr.classList.toggle('active', tr.dataset.rid === String(rid)));
    // Always re-attach the WS — the server replays the log from offset 0, so a
    // terminal run shows its full log and a live run keeps streaming. This is
    // the re-attach-on-refresh behavior.
    openWs(rid);
    if (status && TERMINAL.has(status)) loadValidationReport(rid);
  }

  async function refreshRuns(autoSelectFirst) {
    try {
      const data = await tapi('/api/runs');
      const runs = Array.isArray(data) ? data : (data && data.runs) || [];
      clear(listBody);
      if (!runs.length) { listBody.appendChild(el('div', { class: 'empty', text: 'No runs yet.' })); return; }
      const tbody = el('tbody');
      runs.forEach((r) => {
        const rid = r.run_id || r.runId || r.id;
        const st = r.status || '';
        const tr = el('tr', { class: 'clickable', dataset: { rid: String(rid) } }, [
          el('td', {}, el('code', { class: 'kv', text: String(rid) })),
          el('td', { text: r.workflow || r.workflow_name || '' }),
          el('td', {}, statusChip(st)),
        ]);
        tr.addEventListener('click', () => selectRun(rid, st));
        tbody.appendChild(tr);
      });
      listBody.appendChild(el('table', {}, [
        el('thead', {}, el('tr', {}, [el('th', { text: 'Run' }), el('th', { text: 'Workflow' }), el('th', { text: 'Status' })])),
        tbody,
      ]));
      // Re-attach to the previously-selected run on refresh, else the newest.
      if (currentRunId && runs.some((r) => String(r.run_id || r.runId || r.id) === String(currentRunId))) {
        const r = runs.find((x) => String(x.run_id || x.runId || x.id) === String(currentRunId));
        selectRun(currentRunId, r && r.status);
      } else if (autoSelectFirst) {
        const r0 = runs[0];
        selectRun(r0.run_id || r0.runId || r0.id, r0.status);
      }
    } catch (e) {
      clear(listBody).appendChild(el('div', { class: 'empty', text: `Failed to load runs: ${e instanceof ApiError ? e.error : e}` }));
    }
  }
  refreshBtn.addEventListener('click', () => refreshRuns(false));
  refreshRuns(true);

  return () => { if (ws) { try { ws.close(); } catch (_) {} } };
}

/* ===================================================================== */
/* Env Catalog editor pane (NATIVE — reuses the settings API)            */
/* ===================================================================== */
async function renderEnvCatalog(root) {
  clear(root);
  root.appendChild(el('h1', { text: 'Env Catalog' }));
  root.appendChild(el('p', { class: 'subtitle', text: 'The full environment catalog (gui/env_catalog.py) via the settings API. Secrets are write-only and stay masked (send only on save). Saves PATCH the env/flags section of the settings doc.' }));

  const data = await tapi('/api/settings');
  const catalogRaw = (data && data.catalog) || [];
  const catalog = Array.isArray(catalogRaw) ? catalogRaw : Object.values(catalogRaw).flat();
  const env = (data && data.env) || {};
  const flags = (data && data.flags) || {};

  const filterInput = el('input', { type: 'search', placeholder: 'Filter by key / category…', 'aria-label': 'Filter env catalog' });
  const fid = uid('filter'); filterInput.id = fid;
  const flabel = el('label', { text: 'Filter' }); flabel.htmlFor = fid;
  root.appendChild(el('div', { class: 'card' }, [el('div', { class: 'field' }, [flabel, filterInput])]));

  const tableCard = el('div', { class: 'card' });
  root.appendChild(tableCard);

  // PATCH the env/flags section for a single key (matches the legacy save shape:
  // {env:{KEY:val}} for non-bool, {flags:{KEY:val}} for bool). Re-render to
  // reflect the server's re-masked state after a secret write.
  async function saveOne(entry, value) {
    const key = entry.key;
    const patch = entry.type === 'bool' ? { flags: { [key]: value } } : { env: { [key]: value } };
    await tapiJSON('/api/settings', 'PATCH', patch);
  }

  function render(filter) {
    clear(tableCard);
    const q = (filter || '').trim().toLowerCase();
    const tbody = el('tbody');
    let shown = 0;
    catalog.forEach((entry) => {
      const key = entry.key;
      const cat = entry.category || '';
      const secret = entry.type === 'secret';
      if (q && !(`${key} ${cat} ${entry.label || ''}`.toLowerCase().includes(q))) return;
      shown++;
      const curVal = env[key] != null ? env[key] : (flags[key] != null ? flags[key] : entry.default);

      let ctl;
      let getValue;
      if (entry.type === 'bool') {
        const cb = el('input', { type: 'checkbox', 'aria-label': `Value for ${key}` });
        cb.checked = curVal === true || curVal === 'true' || curVal === 1;
        ctl = cb; getValue = () => cb.checked;
      } else if (entry.type === 'enum' && Array.isArray(entry.enum)) {
        const sel = el('select', { 'aria-label': `Value for ${key}` });
        sel.appendChild(el('option', { value: '', text: '— unset —' }));
        entry.enum.forEach((o) => {
          const opt = el('option', { value: o, text: o });
          if (String(curVal) === String(o)) opt.selected = true;
          sel.appendChild(opt);
        });
        ctl = sel; getValue = () => sel.value;
      } else if (secret) {
        const setState = curVal === 'set' || curVal === true;
        const inp = el('input', {
          type: 'password',
          placeholder: setState ? '•••••• (set — leave blank to keep)' : 'paste secret to set',
          'aria-label': `Value for ${key}`, autocomplete: 'off',
        });
        ctl = inp; getValue = () => inp.value;
      } else {
        const inp = el('input', {
          type: entry.type === 'number' ? 'number' : 'text',
          value: (curVal != null && curVal !== 'set') ? curVal : '',
          'aria-label': `Value for ${key}`, autocomplete: 'off',
        });
        ctl = inp; getValue = () => (entry.type === 'number' ? (inp.value === '' ? null : Number(inp.value)) : inp.value);
      }

      const saveBtn = el('button', { class: 'primary sm', text: 'Save' });
      saveBtn.addEventListener('click', async () => {
        const v = getValue();
        if (secret && v === '') { toast('Skipped', `${key} left unchanged (blank secret).`, 'info', 2500); return; }
        saveBtn.disabled = true;
        try {
          await saveOne(entry, v);
          toast('Saved', `${key} updated.`, 'success', 2500);
          if (secret && ctl.tagName === 'INPUT') ctl.value = '';
        } catch (e2) {
          toastErr(e2, `Save ${key} failed`);
        } finally {
          saveBtn.disabled = false;
        }
      });

      tbody.appendChild(el('tr', {}, [
        el('td', {}, el('code', { text: key })),
        el('td', { text: cat }),
        el('td', { text: secret ? 'secret' : (entry.type || '') }),
        el('td', {}, ctl),
        el('td', {}, saveBtn),
      ]));
    });
    if (!shown) { tableCard.appendChild(el('p', { class: 'muted', text: 'No catalog entries match.' })); return; }
    tableCard.appendChild(el('table', {}, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'Key' }), el('th', { text: 'Category' }), el('th', { text: 'Type' }), el('th', { text: 'Value' }), el('th', { text: '' })])),
      tbody,
    ]));
  }
  filterInput.addEventListener('input', () => render(filterInput.value));
  render('');
}

/* ===================================================================== */
/* Events feed pane (NATIVE — live-polls the activity bridge)            */
/* ===================================================================== */
async function renderEvents(root) {
  clear(root);
  root.appendChild(el('h1', { text: 'Events' }));
  root.appendChild(el('p', { class: 'subtitle', text: 'The Claude ↔ GUI activity bridge (/api/activity/events), live-polled.' }));

  const live = el('p', { class: 'visually-hidden', role: 'status', 'aria-live': 'polite' });
  const feed = el('div', { class: 'card dev-events', role: 'log', 'aria-label': 'Activity events', 'aria-live': 'polite' });
  const pauseBtn = el('button', { class: 'ghost sm', text: 'Pause' });
  root.append(el('div', { class: 'flex wrap', style: 'gap:8px;margin:8px 0' }, [pauseBtn]), live, feed);

  let since = 0;
  let paused = false;
  let timer = null;
  let count = 0;

  function row(ev) {
    const seq = ev.seq != null ? ev.seq : '';
    const src = ev.source || ev.role || '';
    const ts = ev.ts || ev.timestamp || '';
    const kind = ev.event_type || ev.type || ev.kind || '';
    const text = ev.message || ev.text || (ev.data != null ? JSON.stringify(ev.data) : '');
    return el('div', { class: `dev-event src-${String(src).toLowerCase()}` }, [
      el('span', { class: 'dev-event-meta', text: `#${seq} ${src} ${kind} ${ts}`.trim() }),
      el('div', { class: 'dev-event-text', text: String(text) }),
    ]);
  }

  async function poll() {
    if (paused) return;
    try {
      const data = await tapi(`/api/activity/events?since=${since}`);
      const events = (data && data.events) || [];
      if (events.length) {
        events.forEach((ev) => {
          feed.appendChild(row(ev));
          if (ev.seq != null) since = Math.max(since, ev.seq + 1);
          count++;
        });
        feed.scrollTop = feed.scrollHeight;
        live.textContent = `${count} event${count === 1 ? '' : 's'} received.`;
      }
    } catch (_) { /* transient; next tick retries */ }
  }
  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  });
  await poll();
  timer = setInterval(poll, 3000);
  return () => { if (timer) clearInterval(timer); };
}

/* ===================================================================== */
/* API Explorer pane (NATIVE raw-request panel + /docs link-out)        */
/* ===================================================================== */
const HIST_KEY = 'ed4all_dev_api_history';
function loadHistory() { try { return JSON.parse(sessionStorage.getItem(HIST_KEY) || '[]'); } catch (_) { return []; } }
function saveHistory(h) { try { sessionStorage.setItem(HIST_KEY, JSON.stringify(h.slice(-20))); } catch (_) {} }

async function renderApiExplorer(root) {
  clear(root);
  root.appendChild(el('h1', { text: 'API Explorer' }));
  // /docs is token-gated and an iframe can't carry the bearer header on a
  // navigation, so we link out (new tab, same sessionStorage token won't help
  // the iframe but the operator can paste it) + build a native raw-request panel.
  root.appendChild(el('p', { class: 'subtitle' }, [
    'Send raw requests via the shared api() wrapper (token attached automatically). The interactive OpenAPI docs open in a ',
    el('a', { href: '/docs', target: '_blank', rel: 'noopener', text: 'new tab (Swagger UI)', 'aria-label': 'Open OpenAPI docs, opens in new tab' }),
    '.',
  ]));

  const method = el('select', { 'aria-label': 'HTTP method' });
  ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].forEach((m) => method.appendChild(el('option', { value: m, text: m })));
  const mid = uid('m'); method.id = mid;
  const path = el('input', { type: 'text', value: '/api/health', placeholder: '/api/...', 'aria-label': 'Request path' });
  const pid = uid('p'); path.id = pid;
  const headers = el('textarea', { rows: '3', placeholder: 'Optional headers, one per line: X-Foo: bar', 'aria-label': 'Request headers' });
  const hid = uid('h'); headers.id = hid;
  const bodyTa = el('textarea', { rows: '5', placeholder: 'Optional JSON body', 'aria-label': 'Request body' });
  const bid = uid('b'); bodyTa.id = bid;
  const sendBtn = el('button', { class: 'primary', text: 'Send' });

  function fld(labelText, ctl) {
    const lab = el('label', { text: labelText }); lab.htmlFor = ctl.id;
    return el('div', { class: 'field' }, [lab, ctl]);
  }

  const form = el('form', { class: 'card dev-api-form' }, [
    el('div', { class: 'flex wrap', style: 'gap:8px' }, [fld('Method', method), fld('Path', path)]),
    fld('Headers', headers),
    fld('Body', bodyTa),
    sendBtn,
  ]);
  const respCard = el('div', { class: 'card dev-api-resp' }, el('p', { class: 'muted', text: 'No request sent yet.' }));
  const histCard = el('div', { class: 'card dev-api-hist' });
  root.append(form, respCard, histCard);

  function parseHeaders(text) {
    const out = {};
    (text || '').split('\n').forEach((line) => {
      const i = line.indexOf(':');
      if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    });
    return out;
  }

  function renderHistory() {
    clear(histCard);
    const h = loadHistory();
    histCard.appendChild(el('h2', { text: 'Request history' }));
    if (!h.length) { histCard.appendChild(el('p', { class: 'muted', text: 'No requests yet (last 20 are remembered this session).' })); return; }
    const ul = el('ul', { class: 'dev-hist-list', 'aria-label': 'Recent requests' });
    h.slice().reverse().forEach((entry) => {
      const btn = el('button', { class: 'ghost sm', text: `${entry.method} ${entry.path}`, 'aria-label': `Replay ${entry.method} ${entry.path}` });
      btn.addEventListener('click', () => {
        method.value = entry.method; path.value = entry.path;
        headers.value = entry.headers || ''; bodyTa.value = entry.body || '';
        send();
      });
      ul.appendChild(el('li', {}, [btn, el('span', { class: 'muted', text: ` → ${entry.status}` })]));
    });
    histCard.appendChild(ul);
  }

  async function send() {
    const m = method.value;
    const p = path.value.trim();
    if (!p) { toast('Path required', 'Enter a request path.', 'error'); return; }
    const hdrs = parseHeaders(headers.value);
    const opts = { method: m, headers: hdrs };
    if (bodyTa.value.trim() && m !== 'GET') {
      opts.body = bodyTa.value;
      if (!Object.keys(hdrs).some((k) => k.toLowerCase() === 'content-type')) hdrs['Content-Type'] = 'application/json';
    }
    sendBtn.disabled = true;
    clear(respCard).appendChild(el('div', { class: 'loading', text: 'Sending…' }));
    let status = 'err';
    try {
      // Use the raw fetch (not api()) so we can show the real status + headers
      // even on a 4xx/5xx, but still attach the operator token like tapi().
      const token = getToken();
      if (token) hdrs['Authorization'] = `Bearer ${token}`;
      const res = await fetch(p, opts);
      status = String(res.status);
      const ctype = res.headers.get('content-type') || '';
      let bodyText;
      if (ctype.includes('application/json')) {
        try { bodyText = JSON.stringify(await res.json(), null, 2); } catch (_) { bodyText = await res.text(); }
      } else { bodyText = await res.text(); }
      const hdrLines = [];
      res.headers.forEach((v, k) => hdrLines.push(`${k}: ${v}`));
      clear(respCard).append(
        el('h2', { text: `Response ${res.status} ${res.statusText}` }),
        el('details', { open: false }, [el('summary', { text: 'Headers' }), el('pre', { class: 'json', text: hdrLines.join('\n') })]),
        el('pre', { class: 'json', text: bodyText || '(empty body)' }),
      );
    } catch (e) {
      status = 'network';
      clear(respCard).appendChild(el('p', { class: 'error', text: `Request failed: ${e && e.message || e}` }));
    } finally {
      sendBtn.disabled = false;
    }
    const h = loadHistory();
    h.push({ method: m, path: p, headers: headers.value, body: bodyTa.value, status });
    saveHistory(h);
    renderHistory();
  }
  form.addEventListener('submit', (e) => { e.preventDefault(); send(); });
  renderHistory();
}

/* ===================================================================== */
/* Pane registry + router                                                */
/* ===================================================================== */
const PANES = {
  runs: renderRunInspector,
  upload: legacyPane('renderUpload', 'Upload & Run'),
  settings: legacyPane('renderSettings', 'Settings / API Keys'),
  routing: legacyPane('renderRouting', 'Model Routing'),
  courses: legacyPane('renderCourses', 'Courses & Topics'),
  retrieval: legacyPane('renderRetrieval', 'Retrieval'),
  env: renderEnvCatalog,
  events: renderEvents,
  api: renderApiExplorer,
};
const DEFAULT_PANE = 'runs';
let activeTeardown = null;

function currentPane() {
  // Hash form: #/dev/<pane>. Tolerate a bare #/<pane> too.
  const raw = (location.hash || '').replace(/^#\/?/, '');
  const segs = raw.split('/').filter(Boolean);
  const name = segs[0] === 'dev' ? segs[1] : segs[0];
  return PANES[name] ? name : DEFAULT_PANE;
}

async function route() {
  if (typeof activeTeardown === 'function') { try { activeTeardown(); } catch (_) {} activeTeardown = null; }
  const pane = currentPane();
  $$('#dev-navlist a').forEach((a) => {
    const on = a.dataset.pane === pane;
    a.classList.toggle('active', on);
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  const main = $('#dev-main');
  main.setAttribute('aria-busy', 'true');
  main.setAttribute('aria-label', `${pane} pane`);
  clear(main).appendChild(el('div', { class: 'loading', text: 'Loading…' }));
  try {
    const teardown = await PANES[pane](main);
    if (typeof teardown === 'function') activeTeardown = teardown;
  } catch (e) {
    clear(main).appendChild(el('div', { class: 'card' }, [
      el('h1', { text: 'Failed to load pane' }),
      el('p', { class: 'subtitle', text: (e instanceof ApiError ? `${e.error}: ${e.detail}` : String(e)) }),
      el('button', { class: 'ghost', text: 'Retry', onclick: () => route() }),
    ]));
    toastErr(e, 'Pane load failed');
  } finally {
    main.setAttribute('aria-busy', 'false');
  }
}

async function pingConnection() {
  const dot = $('#dev-conn-dot'), label = $('#dev-conn-label');
  if (!dot) return;
  try {
    await tapi('/api/settings');
    dot.className = 'conn-dot ok'; label.textContent = 'connected';
  } catch (e) {
    dot.className = 'conn-dot bad';
    label.textContent = (e instanceof ApiError && e.status === 0) ? 'offline' : 'api error';
  }
}

function boot() {
  initErrorDrawer();
  window.addEventListener('hashchange', route);
  if (!location.hash) location.hash = `#/dev/${DEFAULT_PANE}`;
  route();
  pingConnection();
  setInterval(pingConnection, 15000);
}
boot();

export { route, currentPane, PANES };
