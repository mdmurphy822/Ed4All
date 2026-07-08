/* Ed4All Control Plane — vanilla SPA. No frameworks, no build step.
 *
 * Contract: plans/gui-control-plane/spec.md §6 (REST) + §7 (frontend).
 * Every fetch goes through api() which normalizes the typed error
 * shape {error, detail} and non-2xx status into a thrown ApiError.
 */
'use strict';

/* =====================================================================
 * Small DOM helpers
 * ===================================================================== */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, '');
    else if (v === false) { /* skip */ }
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

let _uid = 0;
function uid(prefix = 'f') { return `${prefix}-${Date.now().toString(36)}-${(_uid++).toString(36)}`; }

/* A11y form-field helper: pairs a <label> with its input via for/id and wraps
 * them in the usual `.field` div. Routes the main form-building sites through
 * this so screen readers announce the label when focus lands on the input. */
function field(labelText, inputEl, help) {
  const id = uid('in');
  inputEl.id = id;
  const label = el('label', { text: labelText });
  label.htmlFor = id;
  return el('div', { class: 'field' }, [
    label,
    inputEl,
    help ? el('div', { class: 'help', text: help }) : null,
  ]);
}

/* Clipboard helper: a tiny "Copy" button bound to `text`. Uses the async
 * Clipboard API with a hidden-textarea + execCommand fallback for non-secure
 * contexts. Fires a success toast on copy. */
async function copyText(text) {
  const s = String(text == null ? '' : text);
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(s);
      return true;
    }
  } catch (_) { /* fall through to legacy path */ }
  try {
    const ta = el('textarea', { style: 'position:fixed;opacity:0;left:-9999px' });
    ta.value = s;
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch (_) { return false; }
}
function copyBtn(text, label = 'Copy') {
  const btn = el('button', { class: 'ghost sm', text: label, 'aria-label': `${label} to clipboard` });
  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const value = typeof text === 'function' ? text() : text;
    const ok = await copyText(value);
    if (ok) toast('Copied', String(value).slice(0, 80), 'success', 2500);
    else toast('Copy failed', 'Clipboard unavailable in this context.', 'error');
  });
  return btn;
}

function fmtBytes(n) {
  if (n == null) return '';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0, v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${u[i]}`;
}

/* =====================================================================
 * API client — typed error handling for {error, detail}
 * ===================================================================== */
class ApiError extends Error {
  constructor(status, error, detail) {
    super(detail || error || `HTTP ${status}`);
    this.status = status;
    this.error = error || 'error';
    this.detail = detail || '';
  }
}

/* =====================================================================
 * Operator token (Marketable-v1 D2)
 *
 * When the server gates the operator surface with ED4ALL_GUI_TOKEN, every
 * operator fetch must carry `Authorization: Bearer <token>`. The token is held
 * in sessionStorage (cleared on tab close), attached by api() below, and a 401
 * triggers a minimal token-entry overlay. When no token is configured the
 * server never 401s, so this stays dormant (no overlay, no header rejected).
 * ===================================================================== */
const TOKEN_KEY = 'ed4all_gui_token';
function getToken() { try { return sessionStorage.getItem(TOKEN_KEY) || ''; } catch (_) { return ''; } }
function setToken(t) { try { if (t) sessionStorage.setItem(TOKEN_KEY, t); else sessionStorage.removeItem(TOKEN_KEY); } catch (_) {} }

/* Minimal token-entry overlay. Shown on a 401; resolves once the user submits a
 * token (stored in sessionStorage). Singleton — concurrent 401s share one
 * prompt so a burst of failed fetches doesn't stack overlays. */
let _tokenPromptInFlight = null;
function promptForToken() {
  if (_tokenPromptInFlight) return _tokenPromptInFlight;
  _tokenPromptInFlight = new Promise((resolve) => {
    const input = el('input', { type: 'password', placeholder: 'Operator token', 'aria-label': 'Operator token', autocomplete: 'off' });
    const submit = el('button', { class: 'primary', text: 'Unlock' });
    const form = el('form', { class: 'token-form' }, [
      el('h2', { text: 'Operator token required' }),
      el('p', { class: 'help', text: 'This control plane is protected. Enter the operator token (ED4ALL_GUI_TOKEN) to continue.' }),
      field('Token', input),
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

async function api(path, opts = {}) {
  // Attach the operator bearer token (if held) to every request. Harmless when
  // the server runs open — an unexpected header is ignored.
  const token = getToken();
  if (token) {
    opts = Object.assign({}, opts, { headers: Object.assign({}, opts.headers, { Authorization: `Bearer ${token}` }) });
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (networkErr) {
    throw new ApiError(0, 'network_error', String(networkErr && networkErr.message || networkErr));
  }
  // 401 → operator token missing/wrong. Prompt for one and retry once with it.
  if (res.status === 401 && !opts._tokenRetry) {
    setToken('');
    const t = await promptForToken();
    if (t) return api(path, Object.assign({}, opts, { _tokenRetry: true }));
  }
  // 204 / empty body
  if (res.status === 204) return null;
  const ctype = res.headers.get('content-type') || '';
  let body = null;
  if (ctype.includes('application/json')) {
    try { body = await res.json(); } catch (_) { body = null; }
  } else {
    const txt = await res.text();
    body = txt ? { error: 'non_json_response', detail: txt.slice(0, 500) } : null;
  }
  if (!res.ok) {
    // FastAPI request-validation errors return body.detail as an ARRAY of
    // {type, loc, msg, ...}. typeof [] === 'object', so the nested-shape branch
    // below would silently drop these; detect the array first and flatten each
    // entry to "field.path: message" (dropping the leading "body" segment of
    // loc) so the toast surfaces the useful field-level messages.
    if (body && Array.isArray(body.detail)) {
      const detail = body.detail
        .map((e) => {
          const loc = Array.isArray(e?.loc) ? e.loc.slice(1).join('.') : '';
          const msg = e?.msg ?? e?.type ?? 'invalid';
          return loc ? `${loc}: ${msg}` : String(msg);
        })
        .filter(Boolean)
        .join('; ') || res.statusText;
      throw new ApiError(res.status, 'validation_error', detail);
    }
    // Tolerate both error-body shapes: the nested {detail:{error,detail}}
    // (legacy routers) and the flattened {error,detail} (migrated routers).
    const nested = body && typeof body.detail === 'object' && body.detail !== null ? body.detail : null;
    const error = (nested?.error ?? body?.error ?? `http_${res.status}`);
    const detail = (nested?.detail ?? (typeof body?.detail === 'string' ? body.detail : null) ?? body?.error ?? res.statusText);
    throw new ApiError(res.status, error, detail);
  }
  return body;
}

const apiJSON = (path, method, body) =>
  api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

/* Live Ollama model discovery (GET /api/settings/ollama-models). Returns the
 * service shape {available, models, detail, host}; never throws — a network /
 * offline failure degrades to {available:false, models:[]} so callers fall
 * back to free-text entry. */
async function fetchOllamaModels() {
  try {
    const r = await api('/api/settings/ollama-models');
    return {
      available: !!(r && r.available),
      models: (r && Array.isArray(r.models)) ? r.models : [],
      detail: (r && r.detail) || '',
      host: (r && r.host) || '',
    };
  } catch (e) {
    return { available: false, models: [], detail: (e && e.detail) || String(e && e.message || e), host: '' };
  }
}

/* =====================================================================
 * Toasts
 * ===================================================================== */
function toast(title, detail, kind = 'info', ttl = 6000) {
  const box = $('#toasts');
  const t = el('div', { class: `toast ${kind}` }, [
    el('div', { class: 'ttitle', text: title }),
    detail ? el('div', { class: 'tdetail', text: detail }) : null,
  ]);
  t.addEventListener('click', () => t.remove());
  box.appendChild(t);
  if (ttl) setTimeout(() => t.remove(), ttl);
}
function toastErr(e, context) {
  if (e instanceof ApiError) {
    toast(context || e.error, e.detail || `(${e.status})`, 'error', 9000);
  } else {
    toast(context || 'Error', String(e && e.message || e), 'error', 9000);
  }
  // eslint-disable-next-line no-console
  console.error(context || 'error', e);
}

/* =====================================================================
 * Router
 * ===================================================================== */
const ROUTES = {
  upload: renderUpload,
  settings: renderSettings,
  routing: renderRouting,
  courses: renderCourses,
  retrieval: renderRetrieval,
  activity: renderActivity,
};
const DEFAULT_TAB = 'upload';

let activeTeardown = null; // optional cleanup fn returned by a render

function currentTab() {
  const h = (location.hash || '').replace(/^#\/?/, '').split('/')[0];
  return ROUTES[h] ? h : DEFAULT_TAB;
}

async function route() {
  if (typeof activeTeardown === 'function') { try { activeTeardown(); } catch (_) {} activeTeardown = null; }
  const tab = currentTab();
  $$('#navlist a').forEach((a) => a.classList.toggle('active', a.dataset.tab === tab));
  const view = clear($('#view'));
  view.appendChild(el('div', { class: 'loading', text: 'Loading…' }));
  try {
    const teardown = await ROUTES[tab](view);
    if (typeof teardown === 'function') activeTeardown = teardown;
  } catch (e) {
    clear(view).appendChild(el('div', { class: 'card' }, [
      el('h1', { text: 'Failed to load tab' }),
      el('p', { class: 'subtitle', text: (e instanceof ApiError ? `${e.error}: ${e.detail}` : String(e)) }),
      el('button', { class: 'ghost', text: 'Retry', onclick: () => route() }),
    ]));
    toastErr(e, 'Tab load failed');
  }
}

window.addEventListener('hashchange', route);

/* =====================================================================
 * Connection indicator
 * ===================================================================== */
async function pingConnection() {
  const dot = $('#conn-status'), label = $('#conn-label');
  try {
    await api('/api/settings');
    dot.className = 'conn-dot ok';
    label.textContent = 'connected';
  } catch (e) {
    dot.className = 'conn-dot bad';
    label.textContent = (e instanceof ApiError && e.status === 0) ? 'offline' : 'api error';
  }
}

/* =====================================================================
 * TAB: Upload & Run
 * GET  /api/workflows
 * GET/POST/DELETE /api/uploads
 * POST /api/runs   |  POST /api/runs/phase
 * WS   /api/ws/runs/{run_id}
 * ===================================================================== */
async function renderUpload(view) {
  const [workflows, uploads] = await Promise.all([
    api('/api/workflows').catch((e) => { toastErr(e, 'workflows'); return []; }),
    api('/api/uploads').catch((e) => { toastErr(e, 'uploads'); return []; }),
  ]);
  const wf = Array.isArray(workflows) ? workflows : (workflows && workflows.workflows) || [];
  const ups = Array.isArray(uploads) ? uploads : (uploads && uploads.uploads) || [];

  clear(view);
  view.appendChild(el('h1', { text: 'Upload & Run' }));
  view.appendChild(el('p', { class: 'subtitle', text: 'Upload a corpus, pick a workflow or single phase, and launch a run with live logs.' }));

  // ---- state ----
  let selectedUpload = null;          // upload_id
  let selectedCorpusPath = null;      // explicit path override
  let ws = null;

  // ---- Upload card ----
  const dz = el('div', { class: 'dropzone', text: 'Drag & drop PDF / IMSCC files here, or click to browse' });
  const fileInput = el('input', { type: 'file', multiple: true, style: 'display:none', accept: '.pdf,.imscc,.zip' });
  const fileList = el('div', { class: 'filelist' });
  dz.appendChild(fileInput);
  dz.addEventListener('click', () => fileInput.click());
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => { if (e.dataTransfer.files.length) doUpload(e.dataTransfer.files); });
  fileInput.addEventListener('change', () => { if (fileInput.files.length) doUpload(fileInput.files); });

  const uploadSelect = el('select');
  const deleteUploadBtn = el('button', { class: 'danger sm', text: 'Remove', 'aria-label': 'Remove selected upload', disabled: true });
  function refreshUploadSelect(list) {
    clear(uploadSelect);
    uploadSelect.appendChild(el('option', { value: '', text: list.length ? '— select an upload —' : 'no uploads yet' }));
    list.forEach((u) => {
      const names = (u.files || []).map((f) => f.name).join(', ');
      uploadSelect.appendChild(el('option', { value: u.upload_id, text: `${u.upload_id} (${names || (u.files || []).length + ' files'})` }));
    });
    deleteUploadBtn.disabled = !uploadSelect.value;
  }
  refreshUploadSelect(ups);
  uploadSelect.addEventListener('change', () => {
    selectedUpload = uploadSelect.value || null;
    deleteUploadBtn.disabled = !uploadSelect.value;
  });
  deleteUploadBtn.addEventListener('click', async () => {
    const id = uploadSelect.value;
    if (!id) return;
    if (!confirm(`Delete upload ${id}? This removes its staged files.`)) return;
    deleteUploadBtn.disabled = true;
    try {
      await api(`/api/uploads/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (selectedUpload === id) selectedUpload = null;
      const fresh = await api('/api/uploads').catch(() => null);
      const list = Array.isArray(fresh) ? fresh : (fresh && fresh.uploads) || [];
      refreshUploadSelect(list);
      toast('Removed', `Upload ${id} deleted.`, 'success');
    } catch (e) {
      toastErr(e, 'Delete upload failed');
      deleteUploadBtn.disabled = !uploadSelect.value;
    }
  });

  async function doUpload(files) {
    const fd = new FormData();
    for (const f of files) fd.append('files', f, f.name);
    clear(fileList).appendChild(el('div', { class: 'loading', text: `Uploading ${files.length} file(s)…` }));
    try {
      const res = await api('/api/uploads', { method: 'POST', body: fd });
      clear(fileList);
      (res.files || []).forEach((f) => fileList.appendChild(el('div', { class: 'fileitem' }, [
        el('span', { text: `${f.name} · ${f.kind || ''}` }),
        el('span', { class: 'muted', text: fmtBytes(f.size) }),
      ])));
      selectedUpload = res.upload_id;
      // re-pull list & select
      const fresh = await api('/api/uploads').catch(() => null);
      const list = Array.isArray(fresh) ? fresh : (fresh && fresh.uploads) || ups;
      refreshUploadSelect(list);
      uploadSelect.value = res.upload_id;
      toast('Uploaded', `${(res.files || []).length} file(s) → ${res.upload_id}`, 'success');
    } catch (e) {
      clear(fileList);
      toastErr(e, 'Upload failed');
    }
  }

  const reuseId = uid('in');
  uploadSelect.id = reuseId;
  const corpusPathInput = el('input', { type: 'text', placeholder: '/abs/path/to/textbook.pdf or ./pdfs/', oninput: (e) => { selectedCorpusPath = e.target.value.trim() || null; } });
  view.appendChild(el('div', { class: 'card' }, [
    el('h2', { text: '1. Corpus' }),
    dz,
    fileList,
    el('div', { class: 'field', style: 'margin-top:14px' }, [
      (() => { const l = el('label', { text: 'Or reuse a prior upload' }); l.htmlFor = reuseId; return l; })(),
      el('div', { class: 'inline' }, [uploadSelect, deleteUploadBtn]),
    ]),
    field('Or explicit corpus path (overrides upload)', corpusPathInput),
  ]));

  // ---- Workflow + options ----
  const modeSingle = el('input', { type: 'checkbox' });
  const wfSelect = el('select');
  wf.forEach((w) => wfSelect.appendChild(el('option', { value: w.name, text: w.name })));
  if (!wf.length) wfSelect.appendChild(el('option', { value: '', text: 'no workflows available' }));

  const phaseSelect = el('select', { disabled: true });
  function refreshPhases() {
    clear(phaseSelect);
    const w = wf.find((x) => x.name === wfSelect.value);
    const phases = (w && w.phases) || [];
    if (!phases.length) phaseSelect.appendChild(el('option', { value: '', text: 'no phases listed' }));
    phases.forEach((p) => {
      const gc = p.validation_gates_count != null ? ` · ${p.validation_gates_count} gates` : '';
      phaseSelect.appendChild(el('option', { value: p.name, text: `${p.name}${gc}` }));
    });
  }
  wfSelect.addEventListener('change', refreshPhases);
  refreshPhases();
  modeSingle.addEventListener('change', () => { phaseSelect.disabled = !modeSingle.checked; });

  const courseName = el('input', { type: 'text', placeholder: 'e.g. PHYS_101' });
  const weeks = el('input', { type: 'number', min: '1', placeholder: 'auto' });
  const projectId = el('input', { type: 'text', placeholder: '(phase runs) existing project export id, optional' });
  const runMode = el('select');
  [['', 'inherit settings'], ['local', 'local'], ['api', 'api']].forEach(([v, t]) => runMode.appendChild(el('option', { value: v, text: t })));
  const provider = el('input', { type: 'text', placeholder: 'override provider (optional)' });
  const model = el('input', { type: 'text', placeholder: 'override model (optional)' });

  const launchBtn = el('button', { text: 'Launch run' });

  view.appendChild(el('div', { class: 'card' }, [
    el('h2', { text: '2. Workflow' }),
    el('div', { class: 'grid2' }, [
      field('Workflow', wfSelect),
      field('Course name', courseName),
    ]),
    el('label', { class: 'toggle', style: 'margin:6px 0 12px' }, [modeSingle, el('span', { text: 'Run a single phase only' })]),
    el('div', { class: 'grid2' }, [
      field('Phase', phaseSelect),
      field('Project id (phase reuse)', projectId),
    ]),
    el('div', { class: 'grid2' }, [
      field('Weeks', weeks),
      field('Mode', runMode),
    ]),
    el('div', { class: 'grid2' }, [
      field('Provider override', provider),
      field('Model override', model),
    ]),
    el('div', { class: 'btn-row' }, [launchBtn]),
  ]));

  // ---- Live console ----
  let currentRunId = null;
  let runStartMs = null;
  let elapsedTimer = null;
  let lineCount = 0;
  let terminalToasted = false; // fire the run-complete toast at most once per run
  const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'canceled', 'error']);

  const statusLine = el('span', { class: 'run-status' }, [el('span', { class: 'dot' }), el('span', { class: 'state', text: 'idle' }), el('span', { class: 'meta muted' })]);
  // The visual Build Console (shared run-progress.js) mounts here. The RAW log
  // affordance (consoleEl below + copy/download) is PRESERVED verbatim — the
  // operator keeps the raw stream; the rings/narrative are an addition above it.
  const progressMount = el('div', { class: 'progress-mount' });
  const consoleEl = el('div', { class: 'console', text: '' });
  const gateWrap = el('div');
  const cancelBtn = el('button', { class: 'danger sm', text: 'Cancel run', disabled: true });
  const copyLogBtn = el('button', { class: 'ghost sm', text: 'Copy log' });
  const dlLogBtn = el('button', { class: 'ghost sm', text: 'Download .log' });
  const idBar = el('div', { class: 'inline', style: 'flex-wrap:wrap;margin-bottom:8px' });
  view.appendChild(el('div', { class: 'card' }, [
    el('div', { class: 'flex wrap' }, [
      el('h2', { text: '3. Run console', style: 'flex:1;margin:0;border:none' }),
      statusLine,
      cancelBtn, copyLogBtn, dlLogBtn,
    ]),
    idBar,
    progressMount,
    consoleEl,
    gateWrap,
  ]));

  // ---- shared Build Console (rings + elapsed + narrative) ----
  // app.js is a CLASSIC script (the type=module flip is a separate Phase-2
  // commit + would break the /dev embed that reads window.renderUpload). A
  // dynamic import() works in a classic script with ZERO module-flip, so we pull
  // the kit lazily and drive it from the same WS line stream. PURE PRESENTATION:
  // the console only parses lines this page already owns.
  let progress = null;           // the live console controller (or null)
  let progressReady = null;      // promise of the imported factory
  function loadProgressFactory() {
    if (!progressReady) progressReady = import('/shared/components/run-progress.js').then((m) => m.runProgressConsole);
    return progressReady;
  }
  // Best-effort per-phase medians for the honest ETA (the sibling endpoint may
  // not exist yet → "estimating…", never a false number).
  async function fetchDurations(wfName) {
    try {
      const data = await api(`/api/runs/phase-durations?workflow=${encodeURIComponent(wfName)}`);
      return {
        durations: (data && typeof data.durations === 'object' && data.durations) || {},
        source: (data && typeof data.source === 'string' && data.source) || null,
      };
    } catch (_) { return { durations: {}, source: null }; }
  }
  // Build (or rebuild) the console for a run, anchored at startMs. Best-effort:
  // an import failure leaves the raw log fully functional (no console shown).
  async function mountProgress(startMs) {
    try {
      const factory = await loadProgressFactory();
      const wfName = wfSelect.value;
      const w = wf.find((x) => x.name === wfName);
      const phases = (w && Array.isArray(w.phases)) ? w.phases : [];
      const dur = await fetchDurations(wfName);
      clear(progressMount);
      if (progress) progress.destroy();
      // The console owns its own role=status narrative line (it is a self-
      // contained subtree inside the dense operator console card). It does NOT
      // reuse <main id="view"> — announce() writes textContent, which would wipe
      // the whole operator UI. The header cancel button is owned by the console;
      // THIS page owns the POST /api/runs/{id}/cancel fetch + the two-stage 202.
      progress = factory({
        phases: phases.map((p) => ({ name: p.name, label: p.label || p.name, optional: p.optional })),
        startMs: startMs || Date.now(),
        durations: dur.durations,
        durationsSource: dur.source,
        onCancel: () => doCancel(),
      });
      progressMount.appendChild(progress.el);
      progress.startFirstRunning();
    } catch (_) { progress = null; /* raw log still works */ }
  }
  function feedProgressLine(line) { if (progress) { try { progress.onLine(line); } catch (_) {} } }
  function feedProgressStatus(s) {
    if (!progress) return;
    try {
      progress.onStatus(s === 'canceled' ? 'cancelled' : s);
      if (TERMINAL.has(s)) progress.freezeElapsed(runStartMs != null ? Date.now() - runStartMs : undefined);
    } catch (_) {}
  }

  function fmtElapsed(ms) {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    return m ? `${m}m ${s % 60}s` : `${s}s`;
  }
  function refreshMeta() {
    const parts = [];
    if (runStartMs != null) parts.push(`elapsed ${fmtElapsed(Date.now() - runStartMs)}`);
    parts.push(`${lineCount} line${lineCount === 1 ? '' : 's'}`);
    statusLine.querySelector('.meta').textContent = ` · ${parts.join(' · ')}`;
  }
  function logLine(text, cls) {
    const atBottom = consoleEl.scrollTop + consoleEl.clientHeight >= consoleEl.scrollHeight - 8;
    consoleEl.appendChild(el('span', { class: `ln ${cls || ''}`, text }));
    lineCount += 1;
    refreshMeta();
    if (atBottom) consoleEl.scrollTop = consoleEl.scrollHeight;
  }
  function setStatus(s) {
    const dot = statusLine.querySelector('.dot');
    dot.className = 'dot ' + (s === 'completed' ? 'completed' : (s === 'failed' || s === 'cancelled' || s === 'canceled' || s === 'error') ? 'failed' : (s === 'running' || s === 'requested') ? 'running' : '');
    statusLine.querySelector('.state').textContent = s;
    if (TERMINAL.has(s)) feedProgressStatus(s);
    if (TERMINAL.has(s)) {
      if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
      cancelBtn.disabled = true;
      refreshMeta();
      if (!terminalToasted) {
        terminalToasted = true;
        const ok = s === 'completed';
        const elapsed = runStartMs != null ? ` in ${fmtElapsed(Date.now() - runStartMs)}` : '';
        toast(ok ? 'Run complete' : `Run ${s}`, `${currentRunId || ''}${elapsed}`.trim(), ok ? 'success' : 'error', ok ? 6000 : 9000);
      }
    }
  }
  function startElapsed() {
    runStartMs = Date.now();
    if (elapsedTimer) clearInterval(elapsedTimer);
    elapsedTimer = setInterval(refreshMeta, 1000);
    refreshMeta();
  }

  // Two-stage cancel. POST /api/runs/{id}/cancel now returns HTTP 202
  // {status:"cancel_requested"} (a 2xx, so api() returns it normally); the
  // terminal `cancelled` arrives LATER via the WS {type:status} frame. We
  // therefore do NOT mark the run terminal here — we show "Cancelling…" and let
  // setStatus(terminal) fire from the WS. A legacy 200 with a synchronous
  // terminal status is still honoured. Used by both the toolbar Cancel button
  // and the console's header cancel (onCancel → doCancel).
  async function doCancel() {
    if (!currentRunId) return;
    if (cancelBtn.disabled) return; // already requested
    if (!confirm(`Cancel run ${currentRunId}?`)) return;
    cancelBtn.disabled = true;
    cancelBtn.textContent = 'Cancelling…';
    try {
      const r = await api(`/api/runs/${encodeURIComponent(currentRunId)}/cancel`, { method: 'POST' });
      const st = r && r.status;
      logLine(`[gui] cancel requested for ${currentRunId} (${st || 'cancel_requested'})`, 'sys');
      if (progress) { try { progress.setCancelling(); } catch (_) {} }
      if (st === 'cancelled' || st === 'canceled' || st === 'interrupted' || st === 'error' || st === 'failed') {
        // Synchronous terminal (legacy 200) — resolve now.
        setStatus(st);
      } else {
        // 202 / cancel_requested: NOT terminal. Wait for the WS terminal frame.
        toast('Cancelling…', 'The run will stop after the current step finishes.', 'info');
      }
    } catch (e) {
      toastErr(e, 'Cancel failed');
      cancelBtn.disabled = false;
      cancelBtn.textContent = 'Cancel run';
    }
  }
  cancelBtn.addEventListener('click', doCancel);
  copyLogBtn.addEventListener('click', async () => {
    const ok = await copyText(consoleEl.textContent || '');
    toast(ok ? 'Log copied' : 'Copy failed', ok ? `${lineCount} line${lineCount === 1 ? '' : 's'}` : '', ok ? 'success' : 'error', 2500);
  });
  dlLogBtn.addEventListener('click', () => {
    const blob = new Blob([consoleEl.textContent || ''], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = el('a', { href: url, download: `${currentRunId || 'run'}.log` });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  function renderGates(gates) {
    clear(gateWrap);
    if (!gates || !gates.length) return;
    gateWrap.appendChild(el('h3', { text: 'Gate results' }));
    const rows = gates.map((g) => {
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

  function openWs(runId) {
    if (ws) { try { ws.close(); } catch (_) {} ws = null; }
    currentRunId = runId;
    cancelBtn.disabled = false;
    cancelBtn.textContent = 'Cancel run'; // reset from a prior "Cancelling…"
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Operator WS is token-gated in full mode; browser JS can't set WS request
    // headers, so the token rides the ?token= query param (server constant-time
    // compares it). Omitted when no token is held (open deploys).
    const token = getToken();
    const q = token ? `?token=${encodeURIComponent(token)}` : '';
    const url = `${proto}//${location.host}/api/ws/runs/${encodeURIComponent(runId)}${q}`;
    logLine(`[gui] connecting ${url.replace(/\?token=[^&]*/, '?token=***')}`, 'sys');
    try {
      ws = new WebSocket(url);
    } catch (e) {
      logLine(`[gui] websocket failed: ${e}`, 'err');
      return;
    }
    ws.onopen = () => logLine('[gui] log stream open', 'sys');
    ws.onmessage = (ev) => {
      let msg = ev.data;
      // Messages may be plain log lines or JSON {type, line|status|gates}
      if (typeof msg === 'string' && msg.startsWith('{')) {
        try {
          const obj = JSON.parse(msg);
          if (obj.type === 'status' || obj.status) { setStatus(obj.status || obj.type); }
          if (obj.line != null) { logLine(obj.line); feedProgressLine(String(obj.line)); }
          if (obj.gates || obj.gate_results) renderGates(obj.gates || obj.gate_results);
          if (obj.error) logLine(`[error] ${obj.error}: ${obj.detail || ''}`, 'err');
          return;
        } catch (_) { /* fall through: treat as raw line */ }
      }
      logLine(String(msg));
      feedProgressLine(String(msg));
    };
    ws.onerror = () => logLine('[gui] websocket error', 'err');
    ws.onclose = () => logLine('[gui] log stream closed', 'sys');
  }

  async function launch() {
    const name = courseName.value.trim();
    if (!name) { toast('Course name required', 'Enter a course name before launching.', 'error'); return; }
    const corpus = selectedCorpusPath || selectedUpload;
    const single = modeSingle.checked;
    if (!single && !corpus) { toast('Corpus required', 'Upload a file or supply a corpus path.', 'error'); return; }

    clear(consoleEl); clear(gateWrap); clear(progressMount);
    lineCount = 0;
    terminalToasted = false;
    startElapsed();
    mountProgress(runStartMs);
    setStatus('requested');
    launchBtn.disabled = true;
    try {
      let res;
      if (single) {
        const body = {
          workflow: wfSelect.value,
          phase: phaseSelect.value,
          course_name: name,
          project_id: projectId.value.trim() || undefined,
          options: collectOptions(),
        };
        res = await apiJSON('/api/runs/phase', 'POST', body);
      } else {
        const body = {
          workflow: wfSelect.value,
          course_name: name,
          corpus,
          weeks: weeks.value ? Number(weeks.value) : undefined,
          mode: runMode.value || undefined,
          provider: provider.value.trim() || undefined,
          model: model.value.trim() || undefined,
          options: collectOptions(),
        };
        res = await apiJSON('/api/runs', 'POST', body);
      }
      const runId = res && (res.run_id || res.runId);
      const wfId = res && res.workflow_id || '';
      logLine(`[gui] launched: run_id=${runId} workflow_id=${wfId} status=${res && res.status || ''}`, 'sys');
      clear(idBar);
      if (runId) idBar.append(el('span', { class: 'muted', text: 'run_id ' }), el('code', { class: 'kv', text: String(runId) }), copyBtn(String(runId)));
      if (wfId) idBar.append(el('span', { class: 'muted', text: ' workflow_id ' }), el('code', { class: 'kv', text: String(wfId) }), copyBtn(String(wfId)));
      setStatus((res && res.status) || 'running');
      // single-phase responses may carry tasks + gate results synchronously
      if (res && (res.gates || res.gate_results)) renderGates(res.gates || res.gate_results);
      if (res && res.tasks) logLine(`[gui] tasks: ${JSON.stringify(res.tasks)}`);
      if (runId) openWs(runId);
      else logLine('[gui] no run_id returned; cannot open log stream', 'err');
    } catch (e) {
      setStatus('failed');
      toastErr(e, 'Launch failed');
      logLine(`[error] ${e instanceof ApiError ? e.error + ': ' + e.detail : e}`, 'err');
    } finally {
      launchBtn.disabled = false;
    }
  }
  function collectOptions() {
    // generic options bag; backend ignores unknowns gracefully
    const o = {};
    if (projectId.value.trim()) o.project_id = projectId.value.trim();
    return o;
  }
  launchBtn.addEventListener('click', launch);

  // ---- Recent runs (reload-resume) ----
  // GET /api/runs is newest-first; clicking re-attaches the existing WS console
  // (the server replays the log from offset 0), so a run survives page reload.
  function reattach(runId, status) {
    clear(consoleEl); clear(gateWrap); clear(idBar); clear(progressMount);
    lineCount = 0;
    terminalToasted = !!(status && TERMINAL.has(status)); // don't re-toast a finished run on reattach
    idBar.append(el('span', { class: 'muted', text: 'run_id ' }), el('code', { class: 'kv', text: String(runId) }), copyBtn(String(runId)));
    if (status && TERMINAL.has(status)) {
      runStartMs = null;
      if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    } else {
      startElapsed();
    }
    // The WS replays the run's log from offset 0, so the console rebuilds its
    // phase state from the replayed [phase]/[progress] lines.
    mountProgress(runStartMs || Date.now());
    setStatus(status || 'running');
    openWs(runId);
    consoleEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  const recentList = el('div', { class: 'recent-runs' }, el('div', { class: 'loading', text: 'Loading runs…' }));
  const refreshRunsBtn = el('button', { class: 'ghost sm', text: 'Refresh' });
  async function refreshRuns() {
    try {
      const data = await api('/api/runs');
      const runs = Array.isArray(data) ? data : (data && data.runs) || [];
      clear(recentList);
      if (!runs.length) { recentList.appendChild(el('div', { class: 'empty', text: 'No runs yet.' })); return; }
      const tbl = el('table', {}, [
        el('thead', {}, el('tr', {}, [el('th', { text: 'Run' }), el('th', { text: 'Workflow' }), el('th', { text: 'Status' }), el('th', { text: '' })])),
      ]);
      const tbody = el('tbody');
      runs.forEach((r) => {
        const rid = r.run_id || r.runId || r.id;
        const st = r.status || '';
        const dotCls = st === 'completed' ? 'completed' : (st === 'failed' || st === 'cancelled' || st === 'canceled' || st === 'error') ? 'failed' : (st === 'running' || st === 'requested') ? 'running' : '';
        const tr = el('tr', { class: 'clickable' }, [
          el('td', {}, el('code', { class: 'kv', text: String(rid) })),
          el('td', { text: r.workflow || r.workflow_name || '' }),
          el('td', { class: 'run-status' }, [el('span', { class: `dot ${dotCls}` }), el('span', { text: st || '—' })]),
          el('td', {}, copyBtn(String(rid))),
        ]);
        tr.addEventListener('click', () => reattach(rid, st));
        tbody.appendChild(tr);
      });
      tbl.appendChild(tbody);
      recentList.appendChild(tbl);
    } catch (e) {
      clear(recentList).appendChild(el('div', { class: 'empty', text: `Failed to load runs: ${e instanceof ApiError ? e.error : e}` }));
    }
  }
  refreshRunsBtn.addEventListener('click', refreshRuns);
  view.appendChild(el('div', { class: 'card' }, [
    el('div', { class: 'flex wrap' }, [el('h2', { text: 'Recent runs', style: 'flex:1;margin:0;border:none' }), refreshRunsBtn]),
    el('p', { class: 'subtitle', style: 'margin:8px 0 12px', text: 'Click a run to re-attach its live log console (survives page reload).' }),
    recentList,
  ]));
  refreshRuns();

  return () => { if (ws) { try { ws.close(); } catch (_) {} } if (elapsedTimer) clearInterval(elapsedTimer); if (progress) { try { progress.destroy(); } catch (_) {} } };
}

/* =====================================================================
 * TAB: Settings / API Keys
 * GET /api/settings  -> { ...doc, catalog, providers, base_models }
 * PATCH /api/settings
 * POST /api/settings/test-provider {provider}
 * ===================================================================== */
async function renderSettings(view) {
  const data = await api('/api/settings');
  const catalog = data.catalog || [];
  const providers = data.providers || [];
  const env = data.env || {};
  const flags = data.flags || {};

  // group catalog by category (server may also provide by_category)
  const byCat = {};
  (Array.isArray(catalog) ? catalog : Object.values(catalog).flat()).forEach((entry) => {
    (byCat[entry.category] = byCat[entry.category] || []).push(entry);
  });

  clear(view);
  view.appendChild(el('h1', { text: 'Settings / API Keys' }));
  view.appendChild(el('p', { class: 'subtitle', text: 'Plain-language settings grouped by AI tier. Each field shows its underlying environment variable as a hint. Secrets are write-only — they send only on save.' }));

  const pending = {}; // key -> value to PATCH

  function fieldFor(entry) {
    const key = entry.key;
    const isSecret = entry.type === 'secret';
    const curVal = (env[key] != null ? env[key] : (flags[key] != null ? flags[key] : entry.default));
    let inputNode;

    if (entry.type === 'bool') {
      const cb = el('input', { type: 'checkbox' });
      cb.checked = curVal === true || curVal === 'true' || curVal === 1;
      cb.addEventListener('change', () => { pending[key] = cb.checked; });
      inputNode = el('label', { class: 'toggle' }, [cb, el('span', { text: cb.checked ? 'enabled' : 'disabled' })]);
      cb.addEventListener('change', () => { inputNode.querySelector('span').textContent = cb.checked ? 'enabled' : 'disabled'; });
    } else if (entry.type === 'enum' && Array.isArray(entry.enum)) {
      const sel = el('select');
      sel.appendChild(el('option', { value: '', text: '— unset —' }));
      entry.enum.forEach((o) => {
        const opt = el('option', { value: o, text: o });
        if (String(curVal) === String(o)) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', () => { pending[key] = sel.value; });
      inputNode = sel;
    } else if (isSecret) {
      // masked: env[key] is "set"/null per spec mask_secrets
      const setState = curVal === 'set' || curVal === true;
      const inp = el('input', { type: 'password', placeholder: setState ? '•••••••• (leave blank to keep)' : 'paste secret to set' });
      inp.addEventListener('input', () => {
        if (inp.value === '') delete pending[key];
        else pending[key] = inp.value;
      });
      inputNode = el('div', { class: 'inline' }, [
        inp,
        el('span', { class: `badge ${setState ? 'set' : 'unset'}`, text: setState ? 'set' : 'unset' }),
      ]);
    } else {
      const inp = el('input', { type: entry.type === 'number' ? 'number' : 'text', value: (curVal != null && curVal !== 'set') ? curVal : '' });
      inp.addEventListener('input', () => { pending[key] = entry.type === 'number' ? (inp.value === '' ? null : Number(inp.value)) : inp.value; });
      inputNode = inp;
    }

    // Friendly human label (e.g. "Outline course model"), NOT the bare env-var.
    const label = el('label', { class: 'field-label', text: entry.label || key });
    // Point the label at the first focusable control inside inputNode (the
    // toggle/secret cases wrap the real input in a div).
    const focusable = inputNode.matches && inputNode.matches('input,select,textarea')
      ? inputNode
      : inputNode.querySelector && inputNode.querySelector('input,select,textarea');
    if (focusable) { focusable.id = focusable.id || uid('cfg'); label.htmlFor = focusable.id; }

    // Help + the bare env-var name kept as a small secondary hint. The operator
    // surface is where technical readers live, so we keep the raw key visible
    // (in a <code>) — but the LABEL the eye lands on is the friendly one. The
    // help text is wired to the control via aria-describedby so SRs read it.
    const helpBits = [];
    if (entry.help) helpBits.push(el('span', { class: 'help-text', text: entry.help }));
    helpBits.push(el('span', { class: 'help-env' }, [
      el('code', { class: 'kv', text: key }),
      entry.applies_to ? el('span', { class: 'help-applies', text: ` · applies to: ${entry.applies_to}` }) : null,
    ]));
    const helpNode = el('div', { class: 'help' }, helpBits);
    if (focusable) {
      const helpId = uid('cfg-h');
      helpNode.id = helpId;
      // Preserve any existing describedby (none today) and append ours.
      const prior = focusable.getAttribute('aria-describedby');
      focusable.setAttribute('aria-describedby', prior ? `${prior} ${helpId}` : helpId);
    }

    return el('div', { class: 'field field-row' }, [
      label,
      inputNode,
      helpNode,
    ]);
  }

  // Render each category card with a FRIENDLY group heading (plain-language by
  // routing: the AI tier / category, not the raw catalog category slug). The
  // raw category slug stays available as a small secondary hint on the card so
  // a technical operator can still map a group back to the catalog.
  const order = ['credentials', 'global', 'global routing', 'local', 'local backend', 'together', 'vision', 'dart', 'courseforge', 'trainforge', 'embedding', 'answer', 'base models (training)'];
  // Friendly group headings keyed by catalog category. Anything not listed
  // falls back to a title-cased version of the slug (never a bare slug).
  const GROUP_TITLES = {
    credentials: 'API keys & credentials',
    global: 'Global AI routing',
    local: 'Local model server (Ollama)',
    together: 'Together AI models',
    vision: 'Vision / image models',
    dart: 'PDF conversion (DART)',
    courseforge: 'Course authoring (Courseforge)',
    trainforge: 'Training & assessments (Trainforge)',
    embedding: 'Search & embeddings',
    answer: 'Course Q&A (answers)',
  };
  const groupTitle = (cat) => GROUP_TITLES[cat]
    || String(cat).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  const cats = Object.keys(byCat).sort((a, b) => {
    const ia = order.indexOf(a), ib = order.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
  for (const cat of cats) {
    const entries = byCat[cat];
    const card = el('div', { class: 'card' }, [
      el('h2', { text: groupTitle(cat) }),
      el('p', { class: 'subtitle group-cat-hint' }, [
        el('span', { text: 'Settings group: ' }),
        el('code', { class: 'kv', text: cat }),
      ]),
    ]);
    const grid = el('div', { class: 'grid2' });
    entries.forEach((e) => grid.appendChild(fieldFor(e)));
    card.appendChild(grid);
    view.appendChild(card);
  }

  // Ollama models helper — live discovery of installed local (Ollama) models.
  // Mirrors the Model Routing vision-model discovery; uses the same
  // /api/settings/ollama-models probe so the operator can confirm what is
  // installed on the local backend before pinning LOCAL_SYNTHESIS_MODEL.
  {
    const card = el('div', { class: 'card' }, [
      el('h2', { text: 'Ollama models (local backend)' }),
      el('p', { class: 'subtitle', text: 'Live probe of the local Ollama server (LOCAL_SYNTHESIS_BASE_URL → /api/tags). Set LOCAL_SYNTHESIS_MODEL above to one of these; flip LOCAL_VISION_CAPABLE for a vision model.' }),
    ]);
    const statusLine = el('div', { class: 'help', text: 'Click Refresh to query the local server.' });
    const listBox = el('div', { class: 'inline' });
    const refreshBtn = el('button', { class: 'ghost sm', text: 'Refresh' });
    refreshBtn.addEventListener('click', async () => {
      refreshBtn.disabled = true;
      statusLine.textContent = 'Querying local Ollama server…';
      clear(listBox);
      const res = await fetchOllamaModels();
      if (res.available && res.models.length) {
        statusLine.textContent = `${res.models.length} model(s) at ${res.host}.`;
        res.models.forEach((m) => listBox.appendChild(el('code', { class: 'kv', text: m })));
      } else if (res.available) {
        statusLine.textContent = `Reachable at ${res.host} but no models installed.`;
      } else {
        statusLine.textContent = `Ollama not reachable: ${res.detail || 'connection failed'}`;
      }
      refreshBtn.disabled = false;
    });
    card.appendChild(el('div', { class: 'btn-row' }, [refreshBtn]));
    card.appendChild(statusLine);
    card.appendChild(listBox);
    view.appendChild(card);
  }

  // Provider test card
  if (providers.length) {
    const card = el('div', { class: 'card' }, [
      el('h2', { text: 'Provider reachability' }),
      el('p', { class: 'subtitle', text: 'Live check: key presence and (where applicable) a 1-token ping or /models probe.' }),
    ]);
    const tbl = el('table', {}, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'Provider' }), el('th', { text: 'Key env' }), el('th', { text: 'Status' }), el('th', { text: '' })])),
    ]);
    const tbody = el('tbody');
    providers.forEach((p) => {
      const name = p.name || p;
      const statusCell = el('td', { class: 'run-status' }, [el('span', { class: 'muted', text: '—' })]);
      const btn = el('button', { class: 'ghost sm', text: 'Test' });
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        clear(statusCell).appendChild(el('span', { class: 'muted', text: 'testing…' }));
        try {
          const r = await apiJSON('/api/settings/test-provider', 'POST', { provider: name });
          const ok = r.ok === true || r.reachable === true || r.status === 'ok';
          clear(statusCell).appendChild(el('span', { class: `badge ${ok ? 'ok' : 'fail'}`, text: ok ? 'reachable' : (r.status || 'unreachable') }));
          if (r.detail) statusCell.appendChild(el('span', { class: 'help', text: ' ' + r.detail }));
        } catch (e) {
          clear(statusCell).appendChild(el('span', { class: 'badge fail', text: e instanceof ApiError ? e.error : 'error' }));
          toastErr(e, `Test ${name}`);
        } finally { btn.disabled = false; }
      });
      tbody.appendChild(el('tr', {}, [
        el('td', { text: name }),
        el('td', {}, el('code', { class: 'kv', text: p.api_key_env || '' })),
        statusCell,
        el('td', {}, btn),
      ]));
    });
    tbl.appendChild(tbody);
    card.appendChild(tbl);
    view.appendChild(card);
  }

  // Save bar
  const saveBtn = el('button', { text: 'Save changes' });
  saveBtn.addEventListener('click', async () => {
    if (!Object.keys(pending).length) { toast('Nothing to save', 'No fields were changed.', 'info'); return; }
    // Split pending into env vs flags by catalog type
    const typeByKey = {};
    (Array.isArray(catalog) ? catalog : Object.values(catalog).flat()).forEach((e) => { typeByKey[e.key] = e.type; });
    const patch = { env: {}, flags: {} };
    for (const [k, v] of Object.entries(pending)) {
      if (typeByKey[k] === 'bool') patch.flags[k] = v;
      else patch.env[k] = v;
    }
    if (!Object.keys(patch.env).length) delete patch.env;
    if (!Object.keys(patch.flags).length) delete patch.flags;
    saveBtn.disabled = true;
    try {
      await apiJSON('/api/settings', 'PATCH', patch);
      toast('Saved', 'Settings persisted.', 'success');
      route(); // re-render to reflect masked state
    } catch (e) {
      toastErr(e, 'Save failed');
    } finally { saveBtn.disabled = false; }
  });
  const applyBtn = el('button', { class: 'ghost', text: 'Apply to process env' });
  applyBtn.addEventListener('click', async () => {
    applyBtn.disabled = true;
    try {
      const r = await apiJSON('/api/settings/apply', 'POST', {});
      const keys = r && (r.applied || r.keys || Object.keys(r || {}));
      toast('Applied', `Env vars set: ${Array.isArray(keys) ? keys.length : '?'}`, 'success');
    } catch (e) { toastErr(e, 'Apply failed'); }
    finally { applyBtn.disabled = false; }
  });
  view.appendChild(el('div', { class: 'btn-row' }, [saveBtn, applyBtn]));
}

/* =====================================================================
 * TAB: Model Routing
 * GET /api/settings (reuse providers/base_models/model_routing)
 * PATCH /api/settings { model_routing: {...} }
 * ===================================================================== */
async function renderRouting(view) {
  const data = await api('/api/settings');
  const providerObjs = data.providers || [];
  const providers = providerObjs.map((p) => p.name || p);
  // Vision-capable subset (provider.vision_capable marker from the catalog):
  // anthropic, together-vision, local (Ollama). Used by the Vision/VLM row.
  const visionProviders = providerObjs
    .filter((p) => p && typeof p === 'object' && p.vision_capable)
    .map((p) => p.name);
  const providerLabel = (name) => {
    const m = providerObjs.find((p) => (p.name || p) === name);
    return (m && m.label) || name;
  };
  const baseModels = data.base_models || [];
  const routing = data.model_routing || {};
  const flags = data.flags || {};

  clear(view);
  view.appendChild(el('h1', { text: 'Model Routing' }));
  view.appendChild(el('p', { class: 'subtitle', text: 'Per-task provider + model. Blank inherits the global routing. Maps to *_PROVIDER / *_MODEL env vars on save.' }));

  // Tasks: keys present in routing + the canonical set from spec §3.
  // ``vision`` is rendered in its own dedicated card below (provider list
  // filtered to vision-capable backends + live Ollama model discovery), so
  // it is excluded from the generic per-task table here.
  const canonical = ['global', 'dart', 'courseforge_outline', 'courseforge_rewrite', 'courseplanner', 'textbook_synthesis', 'trainforge_synthesis', 'trainforge_assessment'];
  const tasks = Array.from(new Set([...canonical, ...Object.keys(routing)])).filter((t) => t !== 'vision');

  const draft = JSON.parse(JSON.stringify(routing));
  function ensure(task) { return (draft[task] = draft[task] || {}); }

  const card = el('div', { class: 'card' });
  const tbl = el('table', {}, [
    el('thead', {}, el('tr', {}, [el('th', { text: 'Task' }), el('th', { text: 'Provider' }), el('th', { text: 'Model' }), el('th', { text: 'Extra' })])),
  ]);
  const tbody = el('tbody');

  function providerSelect(task, field) {
    const sel = el('select');
    sel.appendChild(el('option', { value: '', text: '— inherit —' }));
    providers.forEach((p) => {
      const opt = el('option', { value: p, text: p });
      if ((routing[task] || {})[field] === p) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => { ensure(task)[field] = sel.value || null; });
    return sel;
  }
  function modelInput(task, field) {
    // free-text model, with base_models as datalist suggestions
    const listId = `bm-${task}-${field}`;
    const inp = el('input', { type: 'text', value: (routing[task] || {})[field] || '', list: listId, placeholder: 'model id (optional)' });
    inp.addEventListener('input', () => { ensure(task)[field] = inp.value.trim() || null; });
    const dl = el('datalist', { id: listId });
    baseModels.forEach((m) => dl.appendChild(el('option', { value: typeof m === 'string' ? m : (m.name || m.id || '') })));
    return el('div', {}, [inp, dl]);
  }

  tasks.forEach((task) => {
    const cur = routing[task] || {};
    const extra = [];
    // global has a mode field; dart has vision_provider
    if (task === 'global') {
      const sel = el('select');
      [['', 'inherit'], ['local', 'local'], ['api', 'api']].forEach(([v, t]) => {
        const o = el('option', { value: v, text: t }); if ((cur.mode || '') === v) o.selected = true; sel.appendChild(o);
      });
      sel.addEventListener('change', () => { ensure('global').mode = sel.value || null; });
      extra.push(el('div', {}, [el('label', { text: 'mode' }), sel]));
    }
    if (task === 'dart') {
      extra.push(el('div', {}, [el('label', { text: 'vision_provider' }), providerSelect('dart', 'vision_provider')]));
    }
    tbody.appendChild(el('tr', {}, [
      el('td', {}, el('code', { class: 'kv', text: task })),
      el('td', {}, providerSelect(task, 'provider')),
      el('td', {}, modelInput(task, 'model')),
      el('td', {}, extra.length ? extra : el('span', { class: 'muted', text: '—' })),
    ]));
  });
  tbl.appendChild(tbody);
  card.appendChild(tbl);
  view.appendChild(card);

  // ---------------------------------------------------------------- Vision / VLM
  // Dedicated routing row for vision-capable backends. Provider dropdown is
  // filtered to the vision-capable subset (anthropic / together-vision /
  // local-as-Ollama). When the provider is ``local`` (Ollama), the model
  // field is populated from a LIVE /api/settings/ollama-models probe; on any
  // other provider (or an unreachable Ollama) it degrades to free-text entry.
  // Writes go to draft.vision and PATCH model_routing.vision on save.
  {
    const vCur = routing.vision || {};
    const vCard = el('div', { class: 'card' }, [
      el('h2', { text: 'Vision / VLM' }),
      el('p', { class: 'subtitle', text: 'Vision-capable backend for DART alt-text / image description (DART_VISION_PROVIDER). Local = Ollama with a vision model + LOCAL_VISION_CAPABLE.' }),
    ]);

    const provSel = el('select');
    provSel.appendChild(el('option', { value: '', text: '— inherit (falls through to DART provider) —' }));
    visionProviders.forEach((p) => {
      const opt = el('option', { value: p, text: providerLabel(p) });
      if ((vCur.provider || '') === p) opt.selected = true;
      provSel.appendChild(opt);
    });

    // Model field: a free-text input backed by a datalist of discovered
    // Ollama models (when provider=local) or together-vision suggestions.
    const listId = 'vision-model-options';
    const modelInp = el('input', {
      type: 'text',
      value: vCur.model || '',
      list: listId,
      placeholder: 'vision model id (optional)',
    });
    const dl = el('datalist', { id: listId });
    const modelHint = el('div', { class: 'help' });
    modelInp.addEventListener('input', () => { ensure('vision').model = modelInp.value.trim() || null; });

    async function refreshModelOptions(providerName) {
      clear(dl);
      if (providerName === 'local') {
        modelHint.textContent = 'Discovering installed Ollama models…';
        const res = await fetchOllamaModels();
        if (res.available && res.models.length) {
          res.models.forEach((m) => dl.appendChild(el('option', { value: m })));
          modelInp.placeholder = 'pick an installed Ollama model';
          modelHint.textContent = `${res.models.length} Ollama model(s) at ${res.host}. Also set LOCAL_SYNTHESIS_MODEL to a vision model + LOCAL_VISION_CAPABLE=true.`;
        } else {
          modelInp.placeholder = 'type a vision model id (Ollama not reachable)';
          modelHint.textContent = `Ollama not reachable (${res.detail || 'no models'}). Free-text entry — set LOCAL_SYNTHESIS_MODEL to your Ollama vision model.`;
        }
      } else if (providerName === 'together-vision') {
        ['meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo', 'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo']
          .forEach((m) => dl.appendChild(el('option', { value: m })));
        modelInp.placeholder = 'together-vision model id';
        modelHint.textContent = 'Saved as TOGETHER_VISION_MODEL (only this provider uses a dedicated vision-model env var).';
      } else if (providerName === 'anthropic') {
        modelInp.placeholder = 'model id (optional)';
        modelHint.textContent = 'Anthropic is always vision-capable; the model is pinned via DART_CLAUDE_MODEL / global routing, not a separate vision-model var.';
      } else {
        modelInp.placeholder = 'select a vision provider first';
        modelHint.textContent = 'Inherits the DART text provider when left unset.';
      }
    }

    provSel.addEventListener('change', () => {
      ensure('vision').provider = provSel.value || null;
      refreshModelOptions(provSel.value);
    });
    // initial population for the persisted provider
    refreshModelOptions(vCur.provider || '');

    const grid = el('div', { class: 'grid2' }, [
      el('div', { class: 'field' }, [el('label', { text: 'Vision provider' }), provSel]),
      el('div', { class: 'field' }, [el('label', { text: 'Vision model' }), modelInp, dl, modelHint]),
    ]);
    vCard.appendChild(grid);
    view.appendChild(vCard);
  }

  // Two-pass toggle
  const twoPass = el('input', { type: 'checkbox' });
  twoPass.checked = flags.COURSEFORGE_TWO_PASS === true || flags.COURSEFORGE_TWO_PASS === 'true';
  view.appendChild(el('div', { class: 'card' }, [
    el('h2', { text: 'Pipeline flags' }),
    el('label', { class: 'toggle' }, [twoPass, el('span', { text: 'COURSEFORGE_TWO_PASS (two-pass outline → rewrite)' })]),
  ]));

  const saveBtn = el('button', { text: 'Save routing' });
  saveBtn.addEventListener('click', async () => {
    // prune empty task objects
    const clean = {};
    for (const [t, v] of Object.entries(draft)) {
      const kept = {};
      for (const [k, val] of Object.entries(v)) if (val != null && val !== '') kept[k] = val;
      clean[t] = kept;
    }
    saveBtn.disabled = true;
    try {
      await apiJSON('/api/settings', 'PATCH', {
        model_routing: clean,
        flags: { COURSEFORGE_TWO_PASS: twoPass.checked },
      });
      toast('Saved', 'Model routing updated.', 'success');
    } catch (e) { toastErr(e, 'Save routing failed'); }
    finally { saveBtn.disabled = false; }
  });
  view.appendChild(el('div', { class: 'btn-row' }, [saveBtn]));
}

/* =====================================================================
 * TAB: Courses & Topics
 * GET /api/courses ; GET /api/courses/{id}
 * GET/PUT /api/courses/{id}/objectives
 * GET/PATCH /api/courses/{id}/classification
 * ===================================================================== */
async function renderCourses(view) {
  const courses = await api('/api/courses').then((d) => Array.isArray(d) ? d : (d && d.courses) || []);

  clear(view);
  view.appendChild(el('h1', { text: 'Courses & Topics' }));
  view.appendChild(el('p', { class: 'subtitle', text: 'Edit terminal/chapter objectives + subjects/tags. Saving writes a reusable synthesized_objectives.json.' }));

  const layout = el('div', { class: 'row' });
  const listCol = el('div', { class: 'col', style: 'flex:0 0 320px' });
  const editCol = el('div', { class: 'col', style: 'flex:2 1 480px' });
  layout.appendChild(listCol);
  layout.appendChild(editCol);
  view.appendChild(layout);

  function courseId(c) { return c.project_id || c.slug || c.course_name || c.id; }

  if (!courses.length) {
    listCol.appendChild(el('div', { class: 'empty', text: 'No courses found. Run a workflow first.' }));
  } else {
    const tbl = el('table', {}, [
      el('thead', {}, el('tr', {}, [el('th', { text: 'Course' }), el('th', { text: 'Wk' }), el('th', { text: 'TO/CO' })])),
    ]);
    const tbody = el('tbody');
    courses.forEach((c) => {
      const slug = c.slug || c.project_id || '';
      const tr = el('tr', { class: 'clickable' }, [
        el('td', {}, [
          el('div', { text: c.course_name || c.slug || courseId(c) }),
          el('div', { class: 'inline muted', style: 'font-size:11px' }, [
            el('code', { class: 'kv', text: slug }),
            slug ? copyBtn(slug) : null,
          ]),
        ]),
        el('td', { text: c.duration_weeks != null ? String(c.duration_weeks) : '' }),
        el('td', { text: `${c.terminal_count != null ? c.terminal_count : '?'}/${c.chapter_count != null ? c.chapter_count : '?'}` }),
      ]);
      tr.addEventListener('click', () => { $$('tr', tbody).forEach((r) => r.classList.remove('selected')); tr.classList.add('selected'); openCourse(c); });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    listCol.appendChild(tbl);
  }

  editCol.appendChild(el('div', { class: 'empty', text: 'Select a course to edit its objectives and classification.' }));

  async function openCourse(c) {
    const id = courseId(c);
    clear(editCol).appendChild(el('div', { class: 'loading', text: 'Loading objectives…' }));
    let obj = null, cls = null, att = null;
    try {
      obj = await api(`/api/courses/${encodeURIComponent(id)}/objectives`);
    } catch (e) { toastErr(e, 'objectives'); }
    try {
      cls = await api(`/api/courses/${encodeURIComponent(id)}/classification`).catch(() => null);
    } catch (_) { /* classification optional */ }
    try {
      att = await api(`/api/courses/${encodeURIComponent(id)}/attestation`).catch(() => null);
    } catch (_) { /* attestation optional (no LibV2 manifest yet) */ }

    clear(editCol);
    editCol.appendChild(el('h2', { text: c.course_name || id }));

    if (!obj) {
      editCol.appendChild(el('div', { class: 'empty', text: 'No objectives document available for this course.' }));
    } else {
      renderObjectiveEditor(editCol, id, obj);
    }
    renderClassificationEditor(editCol, id, cls || {});
    // Attestation only applies to an archived course (LibV2 manifest). A course
    // with no manifest returns 404 → att stays null and the card is skipped.
    if (att) renderAttestationEditor(editCol, id, att);
  }

  function renderObjectiveEditor(root, id, obj) {
    // Normalize: support Courseforge form (terminal_objectives/chapter_objectives)
    // and LibV2 form (terminal_outcomes/component_objectives).
    const terms = obj.terminal_objectives || obj.terminal_outcomes || [];
    const chaps = obj.chapter_objectives || obj.component_objectives || [];
    const draft = { terminal: terms.map(normLO), chapter: chaps.map(normLO) };

    function normLO(o) {
      if (typeof o === 'string') return { id: '', text: o };
      // Keep every field the server sent (citations, chunk refs, bloom_level,
      // sub-objectives, …) so a text-only edit round-trips the rest verbatim;
      // the editor only rebinds id + text on top of the full object.
      return { ...o, id: o.id || o.lo_id || '', text: o.text || o.statement || o.objective || '' };
    }

    function objSection(title, listKey, idHint) {
      const wrap = el('div');
      wrap.appendChild(el('h3', { text: title }));
      const body = el('div');
      function redraw() {
        clear(body);
        draft[listKey].forEach((lo, i) => {
          const idIn = el('input', { type: 'text', value: lo.id, placeholder: idHint, style: 'flex:0 0 110px' });
          idIn.addEventListener('input', () => { lo.id = idIn.value.trim(); });
          const txtIn = el('input', { type: 'text', value: lo.text, placeholder: 'objective statement' });
          txtIn.addEventListener('input', () => { lo.text = txtIn.value; });
          const del = el('button', { class: 'danger sm', text: '✕', 'aria-label': 'Remove objective' });
          del.addEventListener('click', () => { draft[listKey].splice(i, 1); redraw(); });
          body.appendChild(el('div', { class: 'inline', style: 'margin-bottom:6px' }, [idIn, txtIn, del]));
        });
      }
      redraw();
      const add = el('button', { class: 'ghost sm', text: '+ Add' });
      add.addEventListener('click', () => { draft[listKey].push({ id: '', text: '' }); redraw(); });
      wrap.appendChild(body);
      wrap.appendChild(add);
      return wrap;
    }

    root.appendChild(el('div', { class: 'card' }, [
      objSection('Terminal objectives', 'terminal', 'TO-01'),
      el('hr', { class: 'sep' }),
      objSection('Chapter objectives', 'chapter', 'CO-01'),
    ]));

    // LO id shape mirrors lib/ontology/learning_objectives.py / the JSON-LD
    // schema: ^[A-Z]{2,}-\d{2,}$. A malformed id silently writes a bad reusable
    // objectives JSON downstream, so validate + block save before PUT.
    const LO_ID_RE = /^[A-Z]{2,}-\d{2,}$/;
    const errBox = el('div', { class: 'help', style: `color:var(--bad)`, role: 'alert' });
    const saveBtn = el('button', { text: 'Save objectives' });
    saveBtn.addEventListener('click', async () => {
      const all = [...draft.terminal, ...draft.chapter].filter((o) => o.text.trim());
      const problems = [];
      const seen = new Map();
      for (const o of all) {
        const id_ = (o.id || '').trim();
        if (!id_) continue; // blank id is allowed (backend mints)
        if (!LO_ID_RE.test(id_)) problems.push(`"${id_}" is malformed (expected e.g. TO-01 / CO-12)`);
        seen.set(id_, (seen.get(id_) || 0) + 1);
      }
      for (const [id_, n] of seen) if (n > 1) problems.push(`"${id_}" is duplicated (${n}×)`);
      if (problems.length) {
        errBox.textContent = `Cannot save — ${problems.join('; ')}`;
        toast('Invalid objective IDs', `${problems.length} problem(s) found.`, 'error');
        return;
      }
      errBox.textContent = '';
      // PUT the FULL objects (unknown/extra fields preserved verbatim); id +
      // text are the only editor outputs. Drop the server-derived stale
      // `statement` so the edited text re-maps to statement on save (the
      // service prefers a present statement over text).
      const toPayload = (list) => list.filter((o) => o.text.trim()).map((o) => {
        const { statement, ...rest } = o;
        return { ...rest, id: o.id || null, text: o.text.trim() };
      });
      const payload = {
        terminal_objectives: toPayload(draft.terminal),
        chapter_objectives: toPayload(draft.chapter),
        mint_method: 'user_supplied_objectives_json',
      };
      saveBtn.disabled = true;
      try {
        await apiJSON(`/api/courses/${encodeURIComponent(id)}/objectives`, 'PUT', payload);
        toast('Saved', 'Objectives written (reusable JSON).', 'success');
      } catch (e) { toastErr(e, 'Save objectives failed'); }
      finally { saveBtn.disabled = false; }
    });
    root.appendChild(el('div', { class: 'btn-row' }, [saveBtn, errBox]));
  }

  function renderClassificationEditor(root, id, cls) {
    const c = cls.classification || cls || {};
    const division = el('input', { type: 'text', value: c.division || '' });
    const domain = el('input', { type: 'text', value: c.primary_domain || '' });
    const topics = el('input', { type: 'text', value: (c.topics || []).join(', '), placeholder: 'comma-separated topics' });
    const subtopics = el('input', { type: 'text', value: (c.subtopics || []).join(', '), placeholder: 'comma-separated subtopics' });
    const tags = el('input', { type: 'text', value: (c.tags || []).join(', '), placeholder: 'comma-separated tags' });

    root.appendChild(el('div', { class: 'card' }, [
      el('h3', { text: 'Classification / subjects' }),
      el('div', { class: 'grid2' }, [
        field('Division', division),
        field('Primary domain', domain),
      ]),
      field('Topics', topics),
      field('Subtopics', subtopics),
      field('Tags', tags),
    ]));

    const csv = (s) => s.split(',').map((x) => x.trim()).filter(Boolean);
    const saveBtn = el('button', { text: 'Save classification' });
    saveBtn.addEventListener('click', async () => {
      const patch = {
        division: division.value.trim() || null,
        primary_domain: domain.value.trim() || null,
        topics: csv(topics.value),
        subtopics: csv(subtopics.value),
        tags: csv(tags.value),
      };
      saveBtn.disabled = true;
      try {
        await apiJSON(`/api/courses/${encodeURIComponent(id)}/classification`, 'PATCH', patch);
        toast('Saved', 'Classification updated.', 'success');
      } catch (e) { toastErr(e, 'Save classification failed'); }
      finally { saveBtn.disabled = false; }
    });
    root.appendChild(el('div', { class: 'btn-row' }, [saveBtn]));
  }

  function renderAttestationEditor(root, id, att) {
    const a = att.attestation || att || {};
    // Current status line. aria-live so a save re-render is announced.
    const status = el('div', { class: 'help', 'aria-live': 'polite' });
    function renderStatus(cur) {
      if (cur && cur.reviewed_by) {
        status.textContent = `Reviewed by ${cur.reviewed_by} — scope: ${cur.scope || '?'}`
          + (cur.reviewed_at ? ` (at ${cur.reviewed_at})` : '')
          + (cur.note ? ` — ${cur.note}` : '');
      } else {
        status.textContent = 'Not yet reviewed by a person.';
      }
    }
    renderStatus(a);

    const reviewer = el('input', { type: 'text', value: a.reviewed_by || '', placeholder: 'reviewer name' });
    const scope = el('select', {}, [
      el('option', { value: 'objectives', text: 'Objectives only' }),
      el('option', { value: 'content', text: 'Content only' }),
      el('option', { value: 'full', text: 'Full course' }),
    ]);
    scope.value = a.scope || 'full';
    const note = el('input', { type: 'text', value: a.note || '', placeholder: 'optional note' });

    root.appendChild(el('div', { class: 'card' }, [
      el('h3', { text: 'Human review attestation' }),
      status,
      field('Reviewer name', reviewer),
      field('Scope', scope),
      field('Note', note),
    ]));

    const errBox = el('div', { class: 'help', style: `color:var(--bad)`, role: 'alert' });
    const saveBtn = el('button', { text: 'Attest review' });
    saveBtn.addEventListener('click', async () => {
      const who = reviewer.value.trim();
      if (!who) {
        errBox.textContent = 'Reviewer name is required to attest a review.';
        toast('Missing reviewer', 'Enter a reviewer name.', 'error');
        return;
      }
      errBox.textContent = '';
      // reviewed_at is server-stamped when omitted; scope + reviewer are ours.
      const patch = { reviewed_by: who, scope: scope.value, note: note.value.trim() || null };
      saveBtn.disabled = true;
      try {
        const saved = await apiJSON(`/api/courses/${encodeURIComponent(id)}/attestation`, 'PATCH', patch);
        renderStatus(saved);
        toast('Attested', 'Human review recorded.', 'success');
      } catch (e) { toastErr(e, 'Attest review failed'); }
      finally { saveBtn.disabled = false; }
    });
    root.appendChild(el('div', { class: 'btn-row' }, [saveBtn, errBox]));
  }
}

/* =====================================================================
 * TAB: Retrieval
 * GET  /api/retrieval/courses
 * POST /api/retrieval/query {slug, query, top_k, filters, mode}
 * GET  /api/retrieval/{slug}/adapters
 * POST /api/retrieval/{slug}/infer {model_id, prompt, max_new_tokens}
 * ===================================================================== */
async function renderRetrieval(view) {
  const courses = await api('/api/retrieval/courses').then((d) => Array.isArray(d) ? d : (d && d.courses) || []);

  clear(view);
  view.appendChild(el('h1', { text: 'Retrieval' }));
  view.appendChild(el('p', { class: 'subtitle', text: 'Query a LibV2 course corpus (BM25 / multi / LLM-rerank) and run adapter inference.' }));

  const courseSel = el('select');
  if (!courses.length) courseSel.appendChild(el('option', { value: '', text: 'no indexed courses' }));
  courses.forEach((c) => {
    const slug = c.slug || c;
    const cnt = c.chunk_count != null ? ` (${c.chunk_count} chunks)` : '';
    courseSel.appendChild(el('option', { value: slug, text: `${slug}${cnt}` }));
  });

  // --- Query card ---
  let mode = 'bm25';
  const segBtns = ['bm25', 'multi', 'llm_rerank'].map((m) =>
    el('button', { class: m === mode ? 'active' : '', text: m, 'aria-pressed': String(m === mode), onclick: (e) => { mode = m; $$('button', seg).forEach((b) => { const on = b.textContent === m; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); }); } }));
  const seg = el('div', { class: 'seg', role: 'group', 'aria-label': 'Retrieval mode' }, segBtns);
  const queryIn = el('input', { type: 'search', placeholder: 'enter a retrieval query…' });
  const topK = el('input', { type: 'number', value: '5', min: '1', max: '50', style: 'width:90px' });
  const queryBtn = el('button', { text: 'Query' });
  const resultsWrap = el('div');

  view.appendChild(el('div', { class: 'card' }, [
    el('h2', { text: 'Query' }),
    el('div', { class: 'grid2' }, [
      field('Course', courseSel),
      el('div', { class: 'field' }, [el('label', { text: 'Mode' }), seg]),
    ]),
    field('Query', queryIn),
    el('div', { class: 'inline' }, [
      field('top_k', topK),
      el('div', { class: 'spacer' }),
      queryBtn,
    ]),
    resultsWrap,
  ]));

  async function runQuery() {
    const slug = courseSel.value;
    const q = queryIn.value.trim();
    if (!slug) { toast('No course', 'Pick a course first.', 'error'); return; }
    if (!q) { toast('Empty query', 'Type a query.', 'error'); return; }
    clear(resultsWrap).appendChild(el('div', { class: 'loading', text: 'Retrieving…' }));
    queryBtn.disabled = true;
    try {
      const data = await apiJSON('/api/retrieval/query', 'POST', {
        slug, query: q, top_k: Number(topK.value) || 5, filters: {}, mode,
      });
      const results = Array.isArray(data) ? data : (data.results || []);
      clear(resultsWrap);
      if (data.decomposition) {
        resultsWrap.appendChild(el('div', { class: 'rationale', style: 'margin-bottom:10px', text: `Decomposition: ${JSON.stringify(data.decomposition)}` }));
      }
      if (!results.length) { resultsWrap.appendChild(el('div', { class: 'empty', text: 'No matching chunks.' })); return; }
      results.forEach((r) => {
        const score = r.score != null ? r.score : (r.bm25_score != null ? r.bm25_score : r.rerank_score);
        resultsWrap.appendChild(el('div', { class: 'result' }, [
          el('div', { class: 'meta' }, [
            el('span', { class: 'score', text: score != null ? `score ${Number(score).toFixed(4)}` : 'score —' }),
            r.chunk_id ? el('span', { class: 'inline' }, [el('span', { text: `chunk ${r.chunk_id}` }), copyBtn(String(r.chunk_id), '⧉')]) : null,
            r.source_id ? el('span', { class: 'inline' }, [el('span', { text: `src ${r.source_id}` }), copyBtn(String(r.source_id), '⧉')]) : null,
            r.learning_outcome_refs && r.learning_outcome_refs.length ? el('span', { text: `LO: ${r.learning_outcome_refs.join(', ')}` }) : null,
          ]),
          el('div', { class: 'body', text: r.text || r.body || r.passage || '' }),
          r.rationale ? el('div', { class: 'rationale', text: r.rationale }) : null,
        ]));
      });
    } catch (e) {
      clear(resultsWrap);
      toastErr(e, 'Query failed');
    } finally { queryBtn.disabled = false; }
  }
  queryBtn.addEventListener('click', runQuery);
  queryIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') runQuery(); });

  // --- Adapter inference card ---
  const adapterSel = el('select');
  adapterSel.appendChild(el('option', { value: '', text: 'select a course to load adapters' }));
  const promptIn = el('textarea', { placeholder: 'prompt for the trained adapter…' });
  const maxTok = el('input', { type: 'number', value: '256', min: '1', max: '4096', style: 'width:110px' });
  const inferBtn = el('button', { text: 'Run inference' });
  const inferOut = el('div');

  view.appendChild(el('div', { class: 'card' }, [
    el('h2', { text: 'Adapter inference' }),
    el('div', { class: 'grid2' }, [
      field('Adapter', adapterSel),
      field('max_new_tokens', maxTok),
    ]),
    field('Prompt', promptIn),
    el('div', { class: 'btn-row' }, [inferBtn]),
    inferOut,
  ]));

  async function loadAdapters() {
    const slug = courseSel.value;
    clear(adapterSel).appendChild(el('option', { value: '', text: 'loading…' }));
    if (!slug) { clear(adapterSel).appendChild(el('option', { value: '', text: 'no course' })); return; }
    try {
      const data = await api(`/api/retrieval/${encodeURIComponent(slug)}/adapters`);
      const list = Array.isArray(data) ? data : (data.adapters || []);
      clear(adapterSel);
      if (!list.length) { adapterSel.appendChild(el('option', { value: '', text: 'no adapters for this course' })); return; }
      list.forEach((a) => {
        const idv = a.model_id || a.id || a.slug;
        const base = a.base_model ? ` · ${a.base_model}` : '';
        adapterSel.appendChild(el('option', { value: idv, text: `${idv}${base}` }));
      });
    } catch (e) {
      clear(adapterSel).appendChild(el('option', { value: '', text: 'failed to load adapters' }));
      toastErr(e, 'adapters');
    }
  }
  courseSel.addEventListener('change', loadAdapters);
  loadAdapters();

  inferBtn.addEventListener('click', async () => {
    const slug = courseSel.value;
    const model_id = adapterSel.value;
    const prompt = promptIn.value.trim();
    if (!slug || !model_id) { toast('No adapter', 'Pick a course + adapter.', 'error'); return; }
    if (!prompt) { toast('Empty prompt', 'Type a prompt.', 'error'); return; }
    clear(inferOut).appendChild(el('div', { class: 'loading', text: 'Running adapter…' }));
    inferBtn.disabled = true;
    try {
      const data = await apiJSON(`/api/retrieval/${encodeURIComponent(slug)}/infer`, 'POST', {
        model_id, prompt, max_new_tokens: Number(maxTok.value) || 256,
      });
      clear(inferOut);
      const gen = data.generation || data.text || data.output || (typeof data === 'string' ? data : JSON.stringify(data, null, 2));
      inferOut.appendChild(el('div', { class: 'result' }, [
        el('div', { class: 'meta' }, [el('span', { text: `adapter ${model_id}` }), data.base_model ? el('span', { text: `base ${data.base_model}` }) : null]),
        el('div', { class: 'body', text: gen }),
      ]));
    } catch (e) {
      clear(inferOut);
      // 503 training_deps_missing surfaces here as a typed toast, not a fake answer
      toastErr(e, 'Inference failed');
      if (e instanceof ApiError && e.status === 503) {
        inferOut.appendChild(el('div', { class: 'empty', text: `Inference unavailable: ${e.error} — ${e.detail}` }));
      }
    } finally { inferBtn.disabled = false; }
  });
}

/* =====================================================================
 * TAB: Activity (Claude <-> GUI)
 * NOTE: The spec (§2/§8) defines the event store as state/gui/events.jsonl,
 *   exposed to Claude via MCP gui_post_event/gui_read_events. The spec's
 *   §6 REST contract does not enumerate a dedicated events route, so this
 *   tab targets GET /api/activity/events and POST /api/activity/post.
 *   These belong to the runs/settings worker's surface (they read/write
 *   the same shared_state.events.jsonl). The UI degrades gracefully if the
 *   endpoint 404s — it shows an empty state and keeps the composer usable.
 * ===================================================================== */
async function renderActivity(view) {
  clear(view);
  view.appendChild(el('h1', { text: 'Activity (Claude ↔ GUI)' }));
  view.appendChild(el('p', { class: 'subtitle', text: 'Bidirectional event log between Claude Code sessions and this GUI (state/gui/events.jsonl).' }));

  const feed = el('div', { class: 'card' }, [el('div', { class: 'loading', text: 'Loading events…' })]);
  const msgIn = el('textarea', { placeholder: 'Post a message to Claude (written as a gui-sourced event)…' });
  const postBtn = el('button', { text: 'Post message' });
  const composer = el('div', { class: 'card' }, [
    el('h3', { text: 'Send to Claude' }),
    el('div', { class: 'field' }, [msgIn]),
    el('div', { class: 'btn-row' }, [postBtn]),
  ]);
  view.appendChild(feed);
  view.appendChild(composer);

  let since = 0;
  let endpointDown = false;
  let timer = null;

  function renderEvents(events, replace) {
    if (replace) clear(feed);
    if (replace && (!events || !events.length)) {
      feed.appendChild(el('div', { class: 'empty', text: endpointDown ? 'Events endpoint unavailable (404). Composer still posts when the backend wires it up.' : 'No events yet.' }));
      return;
    }
    events.forEach((evt) => {
      const src = evt.source || 'gui';
      const ts = evt.ts || evt.timestamp || evt.time || '';
      const payload = evt.payload != null ? evt.payload : (evt.message != null ? evt.message : evt);
      feed.appendChild(el('div', { class: `event ${src}` }, [
        el('div', { class: 'src', text: src }),
        el('div', { class: 'ebody' }, [
          el('div', { class: 'ekind', text: evt.kind || evt.type || 'message' }),
          el('div', {}, typeof payload === 'string'
            ? document.createTextNode(payload)
            : el('pre', { text: JSON.stringify(payload, null, 2) })),
        ]),
        el('div', { class: 'ets', text: typeof ts === 'number' ? new Date(ts * (ts < 1e12 ? 1000 : 1)).toLocaleTimeString() : String(ts).slice(11, 19) }),
      ]));
      if (evt.seq != null) since = Math.max(since, evt.seq + 1);
    });
  }

  async function poll(replace) {
    try {
      const data = await api(`/api/activity/events?since=${since}`);
      endpointDown = false;
      const events = Array.isArray(data) ? data : (data.events || []);
      if (replace) since = 0;
      renderEvents(events, replace);
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        endpointDown = true;
        if (replace) renderEvents([], true);
        // stop hammering a missing endpoint
        if (timer) { clearInterval(timer); timer = null; }
      } else if (replace) {
        clear(feed).appendChild(el('div', { class: 'empty', text: `Failed to load events: ${e instanceof ApiError ? e.error : e}` }));
        toastErr(e, 'events');
      }
    }
  }

  postBtn.addEventListener('click', async () => {
    const text = msgIn.value.trim();
    if (!text) return;
    postBtn.disabled = true;
    try {
      await apiJSON('/api/activity/post', 'POST', { source: 'gui', kind: 'human_message', payload: { text } });
      msgIn.value = '';
      toast('Posted', 'Message added to the event log.', 'success');
      since = 0; await poll(true);
      if (!timer) timer = setInterval(() => poll(false), 4000);
    } catch (e) {
      toastErr(e, 'Post failed');
    } finally { postBtn.disabled = false; }
  });

  await poll(true);
  if (!endpointDown) timer = setInterval(() => poll(false), 4000);
  return () => { if (timer) clearInterval(timer); };
}

/* =====================================================================
 * Boot
 * ===================================================================== */
function boot() {
  if (!location.hash) location.hash = '#/' + DEFAULT_TAB;
  route();
  pingConnection();
  setInterval(pingConnection, 15000);
}
document.addEventListener('DOMContentLoaded', boot);
