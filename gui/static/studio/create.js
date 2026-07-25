/* Ed4All Studio — Create course wizard (Marketable-v1 C3).
 *
 * A non-developer uploads a textbook PDF, configures minimally, launches a
 * course build, and watches friendly phase progress. Three steps:
 *
 *   1. Upload    — drag-drop / file-picker PDF(s) (POST /api/uploads), with
 *                  client-side type + size validation.
 *   2. Configure — course name (slug-rule hint only), optional weeks, a
 *                  provider/model summary read from settings (link to the
 *                  Studio settings page), advanced options collapsed.
 *   3. Launch    — POST /api/runs, then a phase-checklist progress view driven
 *                  by the existing WS (/ws/runs/{run_id}); friendly phase names
 *                  with pending/running/done/failed states, elapsed time, and a
 *                  final "Open course" link (success) or actionable error +
 *                  run-log link (failure).
 *
 * Wizard state survives refresh: launching navigates to #/create/<run_id>, so a
 * reload re-attaches to the running run via the run registry + WS.
 *
 * Imports the shared toolkit so it never re-implements fetch / DOM helpers, and
 * receives the shell helpers (status live-region, crumbs, busy flag) from
 * studio.js so a11y status announcements land in the one polite live region.
 */

import { api, apiJSON } from '/shared/api.js';
import { el, clear, uid } from '/shared/dom.js';
import { toast, toastErr } from '/shared/toast.js';
import { runProgressConsole } from '/shared/components/run-progress.js';
import { stageRail } from '/shared/components/stage-rail.js';
import { emptyState } from '/shared/components/empty-state.js';
import { statusPill } from '/shared/components/pill.js';
import { mountSeatMonitor } from '/studio/seats.js';
import { fmtAge } from '/studio/seats.js';

const WORKFLOW = 'textbook_to_course';
const ACCEPT_EXT = ['.pdf'];
const MAX_FILE_BYTES = 200 * 1024 * 1024; // 200 MB client-side guard
// Slug rule mirrors the course_name validation surface: letters, digits,
// underscore, hyphen (PHYS_101, BIO-201). Client-side HINT only — the server is
// authoritative.
const SLUG_HINT_RE = /^[A-Za-z0-9_-]{2,}$/;

/* ------------------------------------------------------------------ entry */

/**
 * Route handler. `segments[0]` (when present) is a run_id to re-attach to.
 * Returns a teardown that closes the live WS.
 */
export async function renderCreate(shell, segments) {
  const runId = segments && segments[0] ? segments[0] : null;
  const sub = segments && segments[1] ? segments[1] : null;
  if (runId && sub === 'log') return renderRunLog(shell, runId);
  if (runId) return renderProgress(shell, runId);
  return renderWizard(shell);
}

/* ============================================================= the wizard */

async function renderWizard(shell) {
  shell.crumbs([{ text: 'Library', href: '#/library' }, { text: 'Create course' }]);
  const v = clear(shell.view());
  shell.setBusy(false);
  shell.setStatus('Create a new course. Step 1 of 3: upload your textbook.');

  // Wizard model held in a closure; steps mutate it.
  const model = { step: 1, files: [], uploadId: null, courseName: '', weeks: '', pauseForReview: false, advanced: {} };

  v.appendChild(el('h1', { text: 'Create a course' }));
  v.appendChild(stepsIndicator(model.step));

  const panel = el('div', { class: 'wizard-panel' });
  v.appendChild(panel);

  // Pre-fetch the provider/model summary (best-effort) so step 2 renders fast.
  let settingsSummary = null;
  try {
    settingsSummary = await fetchSettingsSummary();
  } catch (_) { /* surfaced in step 2 as a muted fallback */ }

  function goto(step) {
    model.step = step;
    v.replaceChild(stepsIndicator(step), v.querySelector('.wizard-steps'));
    renderStep();
  }

  function renderStep() {
    clear(panel);
    if (model.step === 1) renderUploadStep(shell, panel, model, goto);
    else if (model.step === 2) renderConfigureStep(shell, panel, model, settingsSummary, goto);
    else renderLaunchStep(shell, panel, model);
  }

  renderStep();
}

/** The 3-step progress structure (an ordered list with aria-current). */
function stepsIndicator(active) {
  const labels = ['Upload', 'Configure', 'Launch'];
  const ol = el('ol', { class: 'wizard-steps', 'aria-label': 'Create course steps' });
  labels.forEach((label, i) => {
    const n = i + 1;
    const state = n < active ? 'done' : n === active ? 'current' : 'upcoming';
    const li = el('li', {
      class: `wizard-step is-${state}`,
      'aria-current': n === active ? 'step' : null,
    }, [
      el('span', { class: 'wizard-step-num', 'aria-hidden': 'true', text: String(n) }),
      el('span', { class: 'wizard-step-label', text: label }),
      el('span', { class: 'visually-hidden', text: n < active ? ' (completed)' : n === active ? ' (current step)' : ' (upcoming)' }),
    ]);
    ol.appendChild(li);
  });
  return ol;
}

/* -------------------------------------------------------- step 1: upload */

function renderUploadStep(shell, panel, model, goto) {
  panel.appendChild(el('h2', { text: 'Step 1: Upload your textbook' }));
  panel.appendChild(el('p', { class: 'muted', text: 'Add one or more PDF files. We convert them to an accessible course.' }));

  const inputId = uid('file');
  const drop = el('div', {
    class: 'dropzone',
    tabindex: '0',
    role: 'button',
    'aria-label': 'Choose PDF files or drop them here',
  }, [
    el('p', { text: 'Drag PDF files here, or' }),
    el('label', { class: 'btn', for: inputId, text: 'Choose files' }),
  ]);
  const fileInput = el('input', {
    id: inputId,
    type: 'file',
    accept: '.pdf,application/pdf',
    multiple: true,
    class: 'visually-hidden',
  });
  drop.appendChild(fileInput);
  panel.appendChild(drop);

  const errLine = el('p', { class: 'error', role: 'alert', hidden: true });
  panel.appendChild(errLine);

  const fileList = el('ul', { class: 'file-list', 'aria-label': 'Selected files' });
  panel.appendChild(fileList);

  const uploadStatus = el('p', { class: 'muted', 'aria-live': 'polite' });
  panel.appendChild(uploadStatus);

  const nextBtn = el('button', { type: 'button', class: 'btn primary', text: 'Next: Configure', disabled: true });
  const nav = el('div', { class: 'wizard-nav' }, [
    el('a', { class: 'btn', href: '#/library', text: 'Cancel' }),
    nextBtn,
  ]);
  panel.appendChild(nav);

  function showErr(msg) {
    errLine.textContent = msg;
    errLine.hidden = !msg;
    if (msg) shell.setStatus(msg);
  }

  function renderFiles() {
    clear(fileList);
    model.files.forEach((f, idx) => {
      fileList.appendChild(el('li', {}, [
        el('span', { class: 'file-name', text: f.name }),
        el('span', { class: 'file-size', text: ` (${fmtBytes(f.size)})` }),
        el('button', {
          type: 'button',
          class: 'btn link-btn',
          'aria-label': `Remove ${f.name}`,
          text: 'Remove',
          onclick: () => { model.files.splice(idx, 1); model.uploadId = null; renderFiles(); syncNext(); },
        }),
      ]));
    });
  }

  function syncNext() {
    nextBtn.disabled = model.files.length === 0;
  }

  function acceptFiles(fileArr) {
    showErr('');
    const accepted = [];
    for (const f of fileArr) {
      const lower = (f.name || '').toLowerCase();
      if (!ACCEPT_EXT.some((ext) => lower.endsWith(ext))) {
        showErr(`${f.name}: only PDF files are supported.`);
        continue;
      }
      if (f.size > MAX_FILE_BYTES) {
        showErr(`${f.name}: file is larger than ${fmtBytes(MAX_FILE_BYTES)}.`);
        continue;
      }
      accepted.push(f);
    }
    if (accepted.length) {
      model.files = model.files.concat(accepted);
      model.uploadId = null; // a new selection invalidates a prior upload
      renderFiles();
      syncNext();
      shell.setStatus(`${model.files.length} file(s) selected.`);
    }
  }

  fileInput.addEventListener('change', () => acceptFiles(Array.from(fileInput.files || [])));
  drop.addEventListener('click', (e) => { if (e.target === drop || e.target.tagName === 'P') fileInput.click(); });
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('is-over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('is-over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
    acceptFiles(Array.from(e.dataTransfer?.files || []));
  });

  nextBtn.addEventListener('click', async () => {
    if (!model.files.length) return;
    // Upload now (so step 2 already has an upload_id); re-uploads are skipped.
    if (!model.uploadId) {
      nextBtn.disabled = true;
      uploadStatus.textContent = 'Uploading…';
      shell.setStatus('Uploading your files.');
      try {
        const fd = new FormData();
        model.files.forEach((f) => fd.append('files', f, f.name));
        const resp = await api('/api/uploads', { method: 'POST', body: fd });
        model.uploadId = resp.upload_id;
        uploadStatus.textContent = `Uploaded ${resp.files.length} file(s).`;
      } catch (e) {
        uploadStatus.textContent = '';
        nextBtn.disabled = false;
        showErr(shell.errText(e));
        toastErr(e, 'Upload failed');
        return;
      }
    }
    goto(2);
  });

  renderFiles();
  syncNext();
}

/* ----------------------------------------------------- step 2: configure */

function renderConfigureStep(shell, panel, model, summary, goto) {
  shell.setStatus('Step 2 of 3: configure your course.');
  panel.appendChild(el('h2', { text: 'Step 2: Configure' }));

  const form = el('form', { class: 'wizard-form', novalidate: true });

  // Course name (required, slug-hint validated client-side).
  const nameId = uid('cname');
  const nameHintId = uid('cname-h');
  const nameField = el('div', { class: 'field' }, [
    el('label', { for: nameId, text: 'Course name' }),
    el('input', {
      id: nameId,
      type: 'text',
      value: model.courseName,
      required: true,
      'aria-describedby': nameHintId,
      autocomplete: 'off',
    }),
    el('p', { id: nameHintId, class: 'field-hint', text: 'Letters, numbers, “_” and “-” only (e.g. PHYS_101). At least 2 characters.' }),
    el('p', { class: 'error', role: 'alert', hidden: true }),
  ]);
  form.appendChild(nameField);
  const nameInput = nameField.querySelector('input');
  const nameErr = nameField.querySelector('.error');

  // Weeks (optional).
  const weeksId = uid('weeks');
  const weeksHintId = uid('weeks-h');
  form.appendChild(el('div', { class: 'field' }, [
    el('label', { for: weeksId, text: 'Course length in weeks (optional)' }),
    el('input', { id: weeksId, type: 'number', min: '1', max: '52', value: model.weeks, 'aria-describedby': weeksHintId, inputmode: 'numeric' }),
    el('p', { id: weeksHintId, class: 'field-hint', text: 'Leave blank to size the course automatically from the textbook.' }),
  ]));
  const weeksInput = form.querySelector(`#${CSS.escape(weeksId)}`);

  // AI-tier flow tree read from settings, with a link to settings per step.
  form.appendChild(providerFlowTree(summary, shell));

  // Pause-for-objectives-review checkpoint (I1 review flow). A first-class
  // configure option (not buried in Advanced): when checked, the build halts
  // after planning the learning objectives so the author can review/edit them
  // before the rest of the course is generated, then resume from the build view.
  form.appendChild(reviewCheckpointField(model));

  // Advanced (collapsed by default).
  form.appendChild(advancedSection(model));

  // Nav.
  const backBtn = el('button', { type: 'button', class: 'btn', text: 'Back' });
  const nextBtn = el('button', { type: 'submit', class: 'btn primary', text: 'Launch build' });
  form.appendChild(el('div', { class: 'wizard-nav' }, [backBtn, nextBtn]));

  backBtn.addEventListener('click', () => goto(1));

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    model.courseName = nameInput.value.trim();
    model.weeks = weeksInput.value.trim();
    nameErr.hidden = true;
    if (!SLUG_HINT_RE.test(model.courseName)) {
      nameErr.textContent = 'Enter a valid course name (letters, numbers, “_”, “-”; at least 2 characters).';
      nameErr.hidden = false;
      nameInput.setAttribute('aria-invalid', 'true');
      nameInput.focus();
      shell.setStatus('Course name is not valid.');
      return;
    }
    nameInput.removeAttribute('aria-invalid');
    goto(3);
  });

  panel.appendChild(form);
  nameInput.focus();
}

/**
 * The AI-tier flow tree: a small pipeline `PDF → Outline → Validate/Rewrite (if
 * two-pass) → Assessments` showing the active provider/model per step (read from
 * GET /api/settings/studio), each step linking to the Studio settings page.
 *
 * Read-only display — it does not change launch behaviour. The Rewrite (and the
 * preceding Validate) tier greys out unless two-pass is enabled (carrying a
 * text "(two-pass only)" note, NOT colour alone). The PDF source node and the
 * Assessments node are always active.
 *
 * A11y: a semantic ordered list (<ol>), one <li> per step; the active
 * provider/model per step is plain text; each settings deep-link is a real <a>
 * (≥24×24px via the .btn/.flow-link rules; :focus-visible from studio.css).
 */
function providerFlowTree(summary, shell) {
  const routing = (summary && summary.routing) || {};
  const twoPass = !!(summary && summary.twoPass);

  // Resolve a per-task {provider, model}, falling through to global routing so a
  // step that inherits the global provider still shows it (never blank).
  const global = routing.global || {};
  const fallbackProvider = (summary && summary.provider) || global.provider || 'local';
  function tier(task) {
    const t = routing[task] || {};
    const provider = (t.provider && String(t.provider).trim()) || fallbackProvider;
    const model = (t.model && String(t.model).trim()) || null;
    return { provider, model };
  }

  const box = el('div', { class: 'summary-box flow-tree-box' });
  box.appendChild(el('h3', { id: 'flow-tree-h', text: 'AI pipeline' }));
  box.appendChild(el('p', { class: 'muted', text: 'How your course is built, and which AI model runs each step. Change any step in settings.' }));

  // Each node: {key, label, provider/model text, active, note}.
  const outline = tier('courseforge_outline');
  const rewrite = tier('courseforge_rewrite');

  const steps = [
    {
      label: 'PDF',
      sub: 'Your uploaded textbook',
      active: true,
    },
    {
      label: 'Outline',
      sub: providerText(outline),
      active: true,
    },
    {
      label: 'Validate & Rewrite',
      sub: twoPass ? providerText(rewrite) : 'Skipped',
      active: twoPass,
      note: twoPass ? null : '(two-pass only)',
    },
    {
      label: 'Assessments',
      sub: providerText(tier('courseplanner')),
      active: true,
    },
  ];

  const ol = el('ol', { class: 'flow-tree', 'aria-labelledby': 'flow-tree-h' });
  steps.forEach((s, i) => {
    const li = el('li', {
      class: `flow-step${s.active ? '' : ' is-inactive'}`,
    });
    // A right-pointing connector between steps (decorative).
    if (i > 0) li.appendChild(el('span', { class: 'flow-arrow', 'aria-hidden': 'true', text: '→ ' }));
    const body = el('span', { class: 'flow-step-body' }, [
      el('span', { class: 'flow-step-label', text: s.label }),
      el('span', { class: 'flow-step-sub', text: s.sub }),
      // Non-colour-only inactive marker: a real text note, not just greying.
      s.note ? el('span', { class: 'flow-step-note', text: ' ' + s.note }) : null,
    ]);
    li.appendChild(body);
    ol.appendChild(li);
  });
  box.appendChild(ol);

  box.appendChild(el('p', { class: 'flow-tree-link' }, [
    el('a', { class: 'flow-link', href: '#/settings', text: 'Change AI model settings' }),
  ]));
  return box;
}

/** Render a step's active provider (+ model when pinned) as plain text. */
function providerText(tier) {
  if (!tier || !tier.provider) return 'Default settings';
  return tier.model ? `${tier.provider} · ${tier.model}` : tier.provider;
}

/**
 * The "Pause for objectives review" checkpoint field (I1 review flow).
 *
 * A labelled checkbox + a describing hint. When checked, ``renderLaunchStep``
 * launches with ``stop_after=course_planning`` so the build halts after the
 * learning objectives are planned; the progress view then offers an
 * objectives-editor link + a "Resume build" button (a plain resume that picks
 * up the edited objectives file in place).
 */
function reviewCheckpointField(model) {
  const id = uid('pause-review');
  const hintId = uid('pause-review-h');
  return el('div', { class: 'field check review-check' }, [
    el('input', {
      id,
      type: 'checkbox',
      checked: !!model.pauseForReview,
      'aria-describedby': hintId,
      onchange: (e) => { model.pauseForReview = e.target.checked; },
    }),
    el('label', { for: id, text: 'Pause for objectives review' }),
    el('p', {
      id: hintId,
      class: 'field-hint',
      text: 'Stop the build after planning the learning objectives so you can review and edit them before the rest of the course is generated. You can resume from the build screen.',
    }),
  ]);
}

function advancedSection(model) {
  const details = el('details', { class: 'advanced' });
  details.appendChild(el('summary', { text: 'Advanced options' }));
  const skipId = uid('skip-assess');
  const skipTrainId = uid('skip-train');
  const wrap = el('div', { class: 'advanced-body' }, [
    el('div', { class: 'field check' }, [
      el('input', {
        id: skipId,
        type: 'checkbox',
        checked: !!model.advanced.no_assessments,
        onchange: (e) => { model.advanced.no_assessments = e.target.checked; },
      }),
      el('label', { for: skipId, text: 'Skip assessment generation' }),
    ]),
    el('div', { class: 'field check' }, [
      el('input', {
        id: skipTrainId,
        type: 'checkbox',
        checked: !!model.advanced.skip_training,
        onchange: (e) => { model.advanced.skip_training = e.target.checked; },
      }),
      el('label', { for: skipTrainId, text: 'Skip training-data synthesis' }),
    ]),
  ]);
  details.appendChild(wrap);
  return details;
}

/* -------------------------------------------------------- step 3: launch */

async function renderLaunchStep(shell, panel, model) {
  shell.setStatus('Step 3 of 3: launching your course build.');
  panel.appendChild(el('h2', { text: 'Step 3: Launch' }));
  const busy = el('p', { class: 'loading', 'aria-live': 'polite', text: 'Starting your course build…' });
  panel.appendChild(busy);

  const options = {};
  if (model.advanced.no_assessments) options.no_assessments = true;
  if (model.advanced.skip_training) options.skip_training = true;
  // I1 review flow: halt the build after course_planning so the author can
  // review + edit the learning objectives before the rest is generated.
  if (model.pauseForReview) options.stop_after = 'course_planning';

  const body = {
    workflow: WORKFLOW,
    course_name: model.courseName,
    corpus: model.uploadId,
    options,
  };
  if (model.weeks) body.weeks = Number(model.weeks);

  let resp;
  try {
    resp = await apiJSON('/api/runs', 'POST', body);
  } catch (e) {
    busy.remove();
    panel.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    panel.appendChild(el('div', { class: 'wizard-nav' }, [
      el('a', { class: 'btn', href: '#/create', text: 'Back to start' }),
    ]));
    shell.setStatus('Could not start the course build.');
    toastErr(e, 'Launch failed');
    return;
  }

  toast('Course build started.', '', 'success');
  // Navigate to the run-scoped route so a refresh re-attaches to this run.
  location.hash = `#/create/${encodeURIComponent(resp.run_id)}`;
}

/* =================================================== progress (re-attach) */

/**
 * Build the stage-tracker rail + its 3s poll loop over
 * GET /api/runs/{id}/progress. Shared by BOTH progress paths — the full
 * GUI-launched page and the degraded CLI-run page (renderCliRunProgress) —
 * so the poll / terminal-stop logic exists exactly once instead of being
 * duplicated per path. The rail itself is PURE PRESENTATION (see
 * gui/services/progress_service.py for the payload); this helper owns the
 * loop, and a terminal status stops it so a finished / failed run renders
 * statically (no pulse, no further polling). Callers append `.el` then
 * `start()`; `stop()` is the route-change teardown; `pollOnce()` takes one
 * last snapshot after a terminal WS frame; `apply()` seeds the rail from a
 * snapshot already in hand (no double-fetch on first paint).
 */
/** Compact pages/hr figure for the band: 1840 → "1.8k", 732 → "732". */
function fmtPagesRate(n) {
  if (!Number.isFinite(n)) return '';
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(Math.round(n));
}

/**
 * Page-metered progress stat for the live stats band (a create.js-owned
 * add-on over the shared stage-rail band). The SemantiK GLM-OCR conversion
 * lane meters HONEST ZERO tokens — its usage rows carry a positive-int
 * `pages` field instead — so during conversion the band would otherwise show
 * an LLM-call count with no token metrics and no sign of page progress.
 * When the payload carries stats.pages > 0 this inserts a "pages" row
 * (e.g. "236 pages · ~2.3k pages/hr" — same div.stage-stat > dt/dd markup +
 * .tabular-nums styling as the band's own rows; plain text, never
 * color-only) right after the "LLM calls" row. And when the run so far is
 * ALL page-metered (calls > 0 with zero prompt+completion tokens), the
 * token-shaped rows (the "tok/s" dash and "0 in · 0 out") are dropped so the
 * pages stat carries the band instead of an empty token area. Token
 * rendering is untouched the moment any real tokens exist. The rate is
 * omitted (count only) when the server sent pages_per_hr: null — nothing
 * fabricated. Re-applied after every rail.update() (which rebuilds the
 * band), so the injection is naturally idempotent per poll.
 */
function applyPagesStat(rootEl, payload) {
  const s = (payload && payload.stats) || {};
  // The band's own <dl> is a DIRECT child of the rail section; the detail
  // disclosure nests another dl.stage-stats, so scope to direct children.
  const dl = rootEl.querySelector(':scope > dl.stage-stats');
  if (!dl) return;
  const pages = s.pages;
  if (!Number.isFinite(pages) || pages <= 0) return;
  const byTerm = new Map();
  Array.from(dl.querySelectorAll(':scope > .stage-stat')).forEach((r) => {
    const dt = r.querySelector('dt');
    if (dt) byTerm.set(dt.textContent, r);
  });
  // The `pages` field is emitted ONLY by the SemantiK GLM-OCR conversion lane
  // (sdk_client.py: extra={"pages": N}); no other surface reports it. So the
  // stat is always OCR page throughput — label it explicitly.
  const rate = s.pages_per_hr;
  const text = Number.isFinite(rate)
    ? `${pages} · ~${fmtPagesRate(rate)} OCR pages/hr`
    : `${pages} OCR pages`;
  const row = el('div', { class: 'stage-stat' }, [
    el('dt', { text: 'OCR pages' }),
    el('dd', { class: 'tabular-nums', text }),
  ]);
  const callsRow = byTerm.get('LLM calls');
  if (callsRow && callsRow.after) callsRow.after(row);
  else dl.appendChild(row);
  const zeroTokens = (Number(s.prompt_tokens) || 0) === 0
    && (Number(s.completion_tokens) || 0) === 0;
  if (zeroTokens && Number(s.calls) > 0) {
    const tokS = byTerm.get('tok/s');
    const tokens = byTerm.get('tokens');
    if (tokS) tokS.remove();
    if (tokens) tokens.remove();
  }
}

/** Live age of a waiting seat — prefers the ISO marker (ticks between polls),
 * mirroring seats.js seatAge, else the server's elapsed_ms. */
function seatWaitAge(seat) {
  if (seat && seat.since) {
    const t = Date.parse(seat.since);
    if (!Number.isNaN(t)) return fmtAge(Date.now() - t);
  }
  if (Number.isFinite(seat && seat.elapsed_ms)) return fmtAge(seat.elapsed_ms);
  return '';
}

/**
 * Seat-swap / cold-start annotation for the stage rail (a create.js-owned
 * add-on over the shared rail, sibling of `applyPagesStat`). The stage tracker
 * attributes only a phase's OWN compute time, so a long vLLM seat swap at a
 * phase boundary (the schedule tears down the conversion seats and cold-starts
 * the next — up to ~10 min: container up, /v1/models silent) renders as the
 * phase hanging with no progress. When the payload carries
 * `stats.seat_activity` this inserts a distinct line directly under the phase
 * nodes (above the stats band, subordinate to the rail) that reads as "the
 * pipeline is waiting for a model seat to come up", NOT as phase compute — and
 * never touches any phase's wall-clock.
 *
 * Two honesty grades, text-not-color-only (WCAG 1.4.1):
 *   - PATIENT (loading): "Swapping model seats … starting <seat> · ~5m elapsed"
 *     — an expected ~9-min cold start, `.stage-seatwait` (accent rule).
 *   - ALARMING (down / unregistered / unknown — the mismatch set): "Waiting for
 *     a model seat that is not coming up …", `.stage-seatwait-concern` (fail
 *     rule) + a ⚠ glyph.
 * Absent field (every needed seat live) → the line is removed. Re-applied after
 * every rail.update(), so it is idempotent per poll.
 */
function applySeatActivity(rootEl, payload) {
  const act = (payload && payload.stats && payload.stats.seat_activity) || null;
  const seats = act && Array.isArray(act.seats) ? act.seats : [];
  let banner = rootEl.querySelector(':scope > .stage-seatwait');
  if (!act || !seats.length) {
    if (banner) banner.remove();
    return;
  }
  const concern = !!act.concern;
  const phase = String(act.phase || 'this phase').replace(/_/g, ' ');
  const parts = seats.map((s) => {
    const name = String(s.seat || '');
    const age = seatWaitAge(s);
    if (s.state === 'loading') return `starting ${name}${age ? ` · ~${age} elapsed` : ''}`;
    if (s.state === 'down') return `${name} is not running${age ? ` · ${age}` : ''}`;
    if (s.state === 'unregistered') return `${name} is not registered`;
    return `${name} is ${s.state || 'unknown'}${age ? ` · ${age}` : ''}`;
  });
  const lead = concern
    ? `Phase ${phase} is waiting for a model seat that is not coming up — `
    : `Swapping model seats for phase ${phase} — `;
  const tail = concern
    ? '. A dispatch to a stopped seat fails the build — this is the pipeline waiting on a seat, not the phase hanging.'
    : ' (a cold start can take up to ~10 minutes; the phase is waiting on the seat, not stuck).';
  const glyph = concern ? '⚠ ' : '⏳ ';
  if (!banner) {
    banner = el('p', { class: 'stage-seatwait' });
    const host = rootEl.querySelector(':scope > .stage-rail-host');
    if (host && host.after) host.after(banner);
    else rootEl.insertBefore(banner, rootEl.firstChild);
  }
  banner.className = concern ? 'stage-seatwait stage-seatwait-concern' : 'stage-seatwait';
  clear(banner);
  banner.appendChild(el('span', { class: 'stage-seatwait-glyph', 'aria-hidden': 'true', text: glyph }));
  banner.appendChild(el('span', { text: `${lead}${parts.join('; ')}${tail}` }));
}

// effective_status / raw-status vocabularies that mean a run is over.
const RAIL_TERMINAL_EFF = ['completed', 'failed', 'cancelled', 'canceled', 'interrupted', 'paused', 'timeout', 'error', 'incomplete'];
const RAIL_TERMINAL_RAW = ['completed', 'failed', 'cancelled', 'canceled', 'interrupted', 'paused', 'timeout', 'error'];

/**
 * Mount the stage-tracker rail + its 3s poll loop over GET
 * /api/runs/{id}/progress.
 *
 * `opts.pipeline` turns on the MERGED-pipeline mode: the single rail shows the
 * build run's phases followed by the follow-on LoRA-training (Stage-B) phases
 * as ONE trailing "training" group — never a separate segregated band. The
 * merge is driven each poll:
 *   - the pipeline CHAIN (GET /api/runs/{id}/pipeline) is refreshed on a slow
 *     ~18s cadence (stage EXISTENCE changes rarely); a 404 / one-stage chain /
 *     any error leaves the plain single build rail untouched;
 *   - build phases come from the build run's /progress and always LEAD;
 *   - training phases come from the training run's /progress (when it is a real
 *     run) with every phase's `group` FORCED to `"training"`, else are
 *     synthesized as dim PENDING nodes from the chain's planned_phases;
 *   - the ACTIVE run (training once it is under way, else the build run) is the
 *     BASE payload the stats band / current-phase pulse / status / course
 *     identity / output-tail derive from — merged phases are laid onto it.
 * Exactly one phase across the merged list is ever `current` (the non-active
 * run is terminal), so band + pulse follow the active run with no conflict.
 *
 * The rail latches terminal (stops polling) only when the WHOLE pipeline is
 * done — the active run is terminal AND no downstream training is still pending
 * after a SUCCESSFUL build — so a build that finished with training still
 * planned keeps polling and the trailing pending nodes flip to a live group
 * with no page reload.
 *
 * `onState(eff, payload, activeRunId)` fires each poll so the host can re-sync
 * the heading / health strip (which follows `activeRunId`) / seat strip.
 */
function mountStageRail(runId, ariaLabel, onState, opts = {}) {
  const rail = stageRail({ ariaLabel });
  const RAIL_POLL_MS = 3000;
  const CHAIN_REFRESH_MS = 18000; // stage EXISTENCE changes rarely; the 3s poll owns live phase state
  const pipeline = !!opts.pipeline;
  let railTimer = null;
  let terminal = false;
  let chain = null;        // last /pipeline chain (pipeline mode only)
  let chainAt = 0;         // last successful chain fetch (ms)
  let activeRunId = runId; // run whose /progress is the BASE payload

  function stop() {
    if (railTimer != null) { clearInterval(railTimer); railTimer = null; }
  }

  // Paint one payload onto the rail + its create.js-owned add-ons and notify
  // the host. Returns {eff, raw}; does NOT latch terminal — the caller owns
  // that so pipeline mode can keep polling for a follow-on training stage after
  // the build run itself is terminal. `effective_status`
  // (gui/services/liveness.py) reflects real process liveness.
  function render(p, active) {
    rail.update(p);
    applyPagesStat(rail.el, p);
    applySeatActivity(rail.el, p);
    const raw = String((p && p.status) || '').toLowerCase();
    const eff = String((p && p.effective_status) || raw).toLowerCase();
    if (typeof onState === 'function') {
      try { onState(eff, p, active || runId); } catch (_) { /* presentation-only */ }
    }
    return { eff, raw };
  }

  // Seed/apply path for callers holding a snapshot (renderCliRunProgress). In
  // plain mode a terminal snapshot latches exactly as before; in pipeline mode
  // it never latches (pollOnce owns latching), so a terminal build snapshot
  // doesn't freeze the rail before training is merged in.
  function apply(p) {
    const { eff, raw } = render(p, runId);
    if (!pipeline && (RAIL_TERMINAL_EFF.includes(eff) || RAIL_TERMINAL_RAW.includes(raw))) {
      terminal = true;
      stop();
    }
  }

  async function pollOncePlain() {
    try {
      apply(await api(`/api/runs/${encodeURIComponent(runId)}/progress`));
    } catch (_) { /* best-effort — the page's primary surface stays usable */ }
    if (terminal) return;
    // "Live output" panel: the current phase's real sidecar tail (bounded
    // display rows). Best-effort on the same cadence.
    try {
      rail.updateTail(await api(`/api/runs/${encodeURIComponent(runId)}/output-tail`));
    } catch (_) { /* best-effort — the panel simply doesn't advance */ }
  }

  async function pollOncePipeline() {
    // Slow-cadence chain refresh — best-effort. A failure keeps the last chain
    // (or null → today's plain single build rail), never blocks the build poll.
    if (Date.now() - chainAt >= CHAIN_REFRESH_MS) {
      try {
        chain = await api(`/api/runs/${encodeURIComponent(runId)}/pipeline`);
        chainAt = Date.now();
      } catch (_) { /* keep prior chain; a 404 / one-stage chain → build-only */ }
    }
    const training = pipelineTrainingStage(chain);
    const trainRunId = (training && training.present && training.run_id != null)
      ? String(training.run_id) : null;

    // The build run's phases always LEAD the rail.
    let buildP = null;
    try {
      buildP = await api(`/api/runs/${encodeURIComponent(runId)}/progress`);
    } catch (_) { /* best-effort */ }

    // The training run's phases (+ base, when it is active) come from its own
    // /progress — fetched only once it is a real run.
    let trainP = null;
    if (trainRunId) {
      try {
        trainP = await api(`/api/runs/${encodeURIComponent(trainRunId)}/progress`);
      } catch (_) { /* not observable yet → treat as planned pending nodes */ }
    }

    // ACTIVE run = the training run once it is under way (build finished), else
    // the build run. Its /progress is the BASE (stats band, current highlight,
    // status, course identity, output-tail).
    const trainActive = !!(trainRunId && trainP);
    const base = trainActive ? trainP : buildP;
    activeRunId = trainActive ? trainRunId : runId;
    if (!base) return; // nothing renderable this tick (both fetches failed)

    const buildPhases = Array.isArray(buildP && buildP.phases) ? buildP.phases : [];
    // Force every training phase into ONE trailing "training" group so they
    // never interleave into the build's generation/validation sections.
    const trainingPhases = trainP
      ? (Array.isArray(trainP.phases) ? trainP.phases : []).map((p) => ({ ...p, group: 'training' }))
      : plannedTrainingPhases(training);

    const merged = { ...base, phases: [...buildPhases, ...trainingPhases] };
    const { eff, raw } = render(merged, activeRunId);

    // Latch only when the WHOLE pipeline is done. Training follows a
    // SUCCESSFUL build only, so keep polling past a terminal state ONLY when
    // the build completed and a training stage is still planned but not yet the
    // live active run; a paused / failed build latches immediately as before.
    const activeTerminal = RAIL_TERMINAL_EFF.includes(eff) || RAIL_TERMINAL_RAW.includes(raw);
    const buildEff = String((buildP && (buildP.effective_status || buildP.status)) || '').toLowerCase();
    const pendingTraining = !!(training && !trainActive && buildEff === 'completed');
    if (activeTerminal && !pendingTraining) {
      terminal = true;
      stop();
    }
    if (terminal) return;
    // Output-tail follows the ACTIVE run so a training phase's live tail shows
    // in the same panel once training is under way.
    try {
      rail.updateTail(await api(`/api/runs/${encodeURIComponent(activeRunId)}/output-tail`));
    } catch (_) { /* best-effort — the panel simply doesn't advance */ }
  }

  const pollOnce = pipeline ? pollOncePipeline : pollOncePlain;

  function start() {
    if (terminal) return; // a seeded terminal snapshot renders statically
    pollOnce();
    railTimer = setInterval(pollOnce, RAIL_POLL_MS);
  }
  return { el: rail.el, apply, pollOnce, start, stop };
}

// Honest header/state copy for each non-building effective_status the server
// derives (gui/services/liveness.py). `building` (and any unlisted state) means
// no override — the normal "Building …" heading stands.
const HONEST_BUILD_STATE = {
  stopping: { suffix: 'stopping', line: 'Stopping — the run is draining to a checkpoint. It will pause when the in-flight unit finishes.' },
  incomplete: { suffix: 'incomplete', line: 'This build did not make it through final completion — no ed4all run process is driving it. The record may be stale.' },
  'stalled?': { suffix: 'stalled?', line: 'This build may be stalled — no progress has been recorded recently.' },
  paused: { suffix: 'paused', line: 'Paused — this build is resumable.' },
};

/**
 * Reflect a run's honest effective_status in the progress heading + a state
 * line (never a fabricated "Building" for an incomplete/paused/stopping run). Pure
 * presentation over the elements already in `v`; announcements route through
 * the shell's single role=status region (no new live region). For a paused (or
 * otherwise resumable) run it appends the exact `ed4all run --resume <id>` hint.
 */
function applyHonestBuildState(shell, v, eff, payload, courseName) {
  const heading = v.querySelector('h1');
  let line = v.querySelector('.build-state-line');
  const info = HONEST_BUILD_STATE[eff];
  if (!info) {
    if (line) line.remove();
    if (heading) heading.textContent = courseName ? `Building ${courseName}` : 'Building your course';
    return;
  }
  const base = courseName || 'your course';
  if (heading) heading.textContent = `${base} — ${info.suffix}`;
  if (!line) {
    line = el('p', { class: 'build-state-line muted' });
    if (heading && heading.after) heading.after(line);
    else v.insertBefore(line, v.firstChild);
  }
  const workflowId = (payload && payload.workflow_id) || '';
  let text = info.line;
  if ((eff === 'paused' || eff === 'stopping' || eff === 'incomplete') && workflowId) {
    text += ` Resume with: ed4all run --resume ${workflowId}`;
  }
  line.textContent = text;
  if (typeof shell.setStatus === 'function') shell.setStatus(`${base}: ${info.suffix}.`);
}

/**
 * Course/book-name meta element for the run header area. Plain text (no
 * color-only signal); empty until a name is known. The /progress payload's
 * `course_name` is the LIVE source — an --auto-name run starts under a
 * provisional slug and the pipeline rebinds course identity mid-run, so the
 * name is re-set on every poll via `set()` (a changed name simply re-renders).
 */
function courseNameMeta(initial) {
  const span = el('span', { class: 'run-course-name' });
  function set(name) {
    clear(span);
    if (!name) return;
    span.appendChild(el('span', { text: String(name) }));
    span.appendChild(el('span', { class: 'sep', 'aria-hidden': 'true', text: ' · ' }));
  }
  set(initial || '');
  return { el: span, set };
}

/* ============================================ run health strip + debug */

// effective_status values that expand the health strip into the Debug panel:
// the honest non-building states (liveness.py) plus the failed family.
const DEBUG_STATES = ['paused', 'stopping', 'incomplete', 'stalled?', 'failed', 'error', 'timeout'];
// States after which the health poll stops (one final snapshot, then no more).
const HEALTH_TERMINAL = ['completed', 'failed', 'cancelled', 'canceled', 'interrupted', 'paused', 'timeout', 'error', 'incomplete'];
// One-line explanation per debug-triggering state (plain language; the
// paused/stopping/incomplete/stalled? copy mirrors HONEST_BUILD_STATE).
const DEBUG_STATE_EXPLAIN = {
  stopping: HONEST_BUILD_STATE.stopping.line,
  incomplete: HONEST_BUILD_STATE.incomplete.line,
  'stalled?': HONEST_BUILD_STATE['stalled?'].line,
  paused: HONEST_BUILD_STATE.paused.line,
  failed: 'This build failed. The findings below come from the run post-mortem diagnostics.',
  error: 'This build ended with an error. The findings below come from the run post-mortem diagnostics.',
  timeout: 'This build timed out. The findings below come from the run post-mortem diagnostics.',
};

/**
 * Split a doctor payload's checks into FAIL-first / WARN buckets. Shared by the
 * per-run health strip (`mountRunHealth`) and the top-level pipeline-health panel
 * (`mountPipelineHealthPanel`) so the severity mapping lives in exactly one
 * place. Missing/garbage payload → empty buckets (never a fabricated finding).
 */
function doctorFindings(payload) {
  const rows = [];
  const groups = (payload && Array.isArray(payload.groups)) ? payload.groups : [];
  groups.forEach((g) => {
    (Array.isArray(g.checks) ? g.checks : []).forEach((c) => rows.push(c));
  });
  const sev = (c) => {
    const s = String((c && c.severity) || '').toLowerCase();
    return s === 'critical' || s === 'error' ? 'fail' : s === 'warning' ? 'warn' : s;
  };
  return {
    fail: rows.filter((c) => sev(c) === 'fail'),
    warn: rows.filter((c) => sev(c) === 'warn'),
  };
}

/** One doctor finding row: severity pill + summary, expandable
 * remediation/detail (the health.js checkRow markup, so the .health-* styles
 * apply as-is). Shared by both doctor surfaces. */
function doctorFindingRow(check, sev) {
  const li = el('li', { class: 'health-check' });
  li.appendChild(el('div', { class: 'health-check-row' }, [
    statusPill(`severity:${sev}`),
    el('span', { class: 'health-check-summary', text: String(check.summary || check.name || '') }),
  ]));
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
    if (detail) details.appendChild(el('p', { class: 'health-detail muted', text: detail }));
    li.appendChild(details);
  }
  return li;
}

/**
 * Mount the compact run-health strip (fed by GET /api/health/doctor/run/{id})
 * for the build-progress page. Compact while building: an overall verdict
 * pill + WARN/FAIL finding counts, polled at a modest cadence (the endpoint
 * carries a ~30s server cache, so the poll is cheap). When the run reaches a
 * bad state (DEBUG_STATES) the strip expands into the Debug panel: run-scoped
 * doctor findings (FAIL first, then WARN) with expandable remediation rows
 * (the health.js row pattern), the effective_status explanation, and a
 * copyable `ed4all assistant --debug --run <id>` command — no new backend LLM
 * calls are wired here.
 *
 * Honesty/degradation contract: no payload (endpoint down / unknown run) →
 * the section renders EMPTY (the element is omitted, never a fabricated
 * verdict). The poll stops itself after a terminal state (one final
 * snapshot). Callers append `.el`, call `start()`, wire `setRunState` into
 * the stage-rail onState callback, and `stop()` on teardown.
 */
function mountRunHealth(runId) {
  const HEALTH_POLL_MS = 60000;
  const box = el('section', { class: 'run-health', 'aria-label': 'Run health' });
  let payload = null;
  let eff = '';
  let wfId = '';
  let timer = null;
  let latched = false;
  // The run this strip currently reports on. In the merged-pipeline rail this
  // follows the ACTIVE run (build → training) via setTarget(), so a training
  // failure surfaces the training run's post-mortem.
  let curId = String(runId || '');
  // Signature of the LAST painted subtree. The strip/Debug-panel DOM is a pure
  // function of {payload findings, effective_status, resolved target id}; when
  // none of those changed between polls we skip the (destructive) rebuild so
  // the ~3s progress poll no longer wipes a text selection the user is making
  // inside the Debug panel (e.g. highlighting the copyable assistant command).
  let lastSig = null;

  // A stable signature of everything `render()` reads to build the subtree.
  function renderSignature() {
    if (!payload) return 'empty';
    const found = doctorFindings(payload);
    const target = String((payload.orchestrator_run_id || '')).trim() || wfId || curId;
    const rowSig = (c) => [c && c.severity, c && (c.summary || c.name), c && c.remediation, c && c.detail];
    return JSON.stringify({
      eff,
      debug: DEBUG_STATES.includes(eff),
      exit_state: payload.exit_state || '',
      target,
      fail: found.fail.map(rowSig),
      warn: found.warn.map(rowSig),
    });
  }

  // True when the user currently has a non-collapsed text selection anchored
  // inside this box — i.e. they are highlighting text to copy. We must not tear
  // that subtree out from under them; defer the rebuild until the selection
  // clears (the next poll re-attempts, since lastSig is left stale).
  function selectionInBox() {
    const sel = typeof window.getSelection === 'function' ? window.getSelection() : null;
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return false;
    return !!((sel.anchorNode && box.contains(sel.anchorNode))
      || (sel.focusNode && box.contains(sel.focusNode)));
  }

  function render() {
    // (a) Nothing that affects the DOM changed → keep the existing subtree (and
    // any in-progress text selection) intact, no rebuild.
    const sig = renderSignature();
    if (sig === lastSig) return;
    // (b) Content DID change, but the user is mid-selection inside the panel —
    // defer the destructive rebuild so their highlight/copy survives. lastSig
    // stays stale so the next poll re-attempts once the selection is released.
    if (lastSig !== null && selectionInBox()) return;
    lastSig = sig;
    clear(box);
    if (!payload) return; // nothing fetched → omit the element entirely
    const found = doctorFindings(payload);
    const state = String(payload.exit_state || 'healthy');
    const sev = state === 'critical' ? 'fail' : state === 'degraded' ? 'warn' : 'ok';
    const parts = [];
    if (found.fail.length) parts.push(`${found.fail.length} failing`);
    if (found.warn.length) parts.push(`${found.warn.length} warning${found.warn.length === 1 ? '' : 's'}`);
    const label = parts.length ? `Run health: ${parts.join(', ')}` : 'Run health: all checks passing';
    box.appendChild(el('div', { class: 'run-health-strip' }, [
      statusPill(`severity:${sev}`),
      el('span', { class: 'run-health-label', text: label }),
    ]));
    if (!DEBUG_STATES.includes(eff)) return;

    // ---- Debug panel (expanded strip) ----
    const panel = el('div', { class: 'run-debug-panel' });
    panel.appendChild(el('h2', { class: 'run-debug-title', text: 'Debug this run' }));
    const explain = DEBUG_STATE_EXPLAIN[eff];
    if (explain) {
      panel.appendChild(el('p', { class: 'run-debug-state muted', text: `State: ${eff}. ${explain}` }));
    }
    const ordered = found.fail.map((c) => ['fail', c]).concat(found.warn.map((c) => ['warn', c]));
    if (ordered.length) {
      const list = el('ul', { class: 'health-checks' });
      ordered.forEach(([sevKey, c]) => list.appendChild(doctorFindingRow(c, sevKey)));
      panel.appendChild(list);
    } else {
      panel.appendChild(el('p', { class: 'muted', text: 'No failing or warning findings from the run diagnostics. Check the build log for detail.' }));
    }
    // "Ask the assistant" affordance: a copyable CLI command, never a new
    // backend LLM call. Prefer the orchestrator run id from the health
    // payload; fall back to the WF workflow id.
    const target = String((payload.orchestrator_run_id || '')).trim() || wfId || curId;
    const cmd = `ed4all assistant --debug --run ${target}`;
    panel.appendChild(el('p', {
      class: 'run-debug-assistant-hint muted',
      text: 'Ask the assistant about this run — it inspects the run state and diagnostics and explains what happened:',
    }));
    panel.appendChild(el('div', { class: 'resume-hint inline' }, [
      el('code', { class: 'kv resume-cmd', text: cmd }),
      copyHintBtn(cmd, 'Copy assistant debug command to clipboard'),
    ]));
    box.appendChild(panel);
  }

  async function pollOnce() {
    try {
      payload = await api(`/api/health/doctor/run/${encodeURIComponent(curId)}`);
    } catch (_) { /* best-effort — keep the last payload (or stay omitted) */ }
    render();
  }
  function stop() {
    if (timer != null) { clearInterval(timer); timer = null; }
  }
  function start() {
    if (latched) return; // a seeded terminal state already took its snapshot
    pollOnce();
    timer = setInterval(pollOnce, HEALTH_POLL_MS);
  }
  /**
   * Re-point the strip at a new (active) run — the merged rail calls this each
   * poll with the active run id; a same-id call is a no-op. On a real change it
   * drops the prior run's payload + latch and resumes polling so the strip
   * reflects the run it now targets (e.g. a training run that started after the
   * build finished + latched).
   */
  function setTarget(newId) {
    const next = String(newId || '');
    if (!next || next === curId) return;
    curId = next;
    payload = null;
    latched = false;
    render();
    if (timer == null) timer = setInterval(pollOnce, HEALTH_POLL_MS);
    pollOnce();
  }
  /** Feed the rail's effective_status + progress payload on every poll. */
  function setRunState(newEff, progressPayload) {
    eff = String(newEff || '').toLowerCase();
    if (progressPayload && progressPayload.workflow_id) wfId = String(progressPayload.workflow_id);
    render();
    if (HEALTH_TERMINAL.includes(eff) && !latched) {
      latched = true;
      stop();
      pollOnce(); // one final snapshot so the strip reflects the end state
    }
  }
  return { el: box, start, stop, setRunState, setTarget };
}

/* ============================================ full-pipeline (Stage A + B) */

/**
 * Top-level collapsible "Pipeline health" panel fed by the GLOBAL preflight
 * (GET /api/health/doctor). Sibling of the per-run health strips: it reports on
 * the host/environment (seats, GPU, disk, config) rather than one run. Reuses
 * `doctorFindings` / `doctorFindingRow` so the finding rows render identically
 * to the run strip.
 *
 * Honesty/degradation: no payload (endpoint down) → the element is OMITTED
 * entirely (`hidden`), never a fabricated verdict. Collapsed while
 * `exit_state === "healthy"`; auto-expanded the moment the state is degraded /
 * critical — but only re-applied on a STATE TRANSITION, so a manual toggle by
 * the operator between transitions is respected. Callers append `.el`, call
 * `start()`, and `stop()` on teardown.
 */
function mountPipelineHealthPanel() {
  const PANEL_POLL_MS = 60000;
  const summary = el('summary', { class: 'pipeline-health-summary' });
  const body = el('div', { class: 'pipeline-health-body' });
  const details = el('details', { class: 'pipeline-health-panel' }, [summary, body]);
  const wrap = el('section', { class: 'pipeline-health', 'aria-label': 'Pipeline health' });
  wrap.hidden = true; // omitted until a payload arrives (no fabricated verdict)
  wrap.appendChild(details);
  let payload = null;
  let prevState = null;
  let timer = null;

  function render() {
    clear(summary);
    clear(body);
    if (!payload) { wrap.hidden = true; return; }
    wrap.hidden = false;
    const found = doctorFindings(payload);
    const state = String(payload.exit_state || 'healthy');
    const sev = state === 'critical' ? 'fail' : state === 'degraded' ? 'warn' : 'ok';
    const parts = [];
    if (found.fail.length) parts.push(`${found.fail.length} failing`);
    if (found.warn.length) parts.push(`${found.warn.length} warning${found.warn.length === 1 ? '' : 's'}`);
    const label = parts.length ? `Pipeline health: ${parts.join(', ')}` : 'Pipeline health: all checks passing';
    summary.appendChild(statusPill(`severity:${sev}`));
    summary.appendChild(el('span', { class: 'pipeline-health-label', text: label }));
    // Auto-open only on a state TRANSITION (respect a manual toggle otherwise).
    if (state !== prevState) {
      details.open = state !== 'healthy';
      prevState = state;
    }
    const ordered = found.fail.map((c) => ['fail', c]).concat(found.warn.map((c) => ['warn', c]));
    if (ordered.length) {
      const list = el('ul', { class: 'health-checks' });
      ordered.forEach(([sevKey, c]) => list.appendChild(doctorFindingRow(c, sevKey)));
      body.appendChild(list);
    } else {
      body.appendChild(el('p', { class: 'muted', text: 'All preflight checks are passing.' }));
    }
  }
  async function pollOnce() {
    try {
      payload = await api('/api/health/doctor');
    } catch (_) { /* best-effort — keep the last payload (or stay omitted) */ }
    render();
  }
  function stop() {
    if (timer != null) { clearInterval(timer); timer = null; }
  }
  function start() {
    pollOnce();
    timer = setInterval(pollOnce, PANEL_POLL_MS);
  }
  return { el: wrap, start, stop };
}

/**
 * The follow-on LoRA-training (Stage-B) stage from a GET /api/runs/{id}/pipeline
 * chain, or null. Matched by workflow or stage id so a 404 / one-stage chain
 * (no training) yields null and the rail stays the plain single build rail.
 */
function pipelineTrainingStage(chain) {
  const stages = chain && Array.isArray(chain.stages) ? chain.stages : [];
  return stages.find((s) => s && (s.workflow === 'trainforge_train' || s.stage === 'training')) || null;
}

/**
 * Synthesize the trailing "training" group's PENDING nodes for a training stage
 * that has not started yet (present:false). One node per planned_phases name,
 * friendly-labelled (the same titleCase lookup the console uses), grouped
 * `training`, no wall-clock — honest "planned, not started", never a fake
 * spinner or percent. Empty when the stage carries no planned_phases.
 */
function plannedTrainingPhases(stage) {
  const names = stage && Array.isArray(stage.planned_phases) ? stage.planned_phases : [];
  return names.map((name) => ({
    name: String(name),
    label: titleCase(String(name)),
    state: 'pending',
    group: 'training',
  }));
}

/* ==================================================== live run (#/live) */

// Statuses that mean a run is over — the complement is "live". Mirrors the
// backend's TERMINAL_STATUSES (gui/services/progress_service.py) minus
// "paused": a paused run is resumable and still the operator's live concern.
const LIVE_RUN_OVER = ['completed', 'failed', 'cancelled', 'canceled', 'interrupted', 'timeout', 'error'];

/**
 * #/live — THIN resolver: find the currently running workflow in the merged
 * run list (GET /api/runs, newest first — includes CLI-launched orchestrator
 * runs) and delegate to the SAME build-progress view (`renderProgress`,
 * which itself degrades to the CLI-run path for orchestrator-only ids). A
 * running/paused run wins over any other non-terminal record (requested /
 * queued); nothing live → an honest empty state pointing at Run history —
 * never a guessed run.
 */
export async function renderLiveRun(shell) {
  shell.crumbs([{ text: 'Live run' }]);
  const v = clear(shell.view());
  shell.setBusy(true);
  shell.setStatus('Finding the live run.');
  let runs;
  try {
    runs = (await api('/api/runs')).runs || [];
  } catch (e) {
    shell.setBusy(false);
    v.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    toastErr(e, 'Could not load runs');
    return;
  }
  const status = (r) => String((r && r.status) || '').toLowerCase();
  const target = runs.find((r) => ['running', 'paused'].includes(status(r)))
    || runs.find((r) => !LIVE_RUN_OVER.includes(status(r)));
  if (!target) {
    shell.setBusy(false);
    v.appendChild(el('h1', { text: 'Live run' }));
    v.appendChild(emptyState({
      title: 'No live run',
      message: 'Nothing is building right now. Past and finished builds are in Run history.',
      cta: { label: 'View run history', href: '#/runs' },
    }));
    shell.setStatus('No live run.');
    return;
  }
  const runId = target.run_id || target.runId || target.id || '';
  return renderProgress(shell, runId);
}

async function renderProgress(shell, runId) {
  shell.crumbs([{ text: 'Library', href: '#/library' }, { text: 'Create course', href: '#/create' }, { text: 'Build progress' }]);
  const v = clear(shell.view());
  shell.setBusy(true);
  shell.setStatus('Loading build progress.');

  v.appendChild(el('h1', { text: 'Building your course' }));

  // Resolve the run record + the canonical phase list (friendly labels) in
  // parallel. The phase list comes from /api/workflows so the order + labels
  // are never hardcoded here.
  let record;
  let phases;
  try {
    [record, phases] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(runId)}`),
      fetchPhaseList(),
    ]);
  } catch (e) {
    // Degraded re-attach for CLI-launched runs. run_service only reads the
    // GUI run registry (state/gui/runs/) BY DESIGN, so a bare orchestrator
    // workflow id — `ed4all run ...` from a terminal — 404s here as
    // unknown_run even while the pipeline is live. progress_service
    // deliberately accepts that same id ("CLI-launched runs are observable
    // too"), so before surfacing the hard error we probe /progress once: if
    // it answers, render the honest degraded page (stage rail + poll only —
    // no log console, no cancel; the GUI never launched this process, so
    // neither capability exists and faking them is forbidden). Any other
    // failure — or a dead /progress too — keeps the error path verbatim.
    if (e && e.status === 404 && e.error === 'unknown_run') {
      let snapshot = null;
      try {
        snapshot = await api(`/api/runs/${encodeURIComponent(runId)}/progress`);
      } catch (_) { snapshot = null; /* not observable either — fall through */ }
      if (snapshot) return renderCliRunProgress(shell, v, runId, snapshot);
    }
    shell.setBusy(false);
    v.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    toastErr(e, 'Could not load run');
    return;
  }

  const courseName = record.course_name || '';
  v.replaceChild(el('h1', { text: courseName ? `Building ${courseName}` : 'Building your course' }), v.querySelector('h1'));

  // Elapsed anchor (from started_at when available).
  const startMs = record.started_at ? Date.parse(record.started_at) : Date.now();

  // Honest-ETA medians (best-effort): the sibling endpoint may not exist yet, so
  // a fetch failure degrades the console to "estimating…" rather than erroring.
  const durations = await fetchPhaseDurations(record.workflow || WORKFLOW);

  // The shared Build Console assembles the component kit: a phaseTimeline (real
  // SVG rings replace the old static glyphs), the canonical elapsed timer, and
  // the SINGLE role=status aria-live region (the narrative line). It is PURE
  // PRESENTATION — this page owns the WS + every fetch; the console only parses
  // the lines we feed it. Friendly phase labels are resolved here from the
  // /api/workflows list (never hardcoded in the kit).
  // Reuse the studio shell's single #status region (the page keeps exactly one
  // polite live-truth region; the console announces phase transitions into it).
  const liveRegion = typeof shell.statusRegion === 'function' ? shell.statusRegion() : null;
  // The console owns the cancel BUTTON; this page owns the cancel FETCH and the
  // two-stage 202 → cancelled resolution (the console structurally never
  // fetches). On a 202 / cancel_requested we flip the button to "Cancelling…"
  // and wait for the WS terminal {type:status} frame to resolve to cancelled.
  const progress = runProgressConsole({
    phases: phases.map((p) => ({ name: p.name, label: p.label || titleCase(p.name), optional: p.optional })),
    startMs,
    liveRegion: liveRegion || undefined,
    durations: durations.durations,
    durationsSource: durations.source,
    onCancel: () => requestCancel(shell, runId, progress),
  });

  // Course/book name + run id sit in the meta line (left of the console's
  // elapsed timer). The name is LIVE: an --auto-name run rebinds course
  // identity mid-run, and /progress carries the freshest course_name — the
  // rail's onState callback below re-syncs the meta line + heading each poll.
  let liveCourseName = courseName;
  const courseMeta = courseNameMeta(courseName);
  progress.el.querySelector('.run-progress-meta').insertBefore(
    el('span', {}, [
      courseMeta.el,
      el('span', { text: `Run ${runId}` }),
      el('span', { class: 'sep', 'aria-hidden': 'true', text: ' · ' }),
    ]),
    progress.elapsedEl,
  );
  v.appendChild(progress.el);

  // Run-health strip (mounted below the rail — visually subordinate to the
  // stage tracker; expands into the Debug panel on a bad effective_status).
  const healthCtl = mountRunHealth(runId);
  // vLLM seat strip (below the health strip): every registered seat + the
  // expected-vs-actual warning when the CURRENT phase declares a seat that is
  // not serving. See gui/static/studio/seats.js + gui/services/seat_service.py.
  const seatCtl = mountSeatMonitor({ runId });

  // Stage-tracker rail + live stats band, fed by GET /api/runs/{id}/progress
  // (config-driven phase list merged with the run's real state — see
  // gui/services/progress_service.py). In MERGED-pipeline mode the ONE rail
  // shows the build phases followed by the follow-on LoRA-training (Stage-B)
  // phases as a trailing "training" group — never a separate segregated band.
  // Poll loop, pipeline-chain merge, and terminal-stop live in the shared
  // mountStageRail helper (also used by the degraded CLI-run page).
  const railCtl = mountStageRail(
    runId,
    `Pipeline progress for ${courseName || runId}`,
    (eff, payload, activeRunId) => {
      const live = payload && typeof payload.course_name === 'string' ? payload.course_name.trim() : '';
      if (live && live !== liveCourseName) {
        liveCourseName = live;
        courseMeta.set(live);
      }
      applyHonestBuildState(shell, v, eff, payload, liveCourseName);
      // The run-health strip follows the ACTIVE run (build → training) so a
      // training failure surfaces the training run's post-mortem.
      healthCtl.setTarget(activeRunId);
      healthCtl.setRunState(eff, payload);
      seatCtl.setRunState(eff);
    },
    { pipeline: true },
  );
  v.appendChild(railCtl.el);
  v.appendChild(healthCtl.el);
  v.appendChild(seatCtl.el);

  // Global preflight "Pipeline health" panel (host/environment doctor —
  // seats, GPU, disk, config — independent of any one run). Mounted below the
  // rail; best-effort, omitted entirely when its own endpoint is down.
  const healthPanelCtl = mountPipelineHealthPanel();
  v.appendChild(healthPanelCtl.el);
  healthPanelCtl.start();

  // The failure panel + terminal CTAs render below the console.
  const finalBox = el('div', { class: 'final-box' });
  v.appendChild(finalBox);

  shell.setBusy(false);

  // The phase explicitly reported failed by a structured WS line (A6) wins over
  // the "last running" heuristic; the console tracks it for us.
  function finalize(status, errMsg) {
    clear(finalBox);
    // The merged rail owns its own lifecycle: it keeps polling for a follow-on
    // training stage (build done → training running) and self-latches when the
    // WHOLE pipeline is done, so finalize does NOT stop it here — stopping on
    // build completion would kill the training poll. A paused / failed build
    // self-latches on its next poll (training follows a successful build only).
    // The seat monitor latches on the build's terminal status (its watch is the
    // build run's seat schedule).
    seatCtl.setRunState(status);
    // I1 review flow: a PAUSED run halted at the objectives-review checkpoint —
    // NOT a failure. The console's onStatus() has no paused arm (its else-branch
    // reads as failed), so drive the paused presentation directly: freeze the
    // timer, hide the cancel control, mark the review phase done, leave the
    // downstream phases pending, and render the review panel.
    if (status === 'paused') {
      progress.freezeElapsed(record.finished_at && record.started_at
        ? Date.parse(record.finished_at) - Date.parse(record.started_at)
        : undefined);
      progress.hideCancel();
      const pausedPhase = record.paused_phase
        || (record.params && record.params.stop_after)
        || 'course_planning';
      progress.setState(pausedPhase, 'done');
      progress.announce('Build paused for objectives review.');
      renderReviewPanel(shell, finalBox, record, courseName, pausedPhase);
      shell.setStatus('Build paused for objectives review. Review the objectives, then resume the build.');
      return;
    }
    // Drive the console's terminal state (rings + the polite narrative line).
    progress.onStatus(status);
    if (status === 'completed') {
      finalBox.appendChild(el('p', { class: 'ok', text: 'Your course is ready.' }));
      const slug = courseSlug(courseName);
      finalBox.appendChild(el('a', { class: 'btn primary', href: `#/viewer/${encodeURIComponent(slug)}`, text: 'Open course' }));
      shell.setStatus('Your course is ready. Open course to view it.');
    } else if (status === 'cancelled' || status === 'interrupted') {
      finalBox.appendChild(el('p', { class: 'warn', text: `Build ${status}.` }));
      // Recovery (no dead-end): Retry from the same params + Resume — without
      // forcing the operator back to step 1 to re-upload.
      renderRecoveryActions(shell, finalBox, record, courseName, null);
      shell.setStatus(`Build ${status}. You can retry or resume below.`);
    } else {
      // A6 failure: the console has already marked the failing phase; surface the
      // failed phase's drawer + the actionable failure panel verbatim.
      const fp = progress.getFailedPhase() || progress.getLastRunning();
      const failedRow = fp ? progress.get(fp) : null;
      const drawer = failedRow && failedRow.detail ? failedRow.detail : finalBox;
      if (failedRow) failedRow.expand();
      renderFailurePanel(shell, drawer, runId, courseName, fp, errMsg, record);
      shell.setStatus('The course build failed. Review what went wrong and what to do next.');
    }
  }

  // If the run is already terminal (refresh after completion), reflect it from
  // the record without opening a WS.
  const terminal = ['completed', 'failed', 'cancelled', 'interrupted', 'paused'];
  if (terminal.includes(record.status)) {
    progress.freezeElapsed(record.finished_at && record.started_at
      ? Date.parse(record.finished_at) - Date.parse(record.started_at)
      : undefined);
    // Replay phase completion from the persisted gate_results (phase_results).
    const pr = record.gate_results;
    if (pr && typeof pr === 'object') Object.keys(pr).forEach((n) => progress.setState(n, 'done'));
    // A6: the persisted failed_phase (set by run_service) drives the failure
    // marker on a refresh, so we don't fall back to the "last running" guess.
    if (record.failed_phase) progress.setState(record.failed_phase, 'failed');
    // Stash it so finalize's failure path uses the persisted phase, not a guess.
    if (record.status === 'failed' && record.failed_phase) {
      progress.onLine(`[phase] ${record.failed_phase} failed`);
    }
    finalize(record.status, record.error || record.failure_reason);
    // A viewed run that is already terminal may still have a LIVE follow-on
    // training stage (build done, training running), so start the merged rail
    // (+ health / seat / the global health panel are already started above) so
    // it keeps polling for it and self-latches when the whole pipeline is done.
    railCtl.start();
    healthCtl.start();
    seatCtl.start();
    return () => { railCtl.stop(); healthCtl.stop(); seatCtl.stop(); healthPanelCtl.stop(); };
  }

  // Live stream. The WS carries {type:line} (the console parses the ISO-prefixed
  // "[phase]"/"[progress]" lines, including Tier-1 timing) and a terminal
  // {type:status}.
  progress.startFirstRunning();
  railCtl.start();
  healthCtl.start();
  seatCtl.start();

  const ws = openRunSocket(runId, {
    onLine: (line) => progress.onLine(line),
    onStatus: (frame) => finalize(frame.status, frame.error),
    onError: (msg) => {
      progress.onError(msg);
      finalBox.appendChild(el('p', { class: 'error', role: 'alert', text: `Lost connection to the build: ${msg}` }));
    },
  });

  // Teardown: close the WS + stop the console timer + rail/health/seat/global
  // health-panel polls on route change.
  return () => { progress.destroy(); ws.close(); railCtl.stop(); healthCtl.stop(); seatCtl.stop(); healthPanelCtl.stop(); };
}

/* ================================== degraded progress (CLI-launched runs) */

/**
 * Degraded-but-honest progress page for a run the GUI did not launch.
 *
 * WHY this exists: `ed4all run ...` writes orchestrator state that
 * progress_service reads, but never a GUI run record, a GUI log file
 * (state/gui/logs/<id>.log), or an in-process driver task. So on this page:
 *   - the runProgressConsole (WS log console) is OMITTED — its log source
 *     does not exist, and faking a stream would violate the no-stubs rule;
 *   - the cancel button is OMITTED — there is no driver task to cancel, so
 *     a wired-looking control here would be a fabricated capability;
 *   - the stage-tracker rail + 3s poll IS rendered, identical to the normal
 *     path (shared mountStageRail), because /progress is real pipeline state
 *     and stops itself at a terminal status just like the normal path.
 * No explanatory banner (owner direction): the page renders the rail /
 * stats / output directly; the shell's single polite live region stays the
 * only announcement channel (the a11y contract of this file).
 */
function renderCliRunProgress(shell, v, runId, snapshot) {
  // Same meta affordance the console path provides: the course/book name
  // (when the /progress payload carries one — it is LIVE, so an --auto-name
  // rebind mid-run updates it on the next poll) + the run id, plainly
  // visible under the "Building your course" heading.
  let liveCourseName = '';
  const courseMeta = courseNameMeta('');
  v.appendChild(el('p', { class: 'muted' }, [
    courseMeta.el,
    el('span', { text: `Run ${runId}` }),
  ]));

  // Run-health strip + Debug panel (below the rail; shared helper).
  const healthCtl = mountRunHealth(runId);
  // vLLM seat strip (shared helper): registered seats + the expected-vs-actual
  // warning for the current phase.
  const seatCtl = mountSeatMonitor({ runId });

  const railCtl = mountStageRail(
    runId,
    `Pipeline progress for ${runId}`,
    (eff, payload, activeRunId) => {
      const live = payload && typeof payload.course_name === 'string' ? payload.course_name.trim() : '';
      if (live && live !== liveCourseName) {
        liveCourseName = live;
        courseMeta.set(live);
      }
      applyHonestBuildState(shell, v, eff, payload, liveCourseName);
      // The run-health strip follows the ACTIVE run (build → training).
      healthCtl.setTarget(activeRunId);
      healthCtl.setRunState(eff, payload);
      seatCtl.setRunState(eff);
    },
    { pipeline: true },
  );
  v.appendChild(railCtl.el);
  v.appendChild(healthCtl.el);
  v.appendChild(seatCtl.el);
  // Global preflight "Pipeline health" panel (host/environment doctor). Mounted
  // below the rail; best-effort, omitted entirely when its own endpoint is down.
  const healthPanelCtl = mountPipelineHealthPanel();
  v.appendChild(healthPanelCtl.el);
  healthPanelCtl.start();
  // Seed the rail from the probe snapshot we already hold (first paint costs
  // no extra fetch); start() then runs the merged 3s loop (build phases +
  // trailing training group). In pipeline mode the seed never latches, so a
  // terminal build snapshot doesn't freeze the rail before training is merged.
  railCtl.apply(snapshot);
  railCtl.start();
  healthCtl.start();
  seatCtl.start();

  shell.setBusy(false);
  shell.setStatus('Showing build progress.');

  // Teardown: the rail + health + seat + global health-panel polls to stop —
  // no WS, no console timer.
  return () => { railCtl.stop(); healthCtl.stop(); seatCtl.stop(); healthPanelCtl.stop(); };
}

/* =============================================== A6 failure panel (UX) */

/**
 * Render the actionable failure panel into `box` for a failed build.
 *
 * Fetches GET /api/runs/<runId>/validation-report and renders:
 *   - a role="alert" intro naming the failed phase + reason,
 *   - a severity-badged failed-gate table (from the digest),
 *   - the validation-report per-block summary (pass/fail/escalated counts by
 *     block type) when a courseforge_validation_report.json is present,
 *   - "what to do next" affordances, each gated by applicability + confirm:
 *       (a) Re-run validation  -> courseforge-validate (always, two-pass courses)
 *       (b) Rewrite failing blocks -> courseforge-rewrite --blocks <types>
 *           (only when the report names failing block types)
 *       (c) Re-run failed phase -> POST /api/runs/phase (when a phase is known)
 *       (d) Download full log   -> link to the run log view.
 *
 * Best-effort: a fetch error still renders the intro + log link, so the operator
 * is never stranded with a raw error string.
 */
async function renderFailurePanel(shell, box, runId, courseName, failedPhase, errMsg, record) {
  clear(box);
  const panel = el('section', { class: 'failure-panel', role: 'alert', 'aria-labelledby': 'failure-h' });
  box.appendChild(panel);
  panel.appendChild(el('h2', { id: 'failure-h', class: 'failure-title', text: 'The course build failed' }));

  const phaseLabel = failedPhase ? titleCase(failedPhase) : null;
  const intro = phaseLabel
    ? `Failed during “${phaseLabel}”${errMsg ? `: ${errMsg}` : '.'}`
    : (errMsg || 'The course build did not finish.');
  panel.appendChild(el('p', { class: 'failure-intro', text: intro }));

  // The action box is a non-alert region so SRs don't re-announce the controls.
  const actions = el('div', { class: 'failure-actions' });
  const detail = el('div', { class: 'failure-detail' });

  let report = null;
  let failedGates = [];
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/validation-report`);
    report = data.report || null;
    failedGates = Array.isArray(data.failed_gates) ? data.failed_gates : [];
  } catch (_) { /* fetch failure: still render the log affordance below */ }

  // Failed-gate table (severity-badged), when the digest has rows.
  if (failedGates.length) {
    detail.appendChild(el('h3', { text: 'Failed checks' }));
    const table = el('table', { class: 'gate-table' }, [
      el('caption', { class: 'visually-hidden', text: 'Validation checks that failed' }),
      el('thead', {}, [el('tr', {}, [
        el('th', { scope: 'col', text: 'Severity' }),
        el('th', { scope: 'col', text: 'Check' }),
        el('th', { scope: 'col', text: 'What went wrong' }),
      ])]),
    ]);
    const tbody = el('tbody', {});
    failedGates.forEach((g) => {
      const sev = (g.severity || 'warning').toLowerCase();
      tbody.appendChild(el('tr', {}, [
        el('td', {}, [el('span', { class: `sev-badge sev-${sev}`, text: sev === 'critical' ? 'Critical' : 'Warning' })]),
        el('td', { text: g.gate_id || (g.phase ? titleCase(g.phase) : '—') }),
        el('td', { text: g.message || 'validation gate failed' }),
      ]));
    });
    table.appendChild(tbody);
    detail.appendChild(table);
  }

  // Per-block summary from the validation report, when present.
  const blockSummary = report ? summarizeBlocks(report.per_block_results) : null;
  if (blockSummary && blockSummary.rows.length) {
    detail.appendChild(el('h3', { text: 'Content blocks' }));
    const table = el('table', { class: 'block-table' }, [
      el('caption', { class: 'visually-hidden', text: 'Per-block-type pass, fail and escalated counts' }),
      el('thead', {}, [el('tr', {}, [
        el('th', { scope: 'col', text: 'Block type' }),
        el('th', { scope: 'col', text: 'Passed' }),
        el('th', { scope: 'col', text: 'Failed' }),
        el('th', { scope: 'col', text: 'Escalated' }),
      ])]),
    ]);
    const tbody = el('tbody', {});
    blockSummary.rows.forEach((r) => {
      tbody.appendChild(el('tr', {}, [
        el('td', { text: r.block_type }),
        el('td', { text: String(r.passed) }),
        el('td', { text: String(r.failed) }),
        el('td', { text: String(r.escalated) }),
      ]));
    });
    table.appendChild(tbody);
    detail.appendChild(table);
  }

  // I4 stage-2 — the per-page / per-block rewrite PICKER. When the report
  // carries per_block_results, let the operator choose exactly which failing
  // blocks (or whole pages) to re-author, instead of only the coarse per-type
  // affordance below. The JS is deliberately dumb: it renders checkboxes,
  // gathers the raw checked values, and posts them as arrays — ALL selection
  // composition / subsumption dedup lives server-side in run_service
  // (_normalize_blocks_param). Unknown-token validation intentionally happens
  // at rewrite RUN time (a loud phase failure), not at enqueue.
  if (report && Array.isArray(report.per_block_results)) {
    renderRewritePicker(shell, detail, report.per_block_results, courseName);
  }

  panel.appendChild(detail);

  // ---- What to do next (affordances). Each confirms before enqueuing. ----
  actions.appendChild(el('h3', { text: 'What to do next' }));
  const list = el('div', { class: 'affordances' });

  // (0) Retry + Resume — the no-dead-end pair. These re-launch from the run's
  // OWN params (no Back-to-start + re-upload). Resume is feature-detected.
  renderRecoveryActions(shell, list, record, courseName, failedPhase);

  // (a) Re-run validation — only meaningful for two-pass Courseforge courses,
  // which is exactly the case where a validation report exists.
  if (report) {
    list.appendChild(affordanceBtn(
      'Re-run validation',
      'Re-run the validation checks against the existing course content.',
      () => enqueuePhaseRun(shell, {
        workflow: 'courseforge_validate',
        phase: 'inter_tier_validation',
        course_name: courseName,
      }, 'Re-run validation for this course?'),
    ));
  }

  // (b) Rewrite failing blocks — only when the report names failing block types.
  const failingTypes = blockSummary ? blockSummary.failingTypes : [];
  if (report && failingTypes.length) {
    list.appendChild(affordanceBtn(
      'Rewrite failing blocks',
      `Regenerate the ${failingTypes.length} failing block type(s): ${failingTypes.join(', ')}.`,
      () => enqueuePhaseRun(shell, {
        workflow: 'courseforge_rewrite',
        phase: 'content_generation_rewrite',
        course_name: courseName,
        options: { blocks: failingTypes.join(',') },
      }, `Rewrite the failing block types (${failingTypes.join(', ')})?`),
    ));
  }

  // (c) Re-run the failed phase directly.
  if (failedPhase) {
    list.appendChild(affordanceBtn(
      'Re-run failed step',
      `Re-run just the “${phaseLabel}” step.`,
      () => enqueuePhaseRun(shell, {
        workflow: WORKFLOW,
        phase: failedPhase,
        course_name: courseName,
      }, `Re-run the “${phaseLabel}” step?`),
    ));
  }

  // (d) Download full log (always available).
  list.appendChild(el('a', {
    class: 'btn',
    href: `#/create/${encodeURIComponent(runId)}/log`,
    text: 'View / download build log',
  }));

  actions.appendChild(list);
  panel.appendChild(actions);
}

/**
 * Render the no-dead-end recovery pair (Retry + Resume) into `container`.
 *
 *   Retry  — re-POST /api/runs with the run's OWN params (record.params), so the
 *            operator never goes Back to start to re-upload the textbook. Confirms,
 *            then navigates to the new run's progress view.
 *   Resume — feature-detected: when the backend advertises resume support on the
 *            run record (record.resume_supported / record.can_resume, the sibling's
 *            flag), a "Resume build" button re-POSTs /api/runs with a resume option
 *            keyed on the workflow_id. Otherwise (today) we surface the copyable
 *            `ed4all run --resume <workflow_id>` command as a hint — never a button
 *            that silently no-ops.
 *
 * Best-effort: missing params disables Retry with an explanatory note rather than
 * enqueuing a paramless run.
 */
function renderRecoveryActions(shell, container, record, courseName, _failedPhase) {
  const params = record && record.params && typeof record.params === 'object' ? record.params : null;
  const workflow = (record && record.workflow) || WORKFLOW;
  const workflowId = (record && record.workflow_id) || '';

  // ---- Retry (re-run from the same params; NO re-upload) ----
  if (params && (params.corpus || params.upload_id)) {
    container.appendChild(affordanceBtn(
      'Retry build',
      'Start a fresh build from the same textbook and settings — no need to re-upload.',
      () => enqueueRetry(shell, record),
      true,
    ));
  } else {
    const wrap = el('div', { class: 'affordance' });
    wrap.appendChild(el('button', { type: 'button', class: 'btn', text: 'Retry build', disabled: true }));
    wrap.appendChild(el('p', { class: 'affordance-hint muted', text: 'Retry is unavailable for this run (its original inputs were not recorded). Start a new course instead.' }));
    container.appendChild(wrap);
  }

  // ---- Resume (feature-detected) ----
  const resumeSupported = !!(record && (record.resume_supported || record.can_resume));
  if (resumeSupported && workflowId) {
    container.appendChild(affordanceBtn(
      'Resume build',
      'Continue this build from where it stopped, reusing the work already completed.',
      () => enqueueResume(shell, record),
      true,
    ));
  } else if (workflowId) {
    // No in-browser resume yet → copyable CLI command (honest, no no-op button).
    const cmd = `ed4all run --resume ${workflowId}`;
    const wrap = el('div', { class: 'affordance' });
    wrap.appendChild(el('p', { class: 'affordance-hint muted', text: 'To continue this build from where it stopped, run this command:' }));
    const row = el('div', { class: 'resume-hint inline' }, [
      el('code', { class: 'kv resume-cmd', text: cmd }),
      copyHintBtn(cmd),
    ]);
    wrap.appendChild(row);
    container.appendChild(wrap);
  }
}

/** Copy-to-clipboard button for a CLI command hint (a real <button>). */
function copyHintBtn(text, ariaLabel) {
  const btn = el('button', { type: 'button', class: 'btn', 'aria-label': ariaLabel || 'Copy resume command to clipboard', text: 'Copy command' });
  btn.addEventListener('click', async () => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        toast('Copied', text, 'success', 2500);
        return;
      }
    } catch (_) { /* fall through */ }
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
    ta.remove();
    toast(ok ? 'Copied' : 'Copy failed', ok ? text : 'Clipboard unavailable.', ok ? 'success' : 'error', 2500);
  });
  return btn;
}

/** Re-POST /api/runs with the terminal run's OWN params (Retry). */
async function enqueueRetry(shell, record) {
  const p = record.params || {};
  const body = {
    workflow: record.workflow || WORKFLOW,
    course_name: record.course_name || p.course_name || '',
    corpus: p.corpus || p.upload_id,
    options: p.options || {},
  };
  if (p.weeks) body.weeks = Number(p.weeks);
  if (p.mode) body.mode = p.mode;
  if (p.provider) body.provider = p.provider;
  if (p.model) body.model = p.model;
  try {
    const resp = await apiJSON('/api/runs', 'POST', body);
    toast('Retrying…', '', 'success');
    location.hash = `#/create/${encodeURIComponent(resp.run_id)}`;
  } catch (e) {
    toastErr(e, 'Could not retry');
    shell.setStatus(`Could not retry: ${shell.errText(e)}`);
  }
}

/** Re-POST /api/runs with a resume option (feature-detected Resume). */
async function enqueueResume(shell, record) {
  const p = record.params || {};
  const body = {
    workflow: record.workflow || WORKFLOW,
    course_name: record.course_name || p.course_name || '',
    corpus: p.corpus || p.upload_id,
    options: p.options || {},
    resume: record.workflow_id,
  };
  try {
    const resp = await apiJSON('/api/runs', 'POST', body);
    toast('Resuming…', '', 'success');
    location.hash = `#/create/${encodeURIComponent(resp.run_id)}`;
  } catch (e) {
    toastErr(e, 'Could not resume');
    shell.setStatus(`Could not resume: ${shell.errText(e)}`);
  }
}

/* ============================================ I1 objectives-review panel */

/**
 * Render the objectives-review panel for a run PAUSED at the review checkpoint
 * (``stop_after=course_planning``).
 *
 * The composed loop's middle step: the build halted after planning the learning
 * objectives, and this panel lets the author (a) open the objectives editor for
 * the run's course and (b) resume the SAME build past the checkpoint with a
 * plain resume — the edited objectives file is picked up in place.
 *
 * A11y: a labelled <section> (aria-labelledby its <h2>); the editor deep-link is
 * a real <a> that announces it opens a new tab (WCAG 3.2.5); the resume control
 * is a real <button>; status changes announce via the shell's single polite
 * live region + the console's role=status line.
 *
 * Best-effort: the paused-at info (objectives file path + course id) is fetched
 * from ``GET /api/runs/<id>/review``; a fetch failure still renders the editor
 * link + resume button so the author is never stranded.
 */
async function renderReviewPanel(shell, box, record, courseName, pausedPhase) {
  clear(box);
  const panel = el('section', { class: 'review-panel', 'aria-labelledby': 'review-h' });
  box.appendChild(panel);
  panel.appendChild(el('h2', { id: 'review-h', class: 'review-title', text: 'Review the learning objectives' }));
  panel.appendChild(el('p', {
    class: 'review-intro',
    text: 'The build paused after planning the learning objectives so you can review and edit them before the rest of the course is generated. When you are done, resume the build below.',
  }));

  // Best-effort paused-at context (objectives file + course id for the editor).
  let info = null;
  try {
    info = await api(`/api/runs/${encodeURIComponent(record.run_id)}/review`);
  } catch (_) { /* degrade: still render the editor link + resume below */ }

  // Surface the EXACT objectives file the editor writes and a plain resume reads
  // in place — this is the "which path will be used" disclosure.
  if (info && info.objectives_path) {
    panel.appendChild(el('p', { class: 'review-path' }, [
      el('span', { class: 'review-path-label', text: 'Objectives file: ' }),
      el('code', { class: 'kv', text: info.objectives_path }),
    ]));
  }

  const courseId = (info && info.course_id) || courseName || '';

  const actions = el('div', { class: 'review-actions' });
  // (a) Edit objectives — the Advanced objectives editor (opens in a new tab so
  // the build view stays put). There is no per-course deep link, so the course
  // id to open is named explicitly below.
  actions.appendChild(el('a', {
    class: 'btn',
    href: '/advanced/#/courses',
    target: '_blank',
    rel: 'noopener',
    'aria-label': 'Edit objectives (opens the Advanced objectives editor in a new tab)',
    text: 'Edit objectives',
  }));
  // (b) Resume build — a plain resume of THIS run past the review checkpoint.
  const resumeBtn = el('button', { type: 'button', class: 'btn primary', text: 'Resume build' });
  resumeBtn.addEventListener('click', () => resumeAfterReview(shell, record, resumeBtn));
  actions.appendChild(resumeBtn);
  panel.appendChild(actions);

  // Name the course to open in the editor (the editor lists courses; there is
  // no per-course deep link), and reassure that resume picks up the edits.
  if (courseId) {
    panel.appendChild(el('p', { class: 'review-hint muted' }, [
      el('span', { text: 'In the objectives editor, open the course ' }),
      el('code', { class: 'kv', text: String(courseId) }),
      el('span', { text: ', save your edits, then return here and resume the build.' }),
    ]));
  }
}

/**
 * Resume a review-paused build past the checkpoint. Re-POSTs /api/runs with the
 * paused run's GUI run_id + ``clear_stop_after`` so the persisted --stop-after
 * marker is stripped and the build runs on; navigates to the new run's progress
 * view (progress continues in the same view).
 */
async function resumeAfterReview(shell, record, btn) {
  if (btn) btn.disabled = true;
  const body = {
    workflow: record.workflow || WORKFLOW,
    resume_run_id: record.run_id,
    clear_stop_after: true,
  };
  try {
    const resp = await apiJSON('/api/runs', 'POST', body);
    toast('Resuming build…', 'Continuing past the objectives-review checkpoint.', 'success');
    shell.setStatus('Resuming the build. Progress continues below.');
    location.hash = `#/create/${encodeURIComponent(resp.run_id)}`;
  } catch (e) {
    if (btn) btn.disabled = false;
    toastErr(e, 'Could not resume');
    shell.setStatus(`Could not resume the build: ${shell.errText(e)}`);
  }
}

/** Build one affordance control: a button with a hint, confirm-on-click. */
function affordanceBtn(label, hint, onConfirm) {
  const wrap = el('div', { class: 'affordance' });
  const btn = el('button', { type: 'button', class: 'btn primary', text: label });
  btn.addEventListener('click', () => {
    if (!window.confirm(hint + '\n\nStart this now?')) return;
    btn.disabled = true;
    onConfirm().finally(() => { btn.disabled = false; });
  });
  wrap.appendChild(btn);
  wrap.appendChild(el('p', { class: 'affordance-hint muted', text: hint }));
  return wrap;
}

/**
 * Enqueue a single-phase / stage-subcommand run via POST /api/runs/phase, then
 * navigate to its progress view. Surfaces failures via toast.
 */
async function enqueuePhaseRun(shell, body, _confirmMsg) {
  try {
    const resp = await apiJSON('/api/runs/phase', 'POST', body);
    toast('Started.', '', 'success');
    location.hash = `#/create/${encodeURIComponent(resp.run_id)}`;
  } catch (e) {
    toastErr(e, 'Could not start');
    shell.setStatus(`Could not start: ${shell.errText(e)}`);
  }
}

/**
 * Roll the report's flat per_block_results[] up to per-block-type counts.
 * Returns {rows:[{block_type,passed,failed,escalated}], failingTypes:[...]}.
 */
function summarizeBlocks(perBlock) {
  if (!Array.isArray(perBlock)) return { rows: [], failingTypes: [] };
  const byType = new Map();
  perBlock.forEach((b) => {
    const t = b.block_type || 'unknown';
    if (!byType.has(t)) byType.set(t, { block_type: t, passed: 0, failed: 0, escalated: 0 });
    const row = byType.get(t);
    if (b.status === 'failed') row.failed += 1;
    else if (b.status === 'passed') row.passed += 1;
    if (b.escalation_marker) row.escalated += 1;
  });
  const rows = Array.from(byType.values()).sort((a, b) => a.block_type.localeCompare(b.block_type));
  const failingTypes = rows.filter((r) => r.failed > 0).map((r) => r.block_type);
  return { rows, failingTypes };
}

/**
 * A row is "failing" (picker-eligible) when it failed validation OR was
 * escalated. ``status`` is only ever ``passed``/``failed`` (see the aggregator's
 * ``_build_per_block_results``); escalation is carried by ``escalation_marker``.
 * We accept a literal ``escalated`` status too, defensively.
 */
function isFailingRow(r) {
  return r && (r.status === 'failed' || r.status === 'escalated' || !!r.escalation_marker);
}

/** Failing check codes a per-block row carries (best-effort, may be empty). */
function failingCodes(row) {
  const codes = [];
  if (Array.isArray(row.gate_chain)) {
    row.gate_chain.forEach((g) => {
      if (g && g.passed === false && g.gate_id) codes.push(g.gate_id);
    });
  }
  if (row.escalation_marker) codes.push(`escalated:${row.escalation_marker}`);
  return codes;
}

/**
 * I4 stage-2 — render the per-page / per-block rewrite PICKER into `box`.
 *
 * The operator picks exactly which failing blocks (or whole pages) to
 * re-author. Grouping: one <details class="page-group"> per distinct page_id
 * (sorted); rows with a null/missing page_id fall under an "Unassigned" group
 * whose blocks are only individually selectable (no whole-page checkbox). Each
 * group's <summary> carries a whole-page checkbox (maps to the ``pages`` scope)
 * + the failing count; inside, a labelled checkbox per failing block (maps to
 * the ``block_ids`` scope).
 *
 * The JS is deliberately dumb: it gathers the RAW checked values and posts them
 * as arrays. ALL selection composition + subsumption dedup lives server-side
 * (run_service._normalize_blocks_param). The checked/disabled mechanics (a
 * checked whole-page checkbox checks+disables its children) are a UI
 * convenience only — the backend is the safety net, so we never dedup here.
 *
 * Unknown-token validation intentionally happens at rewrite RUN time (a loud
 * phase failure), NOT at enqueue — an operator typo can only arise from a stale
 * report, and the rewrite handler validates every id/page against the real
 * outline block set.
 */
function renderRewritePicker(shell, box, perBlock, courseName) {
  const failing = (Array.isArray(perBlock) ? perBlock : []).filter(isFailingRow);
  if (!failing.length) return;

  // Group by page_id; null/missing -> the special "Unassigned" bucket.
  const UNASSIGNED = '\u0000unassigned';
  const groups = new Map();
  failing.forEach((r) => {
    const key = r.page_id || UNASSIGNED;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  });
  const pageKeys = Array.from(groups.keys()).sort((a, b) => {
    if (a === UNASSIGNED) return 1;   // Unassigned sinks to the bottom
    if (b === UNASSIGNED) return -1;
    return a.localeCompare(b);
  });

  const picker = el('div', { class: 'rewrite-picker' });
  picker.appendChild(el('h3', { text: 'Choose what to rewrite' }));
  picker.appendChild(el('p', {
    class: 'picker-intro muted',
    text: 'Select individual failing blocks, or tick a whole page to re-author everything on it. Only the selected content is regenerated.',
  }));

  const pageBoxes = [];
  const blockBoxes = [];
  let summaryBar; let cta;

  function refresh() {
    const pagesSel = pageBoxes.filter((b) => b.checked).length;
    // Individually-selected blocks = checked AND enabled (a checked+disabled
    // child is implied by its page, counted under the page, not on its own).
    const blocksSel = blockBoxes.filter((b) => b.checked && !b.disabled).length;
    const anySel = pagesSel > 0 || blockBoxes.some((b) => b.checked);
    summaryBar.textContent = anySel
      ? `${blocksSel} block${blocksSel === 1 ? '' : 's'} + ${pagesSel} whole page${pagesSel === 1 ? '' : 's'} selected`
      : 'Nothing selected yet.';
    cta.disabled = !anySel;
  }

  pageKeys.forEach((key) => {
    const rows = groups.get(key);
    const isUnassigned = key === UNASSIGNED;
    const details = el('details', { class: 'page-group' });
    const summary = el('summary', { class: 'page-group-summary' });
    const childBoxes = [];

    // Whole-page checkbox (module/page scope) — omitted for the Unassigned
    // bucket (those blocks have no page to scope).
    let pageCheck = null;
    if (!isUnassigned) {
      const pcId = uid('pg');
      pageCheck = el('input', { type: 'checkbox', id: pcId, class: 'page-check', value: key });
      pageBoxes.push(pageCheck);
      // Toggling the checkbox must NOT open/close the disclosure.
      pageCheck.addEventListener('click', (ev) => ev.stopPropagation());
      pageCheck.addEventListener('change', () => {
        childBoxes.forEach((cb) => {
          if (pageCheck.checked) { cb.checked = true; cb.disabled = true; }
          else { cb.disabled = false; }
        });
        pageCheck.indeterminate = false;
        refresh();
      });
      summary.appendChild(pageCheck);
      summary.appendChild(el('label', { for: pcId, class: 'page-group-label' }, [
        el('code', { class: 'picker-page-id', text: key }),
      ]));
    } else {
      summary.appendChild(el('span', { class: 'page-group-label', text: 'Unassigned blocks' }));
    }
    summary.appendChild(el('span', {
      class: 'page-group-count muted',
      text: `${rows.length} failing`,
    }));
    details.appendChild(summary);

    const fieldset = el('fieldset', { class: 'block-fieldset' });
    const legendText = isUnassigned
      ? 'Failing blocks without a page'
      : `Failing blocks on ${key}`;
    fieldset.appendChild(el('legend', { class: 'visually-hidden', text: legendText }));

    rows.forEach((r) => {
      const cbId = uid('blk');
      const cb = el('input', { type: 'checkbox', id: cbId, class: 'block-check', value: r.block_id || '' });
      childBoxes.push(cb);
      blockBoxes.push(cb);
      cb.addEventListener('change', () => {
        if (pageCheck) {
          const n = childBoxes.filter((c) => c.checked).length;
          pageCheck.indeterminate = n > 0 && n < childBoxes.length;
        }
        refresh();
      });
      const label = el('label', { for: cbId, class: 'block-choice' }, [
        el('span', { class: 'block-choice-type', text: r.block_type || 'block' }),
        el('code', { class: 'block-choice-id', text: r.block_id || '(no id)' }),
      ]);
      const codes = failingCodes(r);
      if (codes.length) {
        label.appendChild(el('span', { class: 'block-choice-codes', text: codes.join(', ') }));
      }
      const row = el('div', { class: 'block-choice-row' }, [cb, label]);
      fieldset.appendChild(row);
    });
    details.appendChild(fieldset);
    picker.appendChild(details);
  });

  // Selection summary (polite live region) + the single CTA.
  const bar = el('div', { class: 'picker-actions' });
  summaryBar = el('p', { class: 'picker-summary', 'aria-live': 'polite', role: 'status' });
  cta = el('button', { type: 'button', class: 'btn primary', text: 'Rewrite selected', disabled: true });
  cta.addEventListener('click', () => {
    const block_ids = blockBoxes.filter((b) => b.checked).map((b) => b.value).filter(Boolean);
    const pages = pageBoxes.filter((b) => b.checked).map((b) => b.value).filter(Boolean);
    if (!block_ids.length && !pages.length) return;
    const parts = [];
    if (block_ids.length) parts.push(`${block_ids.length} block(s)`);
    if (pages.length) parts.push(`${pages.length} whole page(s)`);
    // Reuse the confirm-before-enqueue flow (mirrors affordanceBtn / the coarse
    // affordances). enqueuePhaseRun toasts on failure via the shared api path.
    if (!window.confirm(`Rewrite the selected ${parts.join(' + ')}?\n\nStart this now?`)) return;
    cta.disabled = true;
    enqueuePhaseRun(shell, {
      workflow: 'courseforge_rewrite',
      phase: 'content_generation_rewrite',
      course_name: courseName,
      options: { block_ids, pages },
    }).finally(() => refresh());
  });
  bar.appendChild(summaryBar);
  bar.appendChild(cta);
  picker.appendChild(bar);

  box.appendChild(picker);
  refresh();
}

/* ----------------------------------------------------------- run log view */

async function renderRunLog(shell, runId) {
  shell.crumbs([
    { text: 'Library', href: '#/library' },
    { text: 'Create course', href: '#/create' },
    { text: 'Build progress', href: `#/create/${encodeURIComponent(runId)}` },
    { text: 'Build log' },
  ]);
  const v = clear(shell.view());
  shell.setBusy(true);
  shell.setStatus('Loading the build log.');

  v.appendChild(el('h1', { text: 'Build log' }));

  let record;
  try {
    record = await api(`/api/runs/${encodeURIComponent(runId)}`);
  } catch (e) {
    shell.setBusy(false);
    v.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    return;
  }

  if (record.error) {
    v.appendChild(el('div', { class: 'final-box' }, [
      el('h2', { text: 'What went wrong' }),
      el('p', { class: 'error', role: 'alert', text: record.error }),
    ]));
  }

  const pre = el('pre', { class: 'run-log', tabindex: '0', 'aria-label': 'Build log output' });
  v.appendChild(pre);
  v.appendChild(el('div', { class: 'wizard-nav' }, [
    el('a', { class: 'btn', href: `#/create/${encodeURIComponent(runId)}`, text: 'Back to progress' }),
    el('a', { class: 'btn', href: '#/create', text: 'Start over' }),
  ]));
  shell.setBusy(false);

  // The WS replays the full log (from offset 0) then closes on a terminal run,
  // so it doubles as a log fetch without a new REST endpoint.
  const ws = openRunSocket(runId, {
    onLine: (line) => { pre.textContent += line + '\n'; },
    onStatus: () => { shell.setStatus('Build log loaded.'); },
    onError: (msg) => { pre.textContent += `\n[connection: ${msg}]\n`; },
  });
  return () => ws.close();
}

/* ------------------------------------------------------------ ws helper */

function openRunSocket(runId, handlers) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws/runs/${encodeURIComponent(runId)}`;
  let closed = false;
  let sock;
  try {
    sock = new WebSocket(url);
  } catch (e) {
    handlers.onError(String(e && e.message ? e.message : e));
    return { close() {} };
  }
  sock.addEventListener('message', (ev) => {
    let frame;
    try { frame = JSON.parse(ev.data); } catch (_) { return; }
    if (frame.type === 'line') handlers.onLine(frame.line || '');
    else if (frame.type === 'status') handlers.onStatus(frame);
    else if (frame.type === 'error') handlers.onError(frame.error || 'unknown error');
  });
  sock.addEventListener('error', () => { if (!closed) handlers.onError('connection error'); });
  return {
    close() { closed = true; try { sock.close(); } catch (_) { /* ignore */ } },
  };
}

/* --------------------------------------------------------------- helpers */

async function fetchSettingsSummary() {
  // Studio-scoped settings: model_routing carries per-task provider/model and
  // the courseforge two-pass flag drives whether the Validate/Rewrite tier is
  // active. The flow tree reads global (fallback) + the two authoring tiers.
  const doc = await api('/api/settings/studio');
  const r = doc.model_routing || {};
  const g = r.global || {};
  const flags = doc.flags || {};
  // Two-pass may be expressed as a flag or, defensively, a string env value.
  const twoPassRaw = flags.COURSEFORGE_TWO_PASS;
  const twoPass = twoPassRaw === true || twoPassRaw === 'true' || twoPassRaw === 1;
  return {
    mode: g.mode,
    provider: g.provider,
    model: g.model,
    routing: r,
    twoPass,
  };
}

async function fetchPhaseList() {
  const data = await api('/api/workflows');
  const wf = (data.workflows || []).find((w) => w.name === WORKFLOW);
  if (!wf || !Array.isArray(wf.phases)) return [];
  return wf.phases;
}

/**
 * Best-effort fetch of the per-phase median durations for the honest ETA.
 * Returns `{durations:{}, source:null}` on ANY failure (the sibling endpoint may
 * not exist yet); the console then renders "estimating…" rather than a lie.
 */
async function fetchPhaseDurations(workflow) {
  try {
    const data = await api(`/api/runs/phase-durations?workflow=${encodeURIComponent(workflow)}`);
    return {
      durations: (data && typeof data.durations === 'object' && data.durations) || {},
      source: (data && typeof data.source === 'string' && data.source) || null,
    };
  } catch (_) {
    return { durations: {}, source: null };
  }
}

/**
 * Host-owned cancel: POST /api/runs/{id}/cancel. The endpoint now returns HTTP
 * 202 `{status:"cancel_requested"}` (the terminal `cancelled` arrives later via
 * the WS status frame); we tolerate BOTH the legacy 200 and the new 202. On
 * either, flip the console's cancel button to the two-stage "Cancelling…" state
 * — we never claim an instant stop.
 */
async function requestCancel(shell, runId, progress) {
  try {
    // apiJSON throws on non-2xx; 202 is 2xx so it returns the body normally.
    const resp = await apiJSON(`/api/runs/${encodeURIComponent(runId)}/cancel`, 'POST', {});
    progress.setCancelling();
    const st = resp && resp.status;
    if (st === 'cancelled' || st === 'interrupted') {
      // A synchronous terminal (some backends resolve immediately) — finalize now.
      progress.onStatus(st);
    }
    shell.setStatus('Cancelling the build. It will stop after the current step finishes.');
    toast('Cancelling…', 'The build will stop after the current step.', 'info');
  } catch (e) {
    toastErr(e, 'Could not cancel');
    shell.setStatus(`Could not cancel: ${shell.errText(e)}`);
  }
}

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/* fmtElapsed was retired here: the elapsed timer is now owned by the shared
 * elapsed.js (via run-progress.js), which carries the identical formatter. */

function titleCase(name) {
  return String(name).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Best-effort course-name → viewer slug (lowercase, hyphenated). */
function courseSlug(name) {
  return String(name).trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}
