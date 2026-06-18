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
  const model = { step: 1, files: [], uploadId: null, courseName: '', weeks: '', advanced: {} };

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

  // Provider / model summary read from settings, with a link to settings.
  form.appendChild(providerSummary(summary, shell));

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

function providerSummary(summary, shell) {
  const box = el('div', { class: 'summary-box' });
  box.appendChild(el('h3', { text: 'AI provider' }));
  if (summary && summary.provider) {
    const model = summary.model ? ` · model ${summary.model}` : ' · default model';
    box.appendChild(el('p', { text: `Mode ${summary.mode || 'local'} · provider ${summary.provider}${model}.` }));
  } else {
    box.appendChild(el('p', { class: 'muted', text: 'Using your saved provider settings (local by default).' }));
  }
  box.appendChild(el('p', {}, [
    el('a', { href: '#/settings', text: 'Change AI provider settings' }),
  ]));
  return box;
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

  // Run id sits in the meta line (left of the console's elapsed timer).
  progress.el.querySelector('.run-progress-meta').insertBefore(
    el('span', {}, [
      el('span', { text: `Run ${runId}` }),
      el('span', { class: 'sep', 'aria-hidden': 'true', text: ' · ' }),
    ]),
    progress.elapsedEl,
  );
  v.appendChild(progress.el);

  // The failure panel + terminal CTAs render below the console.
  const finalBox = el('div', { class: 'final-box' });
  v.appendChild(finalBox);

  shell.setBusy(false);

  // The phase explicitly reported failed by a structured WS line (A6) wins over
  // the "last running" heuristic; the console tracks it for us.
  function finalize(status, errMsg) {
    clear(finalBox);
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
  const terminal = ['completed', 'failed', 'cancelled', 'interrupted'];
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
    return;
  }

  // Live stream. The WS carries {type:line} (the console parses the ISO-prefixed
  // "[phase]"/"[progress]" lines, including Tier-1 timing) and a terminal
  // {type:status}.
  progress.startFirstRunning();

  const ws = openRunSocket(runId, {
    onLine: (line) => progress.onLine(line),
    onStatus: (frame) => finalize(frame.status, frame.error),
    onError: (msg) => {
      progress.onError(msg);
      finalBox.appendChild(el('p', { class: 'error', role: 'alert', text: `Lost connection to the build: ${msg}` }));
    },
  });

  // Teardown: close the WS + stop the console timer when the route changes.
  return () => { progress.destroy(); ws.close(); };
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

/** Copy-to-clipboard button for the resume CLI hint (a real <button>). */
function copyHintBtn(text) {
  const btn = el('button', { type: 'button', class: 'btn', 'aria-label': 'Copy resume command to clipboard', text: 'Copy command' });
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
  // Studio-scoped settings: model_routing.global carries mode/provider/model.
  const doc = await api('/api/settings/studio');
  const g = (doc.model_routing && doc.model_routing.global) || {};
  return { mode: g.mode, provider: g.provider, model: g.model };
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
