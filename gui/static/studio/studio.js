/* Ed4All Studio — end-user Library + IMSCC course viewer (vanilla ES module).
 *
 * Routes (shared hash router):
 *   #/library                  → course card grid (GET /api/library)
 *   #/viewer/<slug>            → manifest tree + content pane + pager
 *   #/viewer/<slug>/<item>     → same, with <item> (resource id) shown
 *
 * The content pane is a SANDBOXED iframe (sandbox="allow-scripts"; NO
 * allow-same-origin) carrying the sanitised page HTML via srcdoc, with the
 * viewer-shim injected so the inert generated components re-animate inside the
 * sandbox without reaching the parent.
 */

import { api, ApiError } from '/shared/api.js';
import { $, el, clear, uid } from '/shared/dom.js';
import { toast, toastErr } from '/shared/toast.js';
import { createRouter } from '/shared/router.js';
import { createAskDrawer } from '/studio/drawer.js';
import { renderCreate } from '/studio/create.js';
import { renderSettings } from '/studio/settings.js';
import { renderRunHistory } from '/studio/run-history.js';
import { renderDashboard } from '/studio/dashboard.js';
import { initPersonaSwitcher } from '/studio/persona-switcher.js';

const view = () => $('#view');
const statusEl = () => $('#status');
const crumbsEl = () => $('#crumbs');

function setStatus(msg) { const s = statusEl(); if (s) s.textContent = msg || ''; }

function setBusy(on) {
  const v = view();
  if (v) v.setAttribute('aria-busy', on ? 'true' : 'false');
}

function crumbs(parts) {
  const nav = clear(crumbsEl());
  parts.forEach((p, i) => {
    if (i) nav.appendChild(el('span', { class: 'sep', text: '/', 'aria-hidden': 'true' }));
    nav.appendChild(p.href ? el('a', { href: p.href, text: p.text }) : el('span', { text: p.text }));
  });
}

/* ===================================================================== */
/* Library                                                               */
/* ===================================================================== */

async function renderLibrary() {
  crumbs([{ text: 'Library' }]);
  const v = clear(view());
  v.appendChild(el('h1', { text: 'Course Library' }));
  v.appendChild(el('p', { class: 'loading', text: 'Loading courses…' }));
  setBusy(true);
  setStatus('Loading the course library.');

  let cards;
  try {
    cards = await api('/api/library');
  } catch (e) {
    clear(v).appendChild(el('h1', { text: 'Course Library' }));
    v.appendChild(el('p', { class: 'error', text: errText(e) }));
    setBusy(false);
    setStatus('Failed to load the course library.');
    toastErr(e, 'Library load failed');
    return;
  }

  clear(v).appendChild(el('h1', { text: 'Course Library' }));
  if (!Array.isArray(cards) || cards.length === 0) {
    v.appendChild(el('p', { class: 'muted', text: 'No archived courses are available to view yet.' }));
    setBusy(false);
    setStatus('No courses available.');
    return;
  }

  const list = el('ul', { class: 'cards' });
  cards.forEach((c) => {
    const metaParts = [`${c.page_count} page${c.page_count === 1 ? '' : 's'}`];
    if (typeof c.disk_bytes === 'number') metaParts.push(humanSize(c.disk_bytes));
    const statusBadge = certificationBadge(c.course_status);
    const li = el('li', { class: 'card-li' }, [
      el('a', { class: 'card', href: `#/viewer/${encodeURIComponent(c.slug)}` }, [
        el('h2', { text: c.title || c.slug }),
        el('p', { class: 'meta', text: metaParts.join(' · ') }),
        // Certification badge (T1) + Ask-ready badge sit in a real text row so
        // the state is never conveyed by colour alone.
        (statusBadge || c.has_vector_index)
          ? el('p', { class: 'card-badges' }, [
            statusBadge,
            c.has_vector_index ? el('span', { class: 'badge', text: 'Ask-ready' }) : null,
          ])
          : null,
      ]),
      el('div', { class: 'card-actions' }, [
        el('button', {
          type: 'button',
          class: 'card-scorecard',
          'aria-label': `Quality report for ${c.title || c.slug}`,
          onclick: () => openScorecardDialog(c),
        }, ['Quality report']),
        el('button', {
          type: 'button',
          class: 'card-delete',
          'aria-label': `Delete ${c.title || c.slug}`,
          onclick: () => openDeleteDialog(c),
        }, ['Delete']),
      ]),
    ]);
    list.appendChild(li);
  });
  v.appendChild(list);
  setBusy(false);
  setStatus(`${cards.length} course${cards.length === 1 ? '' : 's'} available.`);
}

/** Human-readable byte size (mirrors LibV2.tools.libv2.remove.human_size). */
function humanSize(n) {
  if (!Number.isFinite(n) || n < 0) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/* ---- certification badge (T1) ---- */
/*
 * Map the 5-value governance course_status enum to a human label + a tone
 * class. The label is REAL TEXT (never colour-only), so the certification
 * state is accessible; the tone class only tints it. An absent / unknown
 * status returns null → no badge (an ungoverned course carries none).
 */
const CERTIFICATION = {
  certified_trainable: { label: 'Certified · Trainable', tone: 'positive' },
  certified_instructional: { label: 'Certified · Instructional', tone: 'positive' },
  certified_accessible: { label: 'Certified · Accessible', tone: 'positive' },
  non_certified_archive: { label: 'Archived (not certified)', tone: 'neutral' },
  failed: { label: 'Did not certify', tone: 'negative' },
};

function certificationBadge(status) {
  const spec = status && CERTIFICATION[status];
  if (!spec) return null;
  return el('span', {
    class: `badge status-badge status-${spec.tone}`,
    text: spec.label,
  });
}

/* ===================================================================== */
/* Destructive course-deletion dialog (D5)                               */
/* ===================================================================== */
/*
 * A proper modal: role=dialog + aria-modal, labelled by its title, focus
 * trapped inside, Esc cancels, the confirm button stays disabled until the
 * operator types the exact slug (standard destructive-confirm pattern). On
 * success the DELETE removes the card + toasts.
 */
function openDeleteDialog(card) {
  const slug = card.slug;
  const titleId = uid('del-title');
  const descId = uid('del-desc');
  const inputId = uid('del-input');

  const input = el('input', {
    id: inputId,
    type: 'text',
    autocomplete: 'off',
    autocapitalize: 'off',
    spellcheck: 'false',
    'aria-describedby': descId,
  });
  const confirmBtn = el('button', {
    type: 'button',
    class: 'btn primary danger',
    disabled: true,
  }, [`Delete ${slug}`]);
  const cancelBtn = el('button', { type: 'button', class: 'btn' }, ['Cancel']);
  const errP = el('p', { class: 'error', role: 'alert', hidden: true });

  // Enable the confirm button only on an exact slug match (trim whitespace).
  input.addEventListener('input', () => {
    confirmBtn.disabled = input.value.trim() !== slug;
  });

  const dialog = el('div', {
    class: 'modal-dialog',
    role: 'dialog',
    'aria-modal': 'true',
    'aria-labelledby': titleId,
  }, [
    el('h2', { id: titleId, text: 'Delete course' }),
    el('p', { id: descId }, [
      'This permanently deletes ',
      el('strong', { text: card.title || slug }),
      ` (${humanSize(card.disk_bytes || 0)}). There is no undo. To confirm, type the course id `,
      el('code', { text: slug }),
      ' below.',
    ]),
    el('div', { class: 'field' }, [
      el('label', { for: inputId, text: 'Course id' }),
      input,
    ]),
    errP,
    el('div', { class: 'modal-actions' }, [cancelBtn, confirmBtn]),
  ]);
  const overlay = el('div', { class: 'modal-overlay' }, [dialog]);

  const lastFocused = document.activeElement;
  let closed = false;
  function close() {
    if (closed) return;
    closed = true;
    document.removeEventListener('keydown', onKeydown, true);
    overlay.remove();
    if (lastFocused && typeof lastFocused.focus === 'function') lastFocused.focus();
  }

  function focusables() {
    return Array.from(
      dialog.querySelectorAll('a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'),
    ).filter((n) => n.offsetParent !== null || n === document.activeElement);
  }

  // Focus trap + Esc cancel (capture phase so it wins over view handlers).
  function onKeydown(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const items = focusables();
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  }

  cancelBtn.addEventListener('click', close);
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  confirmBtn.addEventListener('click', async () => {
    confirmBtn.disabled = true;
    cancelBtn.disabled = true;
    errP.hidden = true;
    try {
      await api(
        `/api/library/${encodeURIComponent(slug)}?confirm=${encodeURIComponent(slug)}`,
        { method: 'DELETE' },
      );
    } catch (e) {
      cancelBtn.disabled = false;
      confirmBtn.disabled = input.value.trim() !== slug;
      errP.textContent = errText(e);
      errP.hidden = false;
      return;
    }
    close();
    // Drop the card from the grid + toast (re-render keeps the live region honest).
    toast('Course deleted', `Removed ${card.title || slug}.`, 'success');
    renderLibrary();
  });

  document.addEventListener('keydown', onKeydown, true);
  document.body.appendChild(overlay);
  input.focus();
}

/* ===================================================================== */
/* Quality scorecard dialog (T2)                                         */
/* ===================================================================== */
/*
 * An accessible modal (role=dialog + aria-modal, labelled by its title, focus
 * trapped, Esc closes) that fetches GET /api/library/<slug>/scorecard and
 * renders the composed governance + eval sections. Each section is
 * present-if-available: an un-evaluated section shows an explicit "Not yet
 * evaluated." line rather than a fabricated number. The body region is an
 * aria-live="polite" container so the async load result is announced.
 */
function openScorecardDialog(card) {
  const slug = card.slug;
  const titleId = uid('sc-title');
  const bodyId = uid('sc-body');

  const closeBtn = el('button', { type: 'button', class: 'btn' }, ['Close']);
  const body = el('div', {
    id: bodyId,
    class: 'scorecard-body',
    'aria-live': 'polite',
    'aria-busy': 'true',
  }, [el('p', { class: 'loading', text: 'Loading the quality report…' })]);

  const dialog = el('div', {
    class: 'modal-dialog scorecard-dialog',
    role: 'dialog',
    'aria-modal': 'true',
    'aria-labelledby': titleId,
  }, [
    el('h2', { id: titleId, text: `Quality report — ${card.title || slug}` }),
    body,
    el('div', { class: 'modal-actions' }, [closeBtn]),
  ]);
  const overlay = el('div', { class: 'modal-overlay' }, [dialog]);

  const lastFocused = document.activeElement;
  let closed = false;
  function close() {
    if (closed) return;
    closed = true;
    document.removeEventListener('keydown', onKeydown, true);
    overlay.remove();
    if (lastFocused && typeof lastFocused.focus === 'function') lastFocused.focus();
  }

  function focusables() {
    return Array.from(
      dialog.querySelectorAll('a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])'),
    ).filter((n) => n.offsetParent !== null || n === document.activeElement);
  }

  function onKeydown(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key !== 'Tab') return;
    const items = focusables();
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  }

  closeBtn.addEventListener('click', close);
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', onKeydown, true);
  document.body.appendChild(overlay);
  closeBtn.focus();

  // Fetch + render (best-effort — a load failure shows an in-dialog error, the
  // dialog itself stays operable).
  (async () => {
    let sc;
    try {
      sc = await api(`/api/library/${encodeURIComponent(slug)}/scorecard`);
    } catch (e) {
      clear(body).appendChild(el('p', { class: 'error', role: 'alert', text: errText(e) }));
      body.setAttribute('aria-busy', 'false');
      return;
    }
    clear(body);
    renderScorecard(body, sc);
    body.setAttribute('aria-busy', 'false');
  })();
}

/** Render the composed scorecard object into the dialog body. */
function renderScorecard(body, sc) {
  const statusBadge = certificationBadge(sc && sc.course_status);
  body.appendChild(el('p', { class: 'sc-status' }, [
    el('span', { class: 'sc-status-label', text: 'Certification: ' }),
    statusBadge || el('span', { class: 'muted', text: 'not governed' }),
  ]));

  const sections = (sc && sc.sections) || {};
  renderScorecardSection(body, 'Retrieval evaluation', sections.retrieval_eval, renderRetrievalEval);
  renderScorecardSection(body, 'Refusal calibration', sections.refusal_calibration, renderRefusalCalibration);
  renderScorecardSection(body, 'Assessment quality', sections.assessment_quality, renderAssessmentQuality);
  renderScorecardSection(body, 'Objective coverage', sections.coverage_map, renderCoverageMap);
}

/** A titled scorecard section; renders the not-yet-evaluated marker when absent. */
function renderScorecardSection(parent, title, section, renderer) {
  const sec = el('section', { class: 'sc-section', 'aria-label': title }, [
    el('h3', { text: title }),
  ]);
  if (!section || section.available !== true) {
    const marker = (section && section.status) || 'not yet evaluated';
    sec.appendChild(el('p', { class: 'muted sc-not-evaluated', text: cap(marker) + '.' }));
  } else {
    renderer(sec, section);
  }
  parent.appendChild(sec);
}

function renderRetrievalEval(sec, section) {
  const meta = [];
  if (section.engine) meta.push(`Engine: ${section.engine}`);
  if (section.generated_at) meta.push(`Run: ${section.generated_at}`);
  if (meta.length) sec.appendChild(el('p', { class: 'meta', text: meta.join(' · ') }));

  const arms = section.arms || {};
  const armNames = Object.keys(arms);
  if (armNames.length) {
    const table = el('table', { class: 'sc-table' }, [
      el('thead', {}, [el('tr', {}, [
        el('th', { scope: 'col', text: 'Arm' }),
        el('th', { scope: 'col', text: 'Key-point coverage' }),
        el('th', { scope: 'col', text: 'Unsupported-claim rate' }),
        el('th', { scope: 'col', text: 'Latency p50 / p95' }),
      ])]),
      el('tbody', {}, armNames.map((name) => {
        const a = arms[name] || {};
        return el('tr', {}, [
          el('th', { scope: 'row', text: cap(name) }),
          el('td', { text: fmtMetric(a.key_point_coverage_rate) }),
          el('td', { text: fmtMetric(a.claim_level_unsupported_rate) }),
          el('td', { text: fmtLatency(a.latency_ms) }),
        ]);
      })),
    ]);
    sec.appendChild(table);
  }

  const rs = section.refusal_safety || {};
  if (typeof rs.answered_not_refused_rate !== 'undefined') {
    sec.appendChild(el('p', { class: 'sc-note', text:
      `Refusal-probe answered rate: ${fmtMetric(rs.answered_not_refused_rate)} (lower is safer).` }));
  }
}

function renderRefusalCalibration(sec, section) {
  const r = section.recommended || {};
  const dl = el('dl', { class: 'sc-dl' });
  addDef(dl, 'Recommended min top-score', fmtMetric(r.min_top_score));
  addDef(dl, 'Refusal precision', fmtMetric(r.refusal_precision));
  if (typeof r.answer_recall !== 'undefined') addDef(dl, 'Answer recall', fmtMetric(r.answer_recall));
  if (typeof r.refusal_recall !== 'undefined') addDef(dl, 'Refusal recall', fmtMetric(r.refusal_recall));
  sec.appendChild(dl);
}

function renderAssessmentQuality(sec, section) {
  const dl = el('dl', { class: 'sc-dl' });
  if (section.status) addDef(dl, 'Status', String(section.status));
  const pd = section.promotion_decision || {};
  if (pd.value) addDef(dl, 'Promotion decision', String(pd.value));
  const s = section.summary || {};
  if (typeof s.total_questions !== 'undefined') addDef(dl, 'Total questions', fmtMetric(s.total_questions));
  if (typeof s.answerable_rate !== 'undefined') addDef(dl, 'Answerable rate', fmtMetric(s.answerable_rate));
  if (typeof s.single_correct_rate !== 'undefined') addDef(dl, 'Single-correct rate', fmtMetric(s.single_correct_rate));
  if (typeof s.source_support_rate !== 'undefined') addDef(dl, 'Source-support rate', fmtMetric(s.source_support_rate));
  if (typeof s.bloom_alignment_rate !== 'undefined') addDef(dl, "Bloom alignment", fmtMetric(s.bloom_alignment_rate));
  sec.appendChild(dl);
}

function renderCoverageMap(sec, section) {
  const s = section.summary || {};
  const dl = el('dl', { class: 'sc-dl' });
  if (typeof s.objectives_with_chunks !== 'undefined') addDef(dl, 'Objectives with source chunks', fmtMetric(s.objectives_with_chunks));
  if (typeof s.objectives_with_questions !== 'undefined') addDef(dl, 'Objectives with questions', fmtMetric(s.objectives_with_questions));
  if (typeof s.objectives_with_training_pairs !== 'undefined') addDef(dl, 'Objectives with training pairs', fmtMetric(s.objectives_with_training_pairs));
  if (typeof s.orphan_chunks_count !== 'undefined') addDef(dl, 'Orphan chunks', fmtMetric(s.orphan_chunks_count));
  sec.appendChild(dl);
}

/* ---- scorecard formatting helpers ---- */
function addDef(dl, term, value) {
  dl.appendChild(el('dt', { text: term }));
  dl.appendChild(el('dd', { text: value }));
}

function cap(s) {
  const str = String(s || '');
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
}

/** Format a metric: null/undefined/"—" → em dash; a 0..1 fraction → 2 dp. */
function fmtMetric(v) {
  if (v === null || typeof v === 'undefined' || v === '—') return '—';
  if (typeof v === 'number' && Number.isFinite(v)) {
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  }
  return String(v);
}

function fmtLatency(lat) {
  if (!lat || typeof lat !== 'object') return '—';
  const p50 = typeof lat.p50 === 'number' ? Math.round(lat.p50) : null;
  const p95 = typeof lat.p95 === 'number' ? Math.round(lat.p95) : null;
  if (p50 === null && p95 === null) return '—';
  return `${p50 === null ? '—' : p50} / ${p95 === null ? '—' : p95} ms`;
}

/* ===================================================================== */
/* Viewer                                                                */
/* ===================================================================== */

async function renderViewer(segments) {
  const slug = segments[0];
  const wantItem = segments[1] || null;
  if (!slug) { location.hash = '#/library'; return; }

  crumbs([{ text: 'Library', href: '#/library' }, { text: slug }]);
  const v = clear(view());
  v.appendChild(el('p', { class: 'loading', text: 'Loading course…' }));
  setBusy(true);
  setStatus('Loading the course outline.');

  let manifest;
  try {
    manifest = await api(`/api/courses/${encodeURIComponent(slug)}/manifest`);
  } catch (e) {
    clear(v).appendChild(el('p', { class: 'error', text: errText(e) }));
    setBusy(false);
    setStatus('Failed to load the course.');
    toastErr(e, 'Course load failed');
    return;
  }

  // Flatten manifest pages (depth-first) for the pager order.
  const pages = [];
  (function walk(items) {
    items.forEach((it) => {
      if (it.item && it.href) pages.push(it);
      if (it.children) walk(it.children);
    });
  })(manifest.items || []);

  crumbs([{ text: 'Library', href: '#/library' }, { text: manifest.title || slug }]);

  clear(v).appendChild(el('h1', { text: manifest.title || slug }));
  if (pages.length === 0) {
    v.appendChild(el('p', { class: 'muted', text: 'This course has no viewable pages.' }));
    setBusy(false);
    setStatus('No pages in this course.');
    return;
  }

  // Layout: tree (left) + content (right).
  const treePane = el('div', { class: 'tree-pane' }, [el('h2', { text: 'Contents' })]);
  const tree = buildTree(manifest.items || []);
  treePane.appendChild(tree.root);

  const pageTitle = el('span', { class: 'page-title', id: 'page-title', 'aria-live': 'polite' });
  const prevBtn = el('button', { type: 'button', text: '← Previous', 'aria-label': 'Previous page' });
  const nextBtn = el('button', { type: 'button', text: 'Next →', 'aria-label': 'Next page' });
  const pager = el('div', { class: 'pager' }, [prevBtn, pageTitle, nextBtn]);
  const frame = el('iframe', {
    class: 'frame',
    title: 'Course page content',
    sandbox: 'allow-scripts',
    referrerpolicy: 'no-referrer',
  });
  const contentPane = el('div', { class: 'content-pane' }, [pager, frame]);

  // Ask drawer (C2): docks to the right of the content pane. The loadCitation
  // callback routes a clicked citation to THIS content pane (in-context) with a
  // fragment scroll + temporary highlight, so answer + source show together.
  const drawer = createAskDrawer({
    slug,
    loadCitation: (itemPath, fragment) => loadCitation(itemPath, fragment),
    // Anchored original-source documents load by URL — the iframe honours the
    // #dart-<block> fragment so the view lands on the cited block.
    loadSourceDoc: (href) => {
      frame.removeAttribute('srcdoc');
      frame.src = href;
      setStatus('Showing the cited original source.');
    },
  });

  const layout = el('div', { class: 'viewer' }, [treePane, contentPane, drawer.root]);
  v.appendChild(layout);

  // Standing link to the accessible textbook (finding 4). The generated course
  // is a derivative; the original DART-converted accessible HTML is archived,
  // discoverable, and servable — surface it in the contents so a learner can
  // always reach the verbatim source. Best-effort: a load failure / empty
  // inventory / disabled toggle simply omits the section (never breaks the
  // viewer). The link loads the source doc INTO the content iframe by URL (the
  // page honours its own #dart-<block> fragments), matching loadSourceDoc.
  attachSourceMaterials(treePane, slug, (href) => {
    frame.removeAttribute('srcdoc');
    frame.src = href;
    setStatus('Showing the accessible textbook.');
  });

  let currentIdx = 0;
  const startItem = wantItem || pages[0].item;
  const startAt = pages.findIndex((p) => p.item === startItem);
  currentIdx = startAt >= 0 ? startAt : 0;

  async function loadPage(idx, { focusFrame = false, fragment = '' } = {}) {
    if (idx < 0 || idx >= pages.length) return;
    currentIdx = idx;
    const page = pages[idx];
    pageTitle.textContent = page.title || page.item;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === pages.length - 1;
    tree.select(page.item);
    // Reflect the page in the hash (without re-dispatching the route).
    const wanted = `#/viewer/${encodeURIComponent(slug)}/${encodeURIComponent(page.item)}`;
    if (location.hash !== wanted) history.replaceState(null, '', wanted);
    setStatus(`Showing ${page.title || page.item}.`);

    try {
      const resp = await fetch(`/api/courses/${encodeURIComponent(slug)}/page?item=${encodeURIComponent(page.item)}`);
      const html = await resp.text();
      if (!resp.ok) throw new ApiError(resp.status, 'page_failed', html.slice(0, 300));
      // The shim reads a fragment hint off <html data-cf-scroll-to="…"> and
      // scrolls + temporarily highlights inside the sandbox (the parent cannot
      // reach the iframe DOM under sandbox="allow-scripts").
      frame.srcdoc = injectShim(html, fragment);
    } catch (e) {
      frame.srcdoc = `<!DOCTYPE html><html lang="en"><body><p style="color:#8a1c1c;font-family:system-ui">Could not load this page: ${escapeHtml(errText(e))}</p></body></html>`;
      toastErr(e, 'Page load failed');
    }
    if (focusFrame) frame.focus();
  }

  /** Match a citation's item_path to a viewer page (by basename) and load it. */
  function loadCitation(itemPath, fragment) {
    // Exact href match FIRST across all pages; the basename fallback only
    // applies when no exact match exists anywhere — filename stems like
    // content_01.html repeat across weeks, so a combined OR-predicate would
    // let an earlier week's basename match shadow the exact page.
    const base = String(itemPath || '').split('/').pop();
    let idx = pages.findIndex((p) => p.href === itemPath);
    if (idx < 0) idx = pages.findIndex((p) => p.href && p.href.split('/').pop() === base);
    if (idx < 0) idx = pages.findIndex((p) => p.item === itemPath);
    if (idx >= 0) {
      loadPage(idx, { focusFrame: true, fragment });
      setStatus(`Showing the cited source: ${pages[idx].title || pages[idx].item}.`);
      return;
    }
    // No matching manifest page → fall back to the archived source viewer.
    const src = `/api/learn/source/${encodeURIComponent(slug)}?item_path=${encodeURIComponent(itemPath)}${fragment ? `#${encodeURIComponent(fragment)}` : ''}`;
    frame.removeAttribute('srcdoc');
    frame.src = src;
    setStatus('Showing the cited source page.');
  }

  prevBtn.addEventListener('click', () => loadPage(currentIdx - 1, { focusFrame: true }));
  nextBtn.addEventListener('click', () => loadPage(currentIdx + 1, { focusFrame: true }));
  tree.onSelect((item) => {
    const idx = pages.findIndex((p) => p.item === item);
    if (idx >= 0) loadPage(idx, { focusFrame: true });
  });

  setBusy(false);
  await loadPage(currentIdx);
  return () => drawer.destroy();
}

/* ---- ARIA tree (WAI-ARIA tree pattern, full keyboard) ---- */
function buildTree(items) {
  const root = el('ul', { role: 'tree', 'aria-label': 'Course contents' });
  /** @type {HTMLElement[]} */
  const treeitems = [];
  let selectListener = null;
  let selectedItem = null;

  function makeNode(node, level) {
    const hasChildren = node.children && node.children.length > 0;
    const li = el('li', {
      role: 'treeitem',
      'aria-level': String(level),
      tabindex: '-1',
    });
    if (hasChildren) li.setAttribute('aria-expanded', 'true');
    if (node.item) li.dataset.item = node.item;
    const label = el('span', { class: 'label' }, [
      hasChildren ? el('span', { class: 'twisty', 'aria-hidden': 'true', text: '▾ ' }) : null,
      el('span', { text: node.title || node.item || node.identifier }),
    ]);
    li.appendChild(label);
    treeitems.push(li);

    if (node.item) {
      li.addEventListener('click', (e) => { e.stopPropagation(); activate(li); });
    }

    if (hasChildren) {
      const group = el('ul', { role: 'group' });
      node.children.forEach((c) => group.appendChild(makeNode(c, level + 1)));
      li.appendChild(group);
    }
    return li;
  }

  function activate(li) {
    const item = li.dataset.item;
    if (!item) return;
    select(item);
    if (typeof selectListener === 'function') selectListener(item);
  }

  function select(item) {
    selectedItem = item;
    treeitems.forEach((t) => {
      const on = t.dataset.item === item;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
    });
    // Ensure at least one tabbable node even before a selection.
    if (!item && treeitems.length) treeitems[0].tabIndex = 0;
  }

  items.forEach((it) => root.appendChild(makeNode(it, 1)));
  if (treeitems.length) treeitems[0].tabIndex = 0;

  // Roving-tabindex keyboard nav across visible treeitems.
  root.addEventListener('keydown', (e) => {
    const focusable = treeitems.filter((t) => t.offsetParent !== null || t === document.activeElement);
    const idx = focusable.indexOf(document.activeElement);
    let next = -1;
    if (e.key === 'ArrowDown') next = Math.min(idx + 1, focusable.length - 1);
    else if (e.key === 'ArrowUp') next = Math.max(idx - 1, 0);
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = focusable.length - 1;
    else if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
      const cur = focusable[idx >= 0 ? idx : 0];
      if (cur) { e.preventDefault(); activate(cur); }
      return;
    } else return;
    e.preventDefault();
    if (next >= 0 && focusable[next]) {
      focusable.forEach((t) => { t.tabIndex = -1; });
      focusable[next].tabIndex = 0;
      focusable[next].focus();
    }
  });

  return {
    root,
    select,
    onSelect(fn) { selectListener = fn; },
  };
}

/* ---- source-materials (accessible textbook) standing link ---- */
/*
 * Fetch the per-course source inventory and, when an accessible DART doc is
 * archived + exposed, append a standing "Accessible textbook" section to the
 * contents pane. Each doc links to the learner source viewer
 * (/api/learn/source/<slug>?item_path=<doc>.html). Clicking loads it INTO the
 * content iframe (the served page keeps its own #dart-<block> anchors, so a
 * later citation deep-link still lands on the cited block). Fully best-effort:
 * a fetch failure, a disabled toggle, or an empty inventory leaves the contents
 * untouched — the textbook link is additive, never load-bearing.
 */
async function attachSourceMaterials(treePane, slug, onOpen) {
  let inv;
  try {
    inv = await api(`/api/courses/${encodeURIComponent(slug)}/source-materials`);
  } catch {
    return; // toggle off / unknown course / transient — silently omit.
  }
  if (!inv || inv.enabled === false) return;
  const docs = Array.isArray(inv.dart_docs) ? inv.dart_docs : [];
  if (docs.length === 0) return;

  const section = el('section', { class: 'source-materials', 'aria-label': 'Source materials' }, [
    el('h2', { text: 'Accessible textbook' }),
    el('p', { class: 'muted', text: 'The original source this course was built from.' }),
  ]);
  const list = el('ul', { class: 'source-docs' });
  docs.forEach((d) => {
    const itemPath = `${d.doc}.html`;
    const href = `/api/learn/source/${encodeURIComponent(slug)}?item_path=${encodeURIComponent(itemPath)}`;
    const link = el('a', {
      class: 'source-doc-link',
      href,
      // Load in-pane rather than navigating the whole app away.
      onclick: (e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return; // honour open-in-new
        e.preventDefault();
        if (typeof onOpen === 'function') onOpen(href);
      },
    }, [d.title || d.doc]);
    list.appendChild(el('li', {}, [link]));
  });
  section.appendChild(list);
  treePane.appendChild(section);
}

/* ---- helpers ---- */
function injectShim(html, fragment = '') {
  const tag = '<script src="/studio/viewer-shim.js"></scr' + 'ipt>';
  let out = html;
  // Stamp a scroll-to hint on <html> so the (sandboxed) shim can scroll +
  // highlight the cited passage; only the slug value (a-z0-9-) is interpolated.
  if (fragment) {
    const safe = String(fragment).replace(/[^a-zA-Z0-9_-]/g, '');
    if (safe) {
      if (/<html\b[^>]*>/i.test(out)) out = out.replace(/<html\b([^>]*)>/i, `<html$1 data-cf-scroll-to="${safe}">`);
      else out = `<html data-cf-scroll-to="${safe}">${out}</html>`;
    }
  }
  if (/<\/body>/i.test(out)) return out.replace(/<\/body>/i, tag + '</body>');
  return out + tag;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function errText(e) {
  if (e instanceof ApiError) return `${e.error}: ${e.detail || `HTTP ${e.status}`}`;
  return String((e && e.message) || e);
}

/* ===================================================================== */
/* Boot                                                                  */
/* ===================================================================== */

// The Create wizard + Settings routes share the same shell helpers (status
// live-region, crumbs, busy flag); pass them in so create.js / settings.js
// don't re-query the DOM or duplicate the toolkit.
const shell = {
  view,
  crumbs,
  setStatus,
  setBusy,
  errText,
  escapeHtml,
  // The single page-level polite live region (#status). The Build Console
  // announces into THIS so the page keeps exactly one live-truth region.
  statusRegion: statusEl,
};

const router = createRouter(
  {
    // Author Dashboard home (#/) — three zones + onboarding empty state.
    dashboard: () => renderDashboard(shell),
    library: renderLibrary,
    viewer: renderViewer,
    create: (segments, raw) => renderCreate(shell, segments, raw),
    // Build Console (#/build/<run_id>): the live-build progress view, reachable
    // from any dashboard run card — not only mid-wizard. Delegates to the same
    // create.js progress renderer (it branches on a run_id in segments[0]).
    build: (segments, raw) => renderCreate(shell, segments, raw),
    settings: (segments, raw) => renderSettings(shell, segments, raw),
    // Run History (Phase 4 §5.1(F)): reachable via the EXISTING studio router —
    // NO Phase-2 auth remount, NO route/auth change.
    runs: () => renderRunHistory(shell),
  },
  {
    // #/ (empty hash) lands on the Author Dashboard.
    defaultRoute: 'dashboard',
    onError(err) {
      clear(view()).appendChild(el('p', { class: 'error', text: errText(err) }));
      setBusy(false);
      toastErr(err, 'Studio route failed');
    },
  },
);

// Keep the persona switcher's active state honest as the hash route changes:
// Author is "current" whenever a Studio (non-Learner/Advanced) route is active,
// which is always here — the Author persona link points at #/ and stays current.
function syncPersonaCurrent() {
  document.querySelectorAll('.persona-switcher [data-persona]').forEach((a) => {
    if (a.dataset.persona === 'author') a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}

// Upgrade the (server-rendered) Advanced link with the graceful-401 explainer.
initPersonaSwitcher(document);
syncPersonaCurrent();

router.start();
