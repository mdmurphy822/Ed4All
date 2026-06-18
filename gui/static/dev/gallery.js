/* Component gallery — renders EVERY shared component in EVERY state.
 *
 * The visual-regression + WCAG fixture for the shared component kit. Imports
 * every component module and renders all five phase states, all pill/gate
 * variants, ring sizes, the empty state, both cards, the field row, and the
 * timeline bar. PURE PRESENTATION: this page calls no API — the kit is data-fed.
 *
 * The single role="status" aria-live="polite" region (#gallery-live in the
 * shell) is the a11y truth surface the kit demonstrates: a "Replay phase
 * sweep" control walks the timeline and announces phase transitions there
 * (polite, debounced — never per-second chatter).
 */

import { $, el, clear } from '/shared/dom.js';
import { progressRing } from '/shared/components/ring.js';
import { phaseRow } from '/shared/components/phase-row.js';
import { phaseTimeline } from '/shared/components/phase-timeline.js';
import { elapsed } from '/shared/components/elapsed.js';
import { statusPill, PILL_KINDS } from '/shared/components/pill.js';
import { gateChip, GATE_RESULTS } from '/shared/components/gate-chip.js';
import { courseCard, runCard } from '/shared/components/card.js';
import { emptyState } from '/shared/components/empty-state.js';
import { fieldRow } from '/shared/components/field-row.js';
import { timelineBar } from '/shared/components/timeline-bar.js';
import { runProgressConsole } from '/shared/components/run-progress.js';

const PHASE_STATES = ['pending', 'running', 'done', 'skipped', 'failed'];

function section(title, blurb, body) {
  return el('section', { class: 'gallery-section' }, [
    el('h2', { text: title }),
    blurb ? el('p', { text: blurb }) : null,
    body,
  ]);
}

function cell(caption, node) {
  return el('div', { class: 'gallery-cell' }, [node, el('span', { class: 'cell-caption', text: caption })]);
}

function row(cells) {
  return el('div', { class: 'gallery-row' }, cells);
}

/* ---- rings: every state x three sizes ---- */
function ringSection() {
  const cells = [];
  [20, 28, 48].forEach((size) => {
    PHASE_STATES.forEach((state) => {
      const pct = state === 'running' ? 60 : (state === 'done' ? 100 : 0);
      cells.push(cell(`${state} · ${size}px`, progressRing({ percent: pct, state, size, label: state })));
    });
  });
  return section('Progress ring', 'Five states at three sizes. The ring is aria-hidden decoration; truth is a label.', row(cells));
}

/* ---- pills: run + answer + severity, every variant ---- */
function pillSection() {
  const cells = [];
  Object.keys(PILL_KINDS.run).forEach((k) => cells.push(cell(`run:${k}`, statusPill(`run:${k}`))));
  Object.keys(PILL_KINDS.answer).forEach((k) => cells.push(cell(`answer:${k}`, statusPill(`answer:${k}`))));
  Object.keys(PILL_KINDS.severity).forEach((k) => cells.push(cell(`severity:${k}`, statusPill(`severity:${k}`))));
  return section('Status pill', 'One source for run statuses, learner answer statuses, and gate severities — glyph + text, never color-only.', row(cells));
}

/* ---- gate chips: every result ---- */
function gateChipSection() {
  const cells = Object.keys(GATE_RESULTS).map((r) =>
    cell(r, gateChip({ result: r, gate_id: 'curie_anchoring' })));
  return section('Gate chip', 'Icon + visually-hidden text label (never color-only).', row(cells));
}

/* ---- elapsed timer ---- */
function elapsedSection() {
  const live = elapsed({ startMs: Date.now() - 73000 });
  const frozen = elapsed({ startMs: Date.now() - 1700000, autostart: false });
  frozen.freeze(1700000);
  return section('Elapsed timer', 'The single canonical timer (tabular-nums, aria-hidden — no per-second screen-reader chatter).',
    row([cell('live', live.el), cell('frozen (finished)', frozen.el)]));
}

/* ---- phase row: each of the five states standalone ---- */
function phaseRowSection() {
  const ol = el('ol', { class: 'phase-checklist', 'aria-label': 'Phase row states' });
  PHASE_STATES.forEach((state) => {
    const r = phaseRow({ name: `phase_${state}`, label: `Phase (${state})`, state });
    if (state === 'running') r.setTaskProgress(23, 50);
    if (state === 'failed') { r.addGate({ result: 'fail', gate_id: 'curie_anchoring' }); }
    if (state === 'done') { r.addGate({ result: 'pass', gate_id: 'source_refs' }); r.addGate({ result: 'warn', gate_id: 'co_terminal_alignment' }); }
    ol.appendChild(r.el);
  });
  return section('Phase row', 'Each of the five states, with task progress + gate chips.', ol);
}

/* ---- phase timeline: a full ordered checklist, sequential states ---- */
function phaseTimelineSection() {
  const tl = phaseTimeline([
    { name: 'dart_conversion', label: 'Convert textbook to accessible HTML' },
    { name: 'staging', label: 'Stage source files' },
    { name: 'content_generation', label: 'Generate course content' },
    { name: 'packaging', label: 'Package course' },
    { name: 'trainforge_assessment', label: 'Generate assessments', optional: true },
  ]);
  tl.setState('dart_conversion', 'done');
  tl.setState('staging', 'done');
  tl.setState('content_generation', 'running');
  tl.setTaskProgress('content_generation', 23, 50);
  tl.addGate('content_generation', { result: 'pass', gate_id: 'source_refs' });
  return section('Phase timeline', 'The ordered checklist container (the live Build Console list).', tl.el);
}

/* ---- cards ---- */
function cardSection() {
  const grid = el('div', { class: 'gallery-grid' }, [
    courseCard({ title: 'Algebra (OpenStax)', href: '#/viewer/algebra', meta: '24 pages · 18.2 MB', askReady: true }),
    courseCard({ title: 'Physics 101 (no link)', meta: '12 pages · 9.0 MB' }),
    runCard({ title: 'PHYS_101', status: 'running', href: '#/build/GUI-1', meta: 'textbook_to_course · building' }),
    runCard({ title: 'BIO_201', status: 'completed', href: '#/build/GUI-2', meta: 'textbook_to_course · 28m' }),
    runCard({ title: 'CHEM_101', status: 'failed', href: '#/build/GUI-3', meta: 'textbook_to_course · 6m' }),
  ]);
  return section('Cards', 'courseCard() + runCard() for the dashboard zones.', grid);
}

/* ---- empty state ---- */
function emptyStateSection() {
  return section('Empty state', 'Message + a required CTA slot (no dead ends).',
    emptyState({
      title: 'No courses yet',
      message: 'Upload a textbook to build your first course.',
      cta: { label: 'Build your first course', href: '#/create' },
    }));
}

/* ---- field row ---- */
function fieldRowSection() {
  const form = el('form', { class: 'gallery-form', novalidate: true }, [
    fieldRow({ label: 'Course name', help: 'Letters, numbers, “_” and “-” only (e.g. PHYS_101).', name: 'course_name', required: true }).el,
    fieldRow({
      label: 'Outline course model', type: 'select', name: 'outline_provider',
      help: 'The AI provider that drafts the course outline.',
      options: [{ value: 'local', label: 'Ollama (local)', selected: true }, { value: 'anthropic', label: 'Anthropic (Claude)' }],
    }).el,
    fieldRow({ label: 'Notes', type: 'textarea', name: 'notes', help: 'Optional build notes.' }).el,
  ]);
  return section('Field row', 'Label-paired input rows ({label, help}) — the jargon-fix primitive.', form);
}

/* ---- timeline bar ---- */
function timelineBarSection() {
  return section('Timeline bar', 'A stacked horizontal bar of run-history phase durations (aria-hidden bar + a semantic legend).',
    timelineBar([
      { name: 'dart_conversion', label: 'Convert', duration_ms: 320000, state: 'done' },
      { name: 'content_generation', label: 'Generate content', duration_ms: 1080000, state: 'done' },
      { name: 'packaging', label: 'Package', duration_ms: 60000, state: 'done' },
      { name: 'trainforge_assessment', label: 'Assessments', duration_ms: 140000, state: 'skipped' },
    ]));
}

/* ---- the "replay phase sweep" demonstrating the single live region ---- */
function liveDemoSection() {
  const tl = phaseTimeline([
    { name: 'a', label: 'Convert textbook' },
    { name: 'b', label: 'Generate content' },
    { name: 'c', label: 'Package course' },
  ]);
  const btn = el('button', { type: 'button', class: 'btn primary', text: 'Replay phase sweep' });
  const live = $('#gallery-live');
  btn.addEventListener('click', () => {
    const order = ['a', 'b', 'c'];
    let i = 0;
    tl.startRunning();
    if (live) live.textContent = 'Converting textbook…';
    const step = () => {
      if (i >= order.length) { if (live) live.textContent = 'Build complete.'; return; }
      const next = tl.markCompleted(order[i], 'done');
      // Debounced phase-transition announcement (NOT per-second).
      if (live && next) live.textContent = `${tl.get(next)?.el.querySelector('.phase-label')?.textContent || next}…`;
      i += 1;
      setTimeout(step, 900);
    };
    setTimeout(step, 900);
  });
  return section('Live region demo', 'One role="status" aria-live="polite" region (in the shell) announces phase transitions, debounced — never per-second.',
    el('div', {}, [el('div', { class: 'gallery-row' }, [btn]), tl.el]));
}

/* ---- the assembled Build Console driven by a SCRIPTED line replay ----
 * The WCAG fixture for run-progress.js: it exercises the real console exactly as
 * the WS hosts (create.js / app.js) drive it — ISO-prefixed [phase] + [progress]
 * lines (which carry Tier-1 per-phase timing) replayed on a "Replay build" tap.
 * The console reuses the shell's single #gallery-live region (no second live
 * region on the page). PURE PRESENTATION: this calls no API. */
function buildConsoleSection() {
  const buildConsole = runProgressConsole({
    phases: [
      { name: 'dart_conversion', label: 'Convert textbook to accessible HTML' },
      { name: 'staging', label: 'Stage source files' },
      { name: 'content_generation', label: 'Generate course content' },
      { name: 'packaging', label: 'Package course' },
      { name: 'trainforge_assessment', label: 'Generate assessments', optional: true },
    ],
    startMs: Date.now(),
    liveRegion: $('#gallery-live') || undefined,
  });
  // A scripted ISO-stamped line stream (the shape run_service appends). The ISO
  // timestamps make the Tier-1 per-phase timing observable.
  const t0 = Date.now();
  const iso = (offsetMs) => new Date(t0 + offsetMs).toISOString();
  const script = [
    `[${iso(0)}] [progress] dart_conversion 1/3`,
    `[${iso(800)}] [progress] dart_conversion 3/3`,
    `[${iso(1200)}] [phase] dart_conversion done — Convert textbook to accessible HTML`,
    `[${iso(1800)}] [phase] staging done — Stage source files`,
    `[${iso(2200)}] [progress] content_generation 12/50`,
    `[${iso(3000)}] [progress] content_generation 50/50`,
    `[${iso(3400)}] [phase] content_generation done — Generate course content`,
    `[${iso(3900)}] [phase] packaging done — Package course`,
    `[${iso(4200)}] [phase] trainforge_assessment skipped — Generate assessments`,
  ];
  const btn = el('button', { type: 'button', class: 'btn primary', text: 'Replay build' });
  btn.addEventListener('click', () => {
    buildConsole.startFirstRunning();
    let i = 0;
    const step = () => {
      if (i >= script.length) { buildConsole.onStatus('completed'); return; }
      buildConsole.onLine(script[i]);
      i += 1;
      setTimeout(step, 700);
    };
    setTimeout(step, 700);
  });
  return section('Build Console', 'The assembled run-progress.js console — rings + elapsed + the single narrative line — driven by a scripted ISO-stamped [phase]/[progress] replay (Tier-1 timing).',
    el('div', {}, [el('div', { class: 'gallery-row' }, [btn]), buildConsole.el]));
}

function render() {
  const main = clear($('#gallery-main'));
  [
    ringSection(),
    pillSection(),
    gateChipSection(),
    elapsedSection(),
    phaseRowSection(),
    phaseTimelineSection(),
    cardSection(),
    emptyStateSection(),
    fieldRowSection(),
    timelineBarSection(),
    liveDemoSection(),
    buildConsoleSection(),
  ].forEach((s) => main.appendChild(s));
}

render();
