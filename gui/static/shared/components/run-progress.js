/* runProgressConsole — the reusable Glass-Box Build Console. Vanilla ES module,
 * NO build step, PURE PRESENTATION.
 *
 * Assembles the Phase-1 component kit into ONE console that replaces the two
 * hand-rolled progress renderers (studio create.js::renderProgress + the
 * operator app.js live console). It owns:
 *
 *   - a phaseTimeline (real SVG progress rings replace the old static glyphs),
 *   - the single canonical `elapsed` timer (aria-hidden decoration),
 *   - a SINGLE role="status" aria-live="polite" "what's happening now" line that
 *     translates the current phase (+ task count, when present) into a sentence,
 *     DEBOUNCED — it announces phase TRANSITIONS, never per-second chatter
 *     (mirrors the verified learner aria-hidden elapsed pattern), and
 *   - a defensive log-line parser that drives all of the above from the WS
 *     `{type:"line"}` stream the host page owns.
 *
 * PURE PRESENTATION + HARD SECURITY CONTRACT: this console takes parsed
 * lines/data/phase descriptors as input and does NOT itself fetch — the host
 * page owns the WebSocket + every REST call. The kit can therefore be imported
 * by the locked-down learner bundle and structurally cannot pull a run-control
 * endpoint into it.
 *
 * --- the log-line grammar parsed (anything unrecognized is IGNORED) ---------
 * Lines arrive ISO-prefixed (run_service appends `[{now_iso()}] …`). The parser
 * handles, defensively:
 *
 *   [<iso>] [phase] <name> done            → markCompleted(name, 'done')
 *   [<iso>] [phase] <name> skipped         → markCompleted(name, 'skipped')
 *   [<iso>] [phase] <name> failed — <...>  → mark the phase failed (A6 signal)
 *   [<iso>] [progress] <name> <done>/<tot> → setTaskProgress(name, done, tot)
 *
 * The leading ISO timestamp (which the legacy create.js regex THREW AWAY) is
 * parsed to drive Tier-1 per-phase timing entirely GUI-side, ZERO backend
 * change: per-phase duration = t(this done line) − t(previous done line); the
 * first phase is anchored to startMs. Durations are exposed via getTimeline()
 * so the host (and a future ETA layer) can read them.
 *
 * When a running phase has no task count its ring stays indeterminate (animated,
 * no fill) — that is CORRECT for validator-only phases, not a bug.
 */

import { el } from '../dom.js';
import { phaseTimeline } from './phase-timeline.js';
import { elapsed as makeElapsed } from './elapsed.js';

/* Line grammar. The ISO prefix is captured (group 1) so we can drive Tier-1
 * timing; the legacy create.js regex discarded it. All tolerant of extra
 * leading text and trailing `— <label>` / `: <reason>` suffixes. */
const _PHASE_DONE_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[phase\]\s+(\S+)\s+(done|skipped)\b/;
const _PHASE_FAIL_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[phase\]\s+(\S+)\s+failed\b/;
const _PROGRESS_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[progress\]\s+(\S+)\s+(\d+)\s*\/\s*(\d+)/;

/** Parse the captured ISO prefix to epoch-ms; null if absent / unparseable. */
function _isoMs(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

/**
 * Build the Build Console.
 *
 * @param {Object} opts
 * @param {Array<{name,label,optional}>} opts.phases  ordered phase descriptors
 *        (friendly labels resolved by the host from /api/workflows — never
 *        hardcoded here).
 * @param {number} [opts.startMs=Date.now()]  epoch-ms the build started (the
 *        elapsed anchor + the first phase's Tier-1 timing anchor).
 * @param {string} [opts.ariaLabel="Course build steps"]  timeline list label.
 * @param {number} [opts.ringSize=28]
 * @param {string} [opts.elapsedPrefix="elapsed "]
 * @param {HTMLElement} [opts.liveRegion]  optional EXISTING single
 *        role="status" aria-live region owned by the host shell. When supplied,
 *        the console announces into it INSTEAD of minting its own — so a host
 *        that already owns one polite live region (the studio shell's #status)
 *        keeps EXACTLY ONE live region on the page (the §9 single-live-truth
 *        contract). When omitted (gallery / operator console), the console
 *        creates + owns its own region.
 * @returns {{
 *   el: HTMLElement,
 *   timelineEl: HTMLElement,
 *   liveEl: HTMLElement,
 *   elapsedEl: HTMLElement,
 *   onLine: (line:string)=>void,
 *   onStatus: (status:string)=>void,
 *   onError: (msg:string)=>void,
 *   markRemainingDone: ()=>void,
 *   freezeElapsed: (ms?:number)=>void,
 *   announce: (text:string)=>void,
 *   getTimeline: ()=>Array<{name,duration_ms,completed_at}>,
 *   getLastRunning: ()=>?string,
 *   getFailedPhase: ()=>?string,
 *   get: (name:string)=>?Object,
 *   order: Array<string>,
 *   destroy: ()=>void,
 * }}
 */
export function runProgressConsole(opts = {}) {
  const phases = Array.isArray(opts.phases) ? opts.phases : [];
  const startMs = typeof opts.startMs === 'number' ? opts.startMs : Date.now();
  const ariaLabel = opts.ariaLabel || 'Course build steps';
  const ringSize = opts.ringSize || 28;
  const elapsedPrefix = opts.elapsedPrefix != null ? opts.elapsedPrefix : 'elapsed ';

  // Friendly-label lookup for the narrative sentence.
  const labelOf = new Map();
  phases.forEach((p) => labelOf.set(p.name, p.label || p.name));

  /* ----- the kit pieces ----- */
  const timeline = phaseTimeline(phases, { ariaLabel, ringSize });
  const timer = makeElapsed({ startMs, prefix: elapsedPrefix, autostart: true });

  // The SINGLE live-truth region: role=status aria-live=polite. EVERYTHING the
  // user must hear (phase transitions, terminal verdict) goes here — debounced,
  // never per-second. The elapsed timer above is aria-hidden decoration.
  // When the host owns one (the studio shell's #status), reuse it so the page
  // keeps exactly one live region; otherwise mint + embed our own.
  const ownsLive = !(opts.liveRegion && opts.liveRegion.nodeType === 1);
  const liveEl = ownsLive
    ? el('p', { class: 'run-progress-live', role: 'status', 'aria-live': 'polite' })
    : opts.liveRegion;

  const meta = el('p', { class: 'run-progress-meta muted' }, [timer.el]);

  const root = el('div', { class: 'run-progress kit' },
    ownsLive ? [meta, liveEl, timeline.el] : [meta, timeline.el]);

  /* ----- Tier-1 per-phase timing (GUI-side, zero backend) ----- */
  // completed_at[name] = ISO-derived ms of the phase's done/skipped line (or, if
  // the line carried no ISO, the wall-clock now). prevDoneMs anchors the next
  // phase's duration; the first phase anchors to startMs.
  let prevDoneMs = startMs;
  const timing = []; // [{name, duration_ms, completed_at}]

  function recordTiming(name, atMs) {
    const completedAt = atMs != null ? atMs : Date.now();
    const duration = Math.max(0, completedAt - prevDoneMs);
    timing.push({ name, duration_ms: duration, completed_at: completedAt });
    prevDoneMs = completedAt;
  }

  /* ----- narrative (debounced phase-transition announcement) ----- */
  let lastAnnouncedPhase = null;
  let lastTaskKey = null;

  function announce(text) {
    if (text && liveEl.textContent !== text) liveEl.textContent = text;
  }

  /** Compose + announce the "what's happening now" sentence for `name`.
   * Debounced: re-announce only on a phase change OR a task-count change, never
   * on the per-second elapsed tick. */
  function announcePhase(name, doneN, totalN) {
    if (!name) return;
    const label = labelOf.get(name) || name;
    const hasTasks = Number(totalN) > 0;
    const taskKey = hasTasks ? `${name}:${doneN}/${totalN}` : null;
    if (name === lastAnnouncedPhase && taskKey === lastTaskKey) return;
    lastAnnouncedPhase = name;
    lastTaskKey = taskKey;
    announce(hasTasks ? `${label} — ${doneN} of ${totalN} done.` : `${label}…`);
  }

  /* ----- state mirrors of the create.js heuristic ----- */
  let lastRunning = null;
  let failedPhase = null;

  function markCompleted(name, marker, atMs) {
    recordTiming(name, atMs);
    const next = timeline.markCompleted(name, marker);
    if (next) { lastRunning = next; announcePhase(next); }
    return next;
  }

  /* ----- public line handler ----- */
  function onLine(line) {
    if (typeof line !== 'string') return;
    const prog = _PROGRESS_RE.exec(line);
    if (prog) {
      const name = prog[2];
      const doneN = Number(prog[3]);
      const totalN = Number(prog[4]);
      timeline.setTaskProgress(name, doneN, totalN);
      // The progress line implies this phase is the running one.
      if (name !== lastRunning && timeline.get(name)) {
        timeline.setState(name, 'running');
        lastRunning = name;
      }
      announcePhase(name, doneN, totalN);
      return;
    }
    const done = _PHASE_DONE_RE.exec(line);
    if (done) { markCompleted(done[2], done[3], _isoMs(done[1])); return; }
    const fail = _PHASE_FAIL_RE.exec(line);
    if (fail) {
      failedPhase = fail[2];
      timeline.setState(fail[2], 'failed');
      // record the failed phase's elapsed slice too (Tier-1 timing).
      recordTiming(fail[2], _isoMs(fail[1]));
      return;
    }
    // Anything else (raw orchestrator chatter) is intentionally ignored: the
    // raw-log affordance is owned by the host page, not this console.
  }

  /** Mark any not-yet-resolved phase done (a build that finished cleanly). */
  function markRemainingDone() {
    timeline.order.forEach((n) => {
      const row = timeline.get(n);
      if (row && row.state() !== 'done' && row.state() !== 'skipped' && row.state() !== 'failed') {
        timeline.setState(n, 'done');
      }
    });
  }

  function onStatus(status) {
    timer.stop();
    if (status === 'completed') {
      markRemainingDone();
      announce('Your course is ready.');
    } else if (status === 'cancelled' || status === 'interrupted') {
      if (lastRunning) timeline.setState(lastRunning, 'pending');
      announce(`Build ${status}.`);
    } else {
      const fp = failedPhase || lastRunning;
      if (fp) timeline.setState(fp, 'failed');
      announce('The course build failed.');
    }
  }

  function onError(msg) {
    timer.stop();
    announce(`Lost connection to the build: ${msg}`);
  }

  /** Seed the first phase as running (the host calls this when no WS line has
   * arrived yet, mirroring create.js). */
  function startFirstRunning() {
    const first = timeline.startRunning();
    if (first) { lastRunning = first; announcePhase(first); }
    return first;
  }

  return {
    el: root,
    timelineEl: timeline.el,
    liveEl,
    elapsedEl: timer.el,
    onLine,
    onStatus,
    onError,
    startFirstRunning,
    markRemainingDone,
    markCompleted,
    setState: timeline.setState,
    setTaskProgress: timeline.setTaskProgress,
    addGate: timeline.addGate,
    freezeElapsed: (ms) => timer.freeze(ms),
    announce,
    getTimeline: () => timing.slice(),
    getLastRunning: () => lastRunning,
    getFailedPhase: () => failedPhase,
    get: timeline.get,
    order: timeline.order,
    destroy: () => { timer.stop(); },
  };
}

export default runProgressConsole;
