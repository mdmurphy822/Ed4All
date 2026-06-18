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
 *   [<iso>] [gate] <name> <gate_id> <pass|fail> <sev>
 *                                          → append a gateChip to the phase's
 *                                            gate strip (Phase-5 live ticker).
 *   [<iso>] [gate] <name> __summary__ <P>/<T>
 *                                          → render/update the "P of T checks
 *                                            passing" summary chip on the phase.
 *
 * The [gate] arm is DEFENSIVE: a sibling worker emits these lines, so the UI
 * must work before/without them. Malformed [gate] lines are ignored, and the
 * per-phase tally is reconstructed GUI-side so a __summary__ that races ahead of
 * (or without) the per-gate lines still renders a coherent "P of T" chip.
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

import { el, clear } from '../dom.js';
import { phaseTimeline } from './phase-timeline.js';
import { elapsed as makeElapsed } from './elapsed.js';
import { progressRing } from './ring.js';

/* Line grammar. The ISO prefix is captured (group 1) so we can drive Tier-1
 * timing; the legacy create.js regex discarded it. All tolerant of extra
 * leading text and trailing `— <label>` / `: <reason>` suffixes. */
const _PHASE_DONE_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[phase\]\s+(\S+)\s+(done|skipped)\b/;
const _PHASE_FAIL_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[phase\]\s+(\S+)\s+failed\b/;
const _PROGRESS_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[progress\]\s+(\S+)\s+(\d+)\s*\/\s*(\d+)/;
/* [gate] arm (Phase 5). The aggregate summary line uses the __summary__ sentinel
 * gate_id + a P/T count; a per-gate line carries a real gate_id + pass|fail +
 * severity. Both tolerate the leading ISO prefix + trailing chatter. */
const _GATE_SUMMARY_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[gate\]\s+(\S+)\s+__summary__\s+(\d+)\s*\/\s*(\d+)/;
const _GATE_RE = /^\s*(?:\[([^\]]+)\]\s+)?\[gate\]\s+(\S+)\s+(\S+)\s+(pass|fail)\b(?:\s+(\S+))?/;

/** Parse the captured ISO prefix to epoch-ms; null if absent / unparseable. */
function _isoMs(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

/* ETA discipline (HARD): the remaining-time estimate is rendered as a coarse
 * RANGE ("about 6m left"), NEVER a per-second zeroing countdown — a precise
 * countdown is a lie when the per-phase medians are noisy. We round UP to a
 * coarse bucket and never present a sub-minute zero ("finishing up" instead).
 *
 * Bucketing: < ~45s → "finishing up"; < 10m → nearest minute; < 60m → nearest
 * 5 minutes; otherwise nearest 10 minutes. The number is always the coarse
 * bucket, so it does not tick down second-by-second. */
function _etaPhrase(remainingMs) {
  if (!Number.isFinite(remainingMs) || remainingMs <= 0) return 'finishing up';
  const totalSec = remainingMs / 1000;
  if (totalSec < 45) return 'finishing up';
  const min = totalSec / 60;
  let bucket;
  if (min < 10) bucket = Math.max(1, Math.round(min));
  else if (min < 60) bucket = Math.round(min / 5) * 5;
  else bucket = Math.round(min / 10) * 10;
  if (bucket >= 60) {
    const h = Math.floor(bucket / 60);
    const m = bucket % 60;
    return m ? `about ${h}h ${m}m left` : `about ${h}h left`;
  }
  return `about ${bucket}m left`;
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
 * @param {Object} [opts.durations]  per-phase median-duration medians the host
 *        fetched from GET /api/runs/phase-durations, shape
 *        `{ <phase>: {median_ms:int|null, n:int} }`. Drives the honest ETA. The
 *        console computes ETA itself from its own getTimeline() + these medians.
 * @param {string} [opts.durationsSource]  "history" | "prior" | "mixed" — the
 *        provenance string from the same endpoint. "prior" (the static fallback
 *        curve) downgrades the ETA label to "rough estimate"; <2 historical
 *        runs (no usable medians) renders "estimating…".
 * @param {(runId?:string)=>void} [opts.onCancel]  host-owned cancel callback;
 *        when supplied a sticky "Cancel build" button renders in the header. The
 *        console NEVER fetches — the host owns POST /api/runs/{id}/cancel and the
 *        two-stage 202→cancelled resolution.
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
  // Median per-phase durations + their provenance (host-fetched). Defensive: the
  // sibling endpoint may not exist yet, so an absent/garbage map degrades to the
  // "estimating…" state rather than a false-precision number.
  let durations = (opts.durations && typeof opts.durations === 'object') ? opts.durations : {};
  let durationsSource = typeof opts.durationsSource === 'string' ? opts.durationsSource : null;
  const onCancel = typeof opts.onCancel === 'function' ? opts.onCancel : null;
  const phaseCount = phases.length;

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

  /* ----- the overall header: aria-hidden progress ring + ETA + cancel ----- */
  // The overall ring is DECORATION (aria-hidden); the a11y truth (phase
  // transitions + ETA changes) is the single role=status line. We rebuild the
  // ring node in place on each meaningful change.
  const overallRingSlot = el('span', { class: 'run-overall-ring', 'aria-hidden': 'true' });
  // The ETA line is a plain muted span (NOT a live region — announcements route
  // through the single role=status line on a MEANINGFUL change only, never the
  // per-second tick). tabular-nums so the bucket digits don't jitter.
  const etaEl = el('span', { class: 'run-eta tabular-nums muted', text: 'estimating…' });
  const overallCount = el('span', { class: 'run-overall-count tabular-nums muted' });

  // Sticky cancel button — host-owned action. Real <button> (focus-visible,
  // >=24px enforced by kit CSS), hidden once terminal. The console NEVER
  // fetches; it invokes the host's onCancel and shows the two-stage text.
  const cancelBtn = onCancel
    ? el('button', { type: 'button', class: 'btn run-cancel-btn', text: 'Cancel build' })
    : null;
  const header = el('div', { class: 'run-progress-header' }, [
    overallRingSlot,
    el('span', { class: 'run-overall-text' }, [overallCount, etaEl]),
    cancelBtn,
  ]);

  const root = el('div', { class: 'run-progress kit' },
    ownsLive ? [meta, header, liveEl, timeline.el] : [meta, header, timeline.el]);

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

  /* ----- overall progress ring + honest ETA ----- */
  // The set of resolved (done/skipped/failed) phases, in completion order, is
  // mirrored from `timing`/`failedPhase`; the in-flight phase's fractional task
  // progress refines the overall ring beyond the integer phases_done/total.
  let curTaskFrac = 0;          // 0..1 fractional progress of the running phase
  let terminal = false;
  let lastEtaPhrase = null;     // debounce ETA announcements to meaningful change

  /** Median ms for a phase (null when unknown / not a positive number). */
  function medianFor(name) {
    const d = durations && durations[name];
    const m = d && Number(d.median_ms);
    return Number.isFinite(m) && m > 0 ? m : null;
  }

  /** True when we have <2 historical runs to lean on: every usable median came
   * from fewer than 2 samples (or there are no medians at all). With so little
   * history we refuse a number and say "estimating…" (the hard discipline). */
  function tooFewSamples() {
    let usable = 0;
    phases.forEach((p) => {
      const d = durations && durations[p.name];
      if (d && Number(d.median_ms) > 0 && Number(d.n) >= 2) usable += 1;
    });
    return usable === 0;
  }

  /** Count phases already resolved (done|skipped|failed) from the timeline. */
  function phasesResolved() {
    let n = 0;
    timeline.order.forEach((name) => {
      const row = timeline.get(name);
      const st = row && row.state();
      if (st === 'done' || st === 'skipped' || st === 'failed') n += 1;
    });
    return n;
  }

  /** Re-render the overall ring (phases_done/total + in-flight task fraction)
   * and the ETA line. Returns the ETA phrase (so the narrative can announce a
   * MEANINGFUL change). Pure presentation; no fetch. */
  function refreshOverall(announceEta) {
    const doneN = phasesResolved();
    const total = phaseCount || 0;
    // Overall fraction: integer phases plus the running phase's task fraction,
    // divided by total. Never exceeds 1.
    const frac = total > 0
      ? Math.min(1, (doneN + (terminal ? 0 : curTaskFrac)) / total)
      : 0;
    const state = terminal ? (failedPhase ? 'failed' : 'done') : 'running';
    clear(overallRingSlot);
    overallRingSlot.appendChild(progressRing({
      percent: Math.round(frac * 100),
      state,
      size: ringSize,
      label: `${doneN} of ${total} steps done`,
    }));
    overallCount.textContent = total ? `Step ${Math.min(doneN + (terminal ? 0 : 1), total)} of ${total} · ` : '';

    // ---- the ETA line (honest range, never a zeroing countdown) ----
    let phrase;
    let cls = 'run-eta tabular-nums muted';
    if (terminal) {
      phrase = '';
    } else if (tooFewSamples()) {
      phrase = 'estimating…';
    } else {
      // Sum medians for phases NOT yet resolved; subtract elapsed-in-current.
      let remaining = 0;
      let haveAny = false;
      timeline.order.forEach((name) => {
        const row = timeline.get(name);
        const st = row && row.state();
        if (st === 'done' || st === 'skipped' || st === 'failed') return;
        const m = medianFor(name);
        if (m != null) { remaining += m; haveAny = true; }
      });
      if (!haveAny) {
        phrase = 'estimating…';
      } else {
        const elapsedInCur = Math.max(0, Date.now() - prevDoneMs);
        const rough = durationsSource === 'prior';
        phrase = _etaPhrase(remaining - elapsedInCur) + (rough ? ' (rough estimate)' : '');
        if (rough) cls += ' is-rough';
      }
    }
    etaEl.className = cls;
    etaEl.textContent = phrase;
    // Announce ETA changes ONLY on a meaningful change (bucket shift), never the
    // per-second elapsed tick — re-uses the single role=status region.
    if (announceEta && phrase && phrase !== lastEtaPhrase && !terminal) {
      lastEtaPhrase = phrase;
    }
    return phrase;
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

  /* ----- Phase-5 live gate ticker ----- */
  // Per-phase tally reconstructed GUI-side from the [gate] lines: how many gates
  // resolved, how many passed, and the WORST result seen (so the summary chip's
  // tone is calm for warnings, red only for a real critical fail). This makes a
  // __summary__ line render coherently even if the per-gate lines race / are
  // absent — the chip is driven by max(backend P/T, our reconstructed tally).
  const gateTally = new Map(); // phase -> {passed, total, worst}
  // Debounce the polite per-phase summary announcement to a MEANINGFUL change
  // (the chip text changed) — never per-gate chatter. Routes through the single
  // role=status line, and only for the phase currently running.
  const lastGateAnnounce = new Map(); // phase -> last announced summary text

  /** Map a [gate] pass|fail + severity onto a gateChip result token. A failing
   * gate at warning severity is a CALM warn (△), not a critical fail (✗). */
  function _gateResult(passFail, severity) {
    if (passFail === 'pass') return 'pass';
    const sev = String(severity || '').toLowerCase();
    return (sev === 'warning' || sev === 'warn') ? 'warn' : 'fail';
  }

  /** Rank a result for the per-phase "worst" tone (pass < warn < fail). */
  function _worse(a, b) {
    const rank = { pass: 0, warn: 1, fail: 2 };
    return (rank[b] || 0) > (rank[a] || 0) ? b : a;
  }

  function markCompleted(name, marker, atMs) {
    recordTiming(name, atMs);
    const next = timeline.markCompleted(name, marker);
    curTaskFrac = 0; // a new phase starts at 0 fractional task progress
    if (next) { lastRunning = next; announcePhase(next); }
    refreshOverall(true);
    return next;
  }

  /** A single resolved gate: append a gateChip to the phase's strip + update the
   * GUI-side tally (so a later __summary__ — or its absence — is coherent). */
  function handleGate(name, gateId, passFail, severity) {
    if (!name || !timeline.get(name)) return; // unknown phase → ignore
    const result = _gateResult(passFail, severity);
    timeline.addGate(name, { result, gate_id: gateId, severity });
    const t = gateTally.get(name) || { passed: 0, total: 0, worst: 'pass' };
    t.total += 1;
    if (result === 'pass') t.passed += 1;
    t.worst = _worse(t.worst, result);
    gateTally.set(name, t);
    // Refresh the chip from the running tally (no backend summary needed yet).
    renderGateSummary(name, t.passed, t.total, t.worst);
  }

  /** The aggregate "P of T" summary line. Reconcile with the GUI-side tally
   * (take the larger total / worst tone) so neither source under-reports. */
  function handleGateSummary(name, passed, total) {
    if (!name || !timeline.get(name)) return;
    if (!Number.isFinite(passed) || !Number.isFinite(total)) return; // malformed
    const t = gateTally.get(name) || { passed: 0, total: 0, worst: 'pass' };
    const P = Math.max(passed, t.passed);
    const T = Math.max(total, t.total);
    // Worst tone: if the summary implies failures we didn't see per-gate, treat
    // them conservatively as warnings (calm) unless a critical was already seen.
    let worst = t.worst;
    if (T - P > 0 && worst === 'pass') worst = 'warn';
    gateTally.set(name, { passed: P, total: T, worst });
    renderGateSummary(name, P, T, worst);
  }

  /** Render the chip + (debounced) announce a MEANINGFUL change of the RUNNING
   * phase's summary through the single role=status line — never per-gate. */
  function renderGateSummary(name, passed, total, worst) {
    timeline.setGateSummary(name, passed, total, worst);
    if (name !== lastRunning || terminal) return;
    const failed = Math.max(0, total - passed);
    let tail = '';
    if (worst === 'warn' && failed > 0) tail = `, ${failed} ${failed === 1 ? 'warning' : 'warnings'}`;
    else if (worst === 'fail' && failed > 0) tail = `, ${failed} failed`;
    const text = `${passed} of ${total} checks passing${tail}.`;
    if (lastGateAnnounce.get(name) === text) return; // debounce: no chatter
    lastGateAnnounce.set(name, text);
    const label = labelOf.get(name) || name;
    announce(`${label} — ${text}`);
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
      curTaskFrac = totalN > 0 ? Math.min(1, doneN / totalN) : 0;
      announcePhase(name, doneN, totalN);
      refreshOverall(true);
      return;
    }
    // ----- Phase-5 live gate ticker (defensive; ignore malformed) -----
    const gsum = _GATE_SUMMARY_RE.exec(line);
    if (gsum) { handleGateSummary(gsum[2], Number(gsum[3]), Number(gsum[4])); return; }
    const gate = _GATE_RE.exec(line);
    if (gate) { handleGate(gate[2], gate[3], gate[4], gate[5]); return; }

    const done = _PHASE_DONE_RE.exec(line);
    if (done) { markCompleted(done[2], done[3], _isoMs(done[1])); return; }
    const fail = _PHASE_FAIL_RE.exec(line);
    if (fail) {
      failedPhase = fail[2];
      timeline.setState(fail[2], 'failed');
      // record the failed phase's elapsed slice too (Tier-1 timing).
      recordTiming(fail[2], _isoMs(fail[1]));
      refreshOverall(false);
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
    terminal = true;
    hideCancel();
    if (status === 'completed') {
      markRemainingDone();
      announce('Your course is ready.');
    } else if (status === 'cancelled' || status === 'interrupted') {
      if (lastRunning) timeline.setState(lastRunning, 'pending');
      announce(`Build ${status}.`);
    } else {
      const fp = failedPhase || lastRunning;
      if (fp) { failedPhase = fp; timeline.setState(fp, 'failed'); }
      announce('The course build failed.');
    }
    refreshOverall(false);
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
    refreshOverall(false);
    return first;
  }

  /* ----- sticky cancel button (host owns the fetch + two-stage 202) ----- */
  let cancelRequested = false;
  function hideCancel() { if (cancelBtn) cancelBtn.hidden = true; }
  /** Reflect the 202/`cancel_requested` stage: the button claims NO instant stop
   * ("finishing current step"); the row resolves to cancelled only when the WS
   * terminal status frame later calls onStatus('cancelled'). */
  function setCancelling() {
    cancelRequested = true;
    if (cancelBtn) {
      cancelBtn.disabled = true;
      cancelBtn.textContent = 'Cancelling… (finishing current step)';
    }
    announce('Cancelling the build — finishing the current step.');
  }
  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      if (terminal || cancelRequested) return;
      // The console NEVER fetches. The host owns POST /api/runs/{id}/cancel and
      // decides (on a 202 / cancel_requested) to call back setCancelling().
      try { onCancel(); } catch (_) { /* host surfaces its own error */ }
    });
  }

  /* Slow ETA recompute: the elapsed-in-current-phase shrinks the remaining
   * estimate, but we MUST NOT chatter per-second. A coarse 15s tick re-buckets
   * and only touches the role=status line on a real bucket change. */
  let etaTimer = null;
  function startEtaTick() {
    if (etaTimer != null) return;
    etaTimer = setInterval(() => {
      if (terminal) return;
      const phrase = refreshOverall(false);
      if (phrase && phrase !== lastEtaPhrase && !/estimating|finishing/.test(phrase)) {
        lastEtaPhrase = phrase;
        // Re-announce ONLY the ETA bucket change (debounced, meaningful-change).
        if (lastAnnouncedPhase) {
          const label = labelOf.get(lastAnnouncedPhase) || lastAnnouncedPhase;
          announce(`${label}… ${phrase}.`);
        }
      }
    }, 15000);
  }
  startEtaTick();

  /** Host hook: re-seed the medians once the phase-durations fetch resolves
   * (the host may mount the console before the durations arrive). */
  function setDurations(map, source) {
    if (map && typeof map === 'object') durations = map;
    if (typeof source === 'string') durationsSource = source;
    refreshOverall(false);
  }

  // Initial paint of the overall ring + ETA.
  refreshOverall(false);

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
    setGateSummary: timeline.setGateSummary,
    freezeElapsed: (ms) => timer.freeze(ms),
    announce,
    setDurations,
    setCancelling,
    hideCancel,
    cancelEl: cancelBtn,
    etaEl,
    overallRingEl: overallRingSlot,
    getEtaText: () => etaEl.textContent,
    getTimeline: () => timing.slice(),
    getLastRunning: () => lastRunning,
    getFailedPhase: () => failedPhase,
    get: timeline.get,
    order: timeline.order,
    destroy: () => { timer.stop(); if (etaTimer != null) { clearInterval(etaTimer); etaTimer = null; } },
  };
}

export default runProgressConsole;
