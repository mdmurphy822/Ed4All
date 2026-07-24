/* Ed4All Studio — Assistant chat panel (#/assistant).
 *
 * A pure chat front-end over POST /api/assistant/chat. The engine server-side
 * owns the whole sandbox (tool whitelist, loopback guard, round cap); this
 * page only renders messages, round-trips the returned `history`, and shows a
 * collapsible per-message tool-call trace. A status pill polls
 * GET /api/assistant/status (seat serving / down + model id).
 *
 * The conversation lives in module state for the browser session (the server
 * is stateless); "New conversation" clears it.
 *
 * "Debug last failure": one GET /api/assistant/debug-context probe on panel
 * mount enables the header button (disabled + reason tooltip otherwise).
 * Clicking starts a fresh debug conversation: the failure summary renders as
 * a system-style banner bubble, then the model's opening diagnosis; every
 * turn of that conversation re-sends {debug: true, run_id} (the server caches
 * the built context per failed run). Deep link: #/assistant/debug[/WF-id]
 * triggers the same flow on mount.
 *
 * "Seat model": a real <button> beside the seat pill, shown ONLY when
 * GET /api/assistant/seat/status reports no candidate seat live and no start
 * in flight. It POSTs /api/assistant/seat (the server starts the assistant's
 * OWN registered seat through the lifecycle coherent-start path — the GUI
 * never names a script), then shows a busy state ("Seating nano…", aria-busy)
 * and polls the same status endpoint every 10s until a seat is live (badge
 * refreshes, button hides) or the start fails (toast + inline message).
 */

import { api, apiJSON } from '/shared/api.js';
import { el, clear, uid } from '/shared/dom.js';
import { toastErr } from '/shared/toast.js';

/** Poll interval (ms) while a seat start is in flight (cold start = minutes). */
const SEAT_POLL_MS = 10000;

/** Live poll timer for the seat control (cleared on every panel mount). */
let seatPollTimer = null;

// Session-scoped conversation state (survives route changes, not reloads).
let history = null;
let transcript = []; // [{role: 'user'|'assistant'|'error'|'banner', text, toolCalls?}]
let debugRunId = null; // non-null while the active conversation is debug-mode

const DEBUG_OPENING = 'Diagnose this failure: what is the most likely root cause, '
  + 'and what should I do next?';

export async function renderAssistant(shell) {
  shell.crumbs([{ text: 'Assistant' }]);
  const v = clear(shell.view());
  shell.setBusy(false);
  shell.setStatus('Assistant ready.');

  v.appendChild(el('h1', { text: 'Assistant' }));
  v.appendChild(el('p', {
    class: 'muted assistant-intro',
    text: 'Ask about operating Ed4All — run and seat status, the book campaign, '
      + 'starting the next book build, or stopping a run. The assistant only '
      + 'uses its built-in status tools; it has no shell or file access.',
  }));

  // ---- status pill (seat serving / down) ---------------------------------
  const pill = el('span', { class: 'assistant-pill is-unknown', text: 'Checking seat…' });
  const statusLine = el('p', { class: 'assistant-status', role: 'status', 'aria-live': 'polite' }, [pill]);
  v.appendChild(statusLine);
  refreshStatus(pill);

  // ---- "Seat model" (only when no seat is live) ---------------------------
  const seatCtl = mountSeatControl(statusLine, pill);

  // ---- "Debug last failure" (header toolbar) ------------------------------
  // Always visible; enabled only when the one mount-time probe finds a failed
  // run. No polling — a single GET /api/assistant/debug-context per mount.
  const debugBtn = el('button', {
    type: 'button',
    class: 'btn assistant-debug-btn',
    text: 'Debug last failure',
    title: 'Checking for a failed run…',
  });
  debugBtn.disabled = true;
  debugBtn.setAttribute('aria-disabled', 'true');
  v.appendChild(el('div', { class: 'assistant-toolbar' }, [debugBtn]));

  // Deep link: #/assistant/debug[/<WF-id>] triggers the flow after the probe.
  const deepLink = (location.hash || '').match(/^#\/?assistant\/debug(?:\/([^/?#]+))?/);
  const probeRunId = deepLink && deepLink[1] ? decodeURIComponent(deepLink[1]) : null;

  let debugInfo = null;
  debugBtn.addEventListener('click', () => {
    if (debugInfo && debugInfo.available) startDebug(debugInfo);
  });
  probeDebug(probeRunId).then((info) => {
    debugInfo = info;
    if (info && info.available) {
      debugBtn.disabled = false;
      debugBtn.removeAttribute('aria-disabled');
      debugBtn.title = info.summary || 'Start a debug conversation about the last failed run.';
      if (deepLink && !transcript.length) startDebug(info);
    } else {
      debugBtn.disabled = true;
      debugBtn.setAttribute('aria-disabled', 'true');
      debugBtn.title = (info && info.reason) || 'No failed run available to debug.';
    }
  });

  // ---- transcript ---------------------------------------------------------
  const log = el('div', {
    class: 'chat-log',
    role: 'log',
    'aria-label': 'Assistant conversation',
    'aria-live': 'polite',
  });
  v.appendChild(log);
  transcript.forEach((m) => log.appendChild(renderMessage(m)));
  if (!transcript.length) {
    log.appendChild(el('p', { class: 'muted chat-empty', text: 'No messages yet. Ask a question below.' }));
  }

  // ---- input form ---------------------------------------------------------
  const inputId = uid('assistant-input');
  const input = el('textarea', {
    id: inputId,
    class: 'chat-input',
    rows: '2',
    placeholder: 'e.g. What is the campaign status?',
    'aria-label': 'Message the assistant',
  });
  const sendBtn = el('button', { type: 'submit', class: 'btn primary', text: 'Send' });
  const clearBtn = el('button', { type: 'button', class: 'btn', text: 'New conversation' });
  const form = el('form', { class: 'chat-form', novalidate: true }, [
    el('label', { class: 'visually-hidden', for: inputId, text: 'Message the assistant' }),
    input,
    el('div', { class: 'chat-actions' }, [clearBtn, sendBtn]),
  ]);
  v.appendChild(form);

  let busy = false;

  async function send() {
    const message = input.value.trim();
    if (!message || busy) return;
    busy = true;
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;
    log.setAttribute('aria-busy', 'true');

    const emptyHint = log.querySelector('.chat-empty');
    if (emptyHint) emptyHint.remove();
    pushMessage(log, { role: 'user', text: message });
    const thinking = el('p', { class: 'loading chat-thinking', text: 'Assistant is thinking…' });
    log.appendChild(thinking);
    log.scrollTop = log.scrollHeight;
    shell.setStatus('Waiting for the assistant.');

    const payload = { message, history };
    if (debugRunId) { payload.debug = true; payload.run_id = debugRunId; }

    let body;
    try {
      body = await apiJSON('/api/assistant/chat', 'POST', payload);
    } catch (e) {
      thinking.remove();
      // A 503 detail carries the seat-start instructions — show it in-line.
      pushMessage(log, { role: 'error', text: shell.errText(e) });
      if (e && e.status !== 503) toastErr(e, 'Assistant request failed');
      shell.setStatus('Assistant request failed.');
      refreshStatus(pill);
      seatCtl.refresh();
      finishTurn();
      return;
    }

    thinking.remove();
    history = body.history || null;
    pushMessage(log, {
      role: 'assistant',
      text: body.reply || '',
      toolCalls: Array.isArray(body.tool_calls) ? body.tool_calls : [],
      model: body.model || null,
      seat: body.seat || null,
    });
    shell.setStatus('Assistant replied.');
    refreshStatus(pill);
    seatCtl.refresh();
    finishTurn();
  }

  function finishTurn() {
    busy = false;
    input.disabled = false;
    sendBtn.disabled = false;
    log.setAttribute('aria-busy', 'false');
    log.scrollTop = log.scrollHeight;
    input.focus();
  }

  /** Start a fresh debug-mode conversation from a probe result. */
  async function startDebug(info) {
    if (busy) return;
    busy = true;
    input.disabled = true;
    sendBtn.disabled = true;
    log.setAttribute('aria-busy', 'true');

    // Fresh conversation seeded in debug mode: banner bubble first.
    history = null;
    transcript = [];
    debugRunId = info.run_id || null;
    clear(log);
    pushMessage(log, {
      role: 'banner',
      text: info.summary || `Debugging failed run ${info.run_id || ''}.`,
    });
    const thinking = el('p', { class: 'loading chat-thinking', text: 'Assistant is diagnosing…' });
    log.appendChild(thinking);
    log.scrollTop = log.scrollHeight;
    shell.setStatus('Assistant is diagnosing the failed run.');

    const payload = { message: DEBUG_OPENING, history: null, debug: true };
    if (debugRunId) payload.run_id = debugRunId;

    let body;
    try {
      body = await apiJSON('/api/assistant/chat', 'POST', payload);
    } catch (e) {
      thinking.remove();
      pushMessage(log, { role: 'error', text: shell.errText(e) });
      if (e && e.status !== 503) toastErr(e, 'Debug request failed');
      shell.setStatus('Debug request failed.');
      refreshStatus(pill);
      seatCtl.refresh();
      finishTurn();
      return;
    }

    thinking.remove();
    history = body.history || null;
    pushMessage(log, {
      role: 'assistant',
      text: body.reply || '',
      toolCalls: Array.isArray(body.tool_calls) ? body.tool_calls : [],
      model: body.model || null,
      seat: body.seat || null,
    });
    shell.setStatus('Assistant replied.');
    refreshStatus(pill);
    seatCtl.refresh();
    finishTurn();
  }

  form.addEventListener('submit', (e) => { e.preventDefault(); send(); });
  // Enter sends; Shift+Enter inserts a newline (standard chat affordance).
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  clearBtn.addEventListener('click', () => {
    history = null;
    transcript = [];
    debugRunId = null;
    clear(log).appendChild(el('p', { class: 'muted chat-empty', text: 'Conversation cleared. Ask a question below.' }));
    shell.setStatus('Conversation cleared.');
    input.focus();
  });

  input.focus();
}

/* --------------------------------------------------------------- messages */

function pushMessage(log, msg) {
  transcript.push(msg);
  log.appendChild(renderMessage(msg));
}

function renderMessage(msg) {
  if (msg.role === 'banner') {
    // System-style debug-context banner (existing advisory tokens).
    return el('div', { class: 'chat-msg chat-banner' }, [
      el('p', { class: 'chat-role', text: 'Debug context' }),
      el('p', { class: 'chat-text', text: msg.text }),
    ]);
  }
  if (msg.role === 'error') {
    return el('div', { class: 'chat-msg chat-error' }, [
      el('p', { class: 'chat-role', text: 'Assistant unavailable' }),
      el('p', { class: 'error chat-text', text: msg.text }),
    ]);
  }
  const isUser = msg.role === 'user';
  const node = el('div', { class: `chat-msg ${isUser ? 'chat-user' : 'chat-assistant'}` }, [
    el('p', { class: 'chat-role', text: isUser ? 'You' : 'Assistant' }),
    el('p', { class: 'chat-text', text: msg.text }),
  ]);
  // Which seat/model answered (S11). Absent/older metadata → no badge.
  if (!isUser && msg.model) {
    node.appendChild(el('span', {
      class: 'assistant-seat-badge',
      text: `${msg.model}${msg.seat ? ` @ ${msg.seat}` : ''}`,
    }));
  }
  if (!isUser && msg.toolCalls && msg.toolCalls.length) {
    node.appendChild(renderToolTrace(msg.toolCalls));
  }
  return node;
}

/** Collapsible per-message tool-call trace (name + arguments + result). */
function renderToolTrace(toolCalls) {
  const n = toolCalls.length;
  const details = el('details', { class: 'tool-trace' }, [
    el('summary', { text: `${n} tool call${n === 1 ? '' : 's'}` }),
  ]);
  toolCalls.forEach((t) => {
    details.appendChild(el('div', { class: 'tool-call' }, [
      el('p', { class: 'tool-name' }, [el('code', { text: t.tool || '(unnamed tool)' })]),
      el('p', { class: 'tool-label', text: 'Arguments' }),
      el('pre', { class: 'tool-pre', text: prettyJson(t.arguments) }),
      el('p', { class: 'tool-label', text: 'Result' }),
      el('pre', { class: 'tool-pre', text: String(t.result ?? '') }),
    ]));
  });
  return details;
}

function prettyJson(raw) {
  if (raw == null) return '';
  try { return JSON.stringify(JSON.parse(raw), null, 2); } catch { return String(raw); }
}

/* ------------------------------------------------------------------ debug */

/** One mount-time probe of GET /api/assistant/debug-context (never throws). */
async function probeDebug(runId) {
  try {
    const q = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
    return await api(`/api/assistant/debug-context${q}`);
  } catch (e) {
    return { available: false, reason: 'Debug probe failed.' };
  }
}

/* ------------------------------------------------------------- seat model */

/**
 * Mount the "Seat model" control after the seat pill.
 *
 * Contract: the button exists in the DOM only while it is actionable —
 * `hidden` whenever any candidate seat is live — and is `disabled` +
 * `aria-busy` while a start is in flight. It never names a script: the POST
 * has no body and the server resolves the seat + its registered launch spec.
 *
 * @param {HTMLElement} host  the status line the control is appended to.
 * @param {HTMLElement} pill  the seat pill to refresh when a seat comes up.
 * @returns {{refresh: function}} the control's re-probe entry point.
 */
function mountSeatControl(host, pill) {
  if (seatPollTimer) { clearTimeout(seatPollTimer); seatPollTimer = null; }

  const btn = el('button', {
    type: 'button',
    class: 'btn assistant-seat-btn',
    text: 'Seat model',
    title: 'Start the assistant model seat (takes a few minutes).',
  });
  btn.hidden = true;
  const note = el('p', {
    class: 'muted assistant-seat-note',
    role: 'status',
    'aria-live': 'polite',
  });
  note.hidden = true;
  host.appendChild(btn);
  host.parentNode.insertBefore(note, host.nextSibling);

  function setNote(text, isError) {
    if (!text) { note.hidden = true; note.textContent = ''; note.classList.remove('error'); return; }
    note.hidden = false;
    note.textContent = text;
    note.classList.toggle('error', !!isError);
  }

  function setBusy(on) {
    btn.hidden = false;
    btn.disabled = !!on;
    if (on) {
      btn.setAttribute('aria-busy', 'true');
      btn.setAttribute('aria-disabled', 'true');
      btn.textContent = 'Seating nano…';
    } else {
      btn.removeAttribute('aria-busy');
      btn.removeAttribute('aria-disabled');
      btn.textContent = 'Seat model';
    }
  }

  function schedulePoll() {
    if (seatPollTimer) clearTimeout(seatPollTimer);
    seatPollTimer = setTimeout(() => { seatPollTimer = null; refresh(); }, SEAT_POLL_MS);
  }

  async function refresh() {
    let st;
    try {
      st = await api('/api/assistant/seat/status');
    } catch {
      // Probe failed — never offer an action we can't reason about.
      btn.hidden = true;
      return;
    }
    const seats = (st && st.seats) || {};
    const anyLive = !!(st && st.live_seat) || Object.keys(seats).some((k) => seats[k]);
    if (anyLive) {
      if (seatPollTimer) { clearTimeout(seatPollTimer); seatPollTimer = null; }
      btn.hidden = true;
      setNote('', false);
      refreshStatus(pill);
      return;
    }
    if (st && st.starting) {
      setBusy(true);
      setNote('Starting the assistant model seat. This takes a few minutes.', false);
      schedulePoll();
      return;
    }
    setBusy(false);
    if (st && st.last_seat_error) setNote(st.last_seat_error, true);
    else if (st && st.detail) setNote(st.detail, true);
    else setNote('No assistant model seat is running.', false);
  }

  btn.addEventListener('click', async () => {
    if (btn.disabled) return;
    setBusy(true);
    setNote('Starting the assistant model seat. This takes a few minutes.', false);
    try {
      await apiJSON('/api/assistant/seat', 'POST', {});
    } catch (e) {
      setBusy(false);
      toastErr(e, 'Could not seat the assistant model');
      setNote(errDetail(e), true);
      return;
    }
    schedulePoll();
  });

  refresh();
  return { refresh };
}

/** Human-readable text out of an api.js ApiError ({status, error, detail}). */
function errDetail(e) {
  if (!e) return 'Seat start failed.';
  return String(e.detail || e.error || e.message || e);
}

/* ----------------------------------------------------------------- status */

async function refreshStatus(pill) {
  let st;
  try {
    st = await api('/api/assistant/status');
  } catch {
    pill.className = 'assistant-pill is-down';
    pill.textContent = 'Assistant status unavailable';
    return;
  }
  if (st && st.seat_serving) {
    pill.className = 'assistant-pill is-ok';
    if (st.model) {
      pill.textContent = `Assistant ready · ${st.model}` + (st.seat ? ` @ ${st.seat}` : '');
    } else {
      pill.textContent = 'Assistant ready';
    }
  } else {
    pill.className = 'assistant-pill is-down';
    pill.textContent = st && st.detail ? 'Assistant misconfigured' : 'Assistant seat down';
    if (st && st.detail) pill.title = st.detail;
  }
}
