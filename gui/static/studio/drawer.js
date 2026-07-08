/* Ed4All Studio — in-context "Ask drawer" (C2).
 *
 * A complementary landmark docked beside the content pane: a learner asks a
 * question WHILE READING and keeps reading; the answer arrives in the drawer
 * without navigation. Non-blocking — the question is dispatched to a durable
 * server-side ask job (POST /api/learn/ask-jobs) and polled (GET
 * /api/learn/ask-jobs/<id>), so a 35-50s local-LLM answer survives a tab
 * refresh and never blocks the content pane.
 *
 * Per-course Q->A history lives at SPA level in sessionStorage keyed by slug,
 * so it persists across page navigation within the course AND across a refresh:
 * on mount, any still-pending job is re-polled and any finished answer is
 * re-rendered from the server (the job file is the source of truth).
 *
 * Citations render server-side as <a href="/api/learn/source/..."> links (the
 * answer_render path). The drawer intercepts those clicks and routes them to
 * the host viewer's content pane (loadCitation) instead of navigating away, so
 * answer + source are visible simultaneously with a temporary passage highlight.
 *
 * a11y: the drawer is <aside role="complementary">; the busy state is announced
 * via an aria-live=polite region; the history is a semantic <ol>; focus is
 * managed on expand/collapse; every control is keyboard-operable.
 */

import { api, apiJSON, ApiError } from '/shared/api.js';
import { $, el, clear, uid } from '/shared/dom.js';

const POLL_MS = 1500;
const HISTORY_CAP = 20; // most-recent N Q->A pairs kept per course

function storageKey(slug) { return `ed4all.studio.ask.${slug}`; }

function loadHistory(slug) {
  try {
    const raw = sessionStorage.getItem(storageKey(slug));
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch (_) { return []; }
}

function saveHistory(slug, history) {
  try {
    // Drop transient runtime fields (interval ids, live DOM-node refs) keyed by
    // an underscore prefix — they must never be serialized into sessionStorage
    // (a DOM node round-trips as `{}` and a stale interval id is meaningless on
    // resume). Only the durable Q/A state persists.
    const trimmed = history.slice(-HISTORY_CAP).map((entry) => {
      const copy = {};
      for (const k of Object.keys(entry)) {
        if (!k.startsWith('_')) copy[k] = entry[k];
      }
      return copy;
    });
    sessionStorage.setItem(storageKey(slug), JSON.stringify(trimmed));
  } catch (_) { /* sessionStorage full / disabled — degrade silently */ }
}

function fmtElapsed(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${s}s`;
}

/**
 * Create an Ask drawer bound to one course.
 *
 * @param {Object} opts
 * @param {string} opts.slug              current course slug
 * @param {(item, fragment) => void} opts.loadCitation
 *        host callback: load the cited page into the content pane + highlight.
 * @param {(href) => void} [opts.loadSourceDoc]
 *        host callback: load an anchored original-source document URL into
 *        the content pane (source-side citations).
 * @returns {{ root: HTMLElement, focus: () => void, destroy: () => void }}
 */
export function createAskDrawer({ slug, loadCitation, loadSourceDoc }) {
  let history = loadHistory(slug);
  const timers = new Set();   // poll/elapsed interval ids to clear on destroy
  let busy = false;
  let collapsed = false;

  // --- shell -------------------------------------------------------------
  const headingId = uid('ask-h');
  const liveId = uid('ask-live');
  const toggleBtn = el('button', {
    type: 'button',
    class: 'ask-toggle',
    'aria-expanded': 'true',
    text: 'Hide Ask panel',
  });
  const heading = el('h2', { id: headingId, class: 'ask-title', text: 'Ask about this course' });

  const input = el('textarea', {
    class: 'ask-input',
    rows: '2',
    'aria-label': 'Your question about this course',
    placeholder: 'Ask a question about this course…',
  });
  const submitBtn = el('button', { type: 'button', class: 'ask-submit', text: 'Ask' });
  // L3 "search all courses" toggle — hidden until the capability probe confirms
  // MORE THAN ONE course is indexed (a single-course library has nothing to
  // union). A visible checkbox's state always WINS over the env default.
  const libWideId = uid('ask-libwide');
  const libWideInput = el('input', {
    type: 'checkbox',
    id: libWideId,
    class: 'ask-libwide-input',
  });
  const libWideLabel = el('label', {
    class: 'ask-libwide-label',
    for: libWideId,
    text: 'Search all courses',
  });
  const libWideWrap = el('div', { class: 'ask-libwide' }, [libWideInput, libWideLabel]);
  libWideWrap.hidden = true; // shown only when >1 course is indexed
  // Always-available (user directive): lives in the static shell, not the
  // lazily-rendered history block; clicking with nothing to clear announces
  // that via the live region instead of hiding the control.
  const clearBtn = el('button', {
    type: 'button',
    class: 'ask-clear',
    text: 'Clear history',
    'aria-label': 'Clear question and answer history',
  });
  const form = el('form', { class: 'ask-form' }, [input, libWideWrap, submitBtn]);
  const historyBar = el('div', { class: 'ask-history-bar' }, [clearBtn]);
  clearBtn.addEventListener('click', () => clearHistory());

  // Polite live region for busy/queue/answer-ready announcements (a11y).
  const live = el('p', { id: liveId, class: 'visually-hidden', role: 'status', 'aria-live': 'polite' });

  // History container: the semantic <ol> is created lazily inside it (an empty
  // <ol> is a WCAG 1.3.1 finding, so we render NO list element until there's a
  // Q/A pair to populate it).
  const historyWrap = el('div', { class: 'ask-history-wrap' });

  const body = el('div', { class: 'ask-body' }, [form, live, historyBar, historyWrap]);
  const header = el('div', { class: 'ask-header' }, [heading, toggleBtn]);
  const root = el('aside', {
    class: 'ask-drawer',
    role: 'complementary',
    'aria-labelledby': headingId,
  }, [header, body]);

  // --- gating: no vector index --------------------------------------------
  async function applyGate() {
    let ready = null;
    try {
      ready = await api(`/api/learn/ask-ready/${encodeURIComponent(slug)}`);
    } catch (_) {
      ready = null; // unknown → leave ask box up; the ask call surfaces errors
    }
    if (ready && ready.exists && ready.has_vector_index === false) {
      // Course exists but has NO vector index — explain + how to fix, and keep
      // asking enabled (the engine honestly downgrades to lexical search).
      const note = el('div', { class: 'ask-noindex', role: 'note' }, [
        el('p', { text: 'This course has no semantic search index yet, so answers use keyword search only and may be less precise.' }),
        el('p', { class: 'muted', text: 'To enable semantic answers, build a vector index for this course (ed4all run rag_training …) and reload.' }),
      ]);
      body.insertBefore(note, form);
    }
    // L3: reveal the "search all courses" toggle only when the library has more
    // than one indexed course to union across. Seed the checkbox from the
    // resolved env default; whatever the learner leaves it at then WINS.
    try {
      const caps = await api('/api/learn/ask-capabilities');
      if (caps && caps.library_wide_eligible) {
        libWideWrap.hidden = false;
        libWideInput.checked = !!caps.library_wide_default;
      }
    } catch (_) { /* probe failed → leave the toggle hidden (single-course) */ }
  }

  // --- history rendering --------------------------------------------------
  function clearHistory() {
    // Keep in-flight entries: pollJob/finishError close over the entry
    // OBJECTS, so dropping only settled entries is race-free — a pending
    // question keeps polling and re-renders into the (shorter) list.
    const kept = history.filter(
      (e) => e.status !== 'done' && e.status !== 'error'
    );
    const removed = history.length - kept.length;
    history = kept;
    saveHistory(slug, history);
    renderHistory();
    announce(
      removed > 0
        ? 'Question and answer history cleared.'
        : 'No finished questions to clear.'
    );
  }

  function renderHistory() {
    clear(historyWrap);
    if (history.length === 0) return; // no empty <ol> (WCAG 1.3.1)
    const historyList = el('ol', {
      class: 'ask-history',
      'aria-label': 'Question and answer history',
    });
    historyWrap.appendChild(historyList);
    // Newest first: storage stays append-order (polling holds entry object
    // references), only the render order is inverted so a fresh answer
    // appears at the top of the drawer instead of below the scroll.
    [...history].reverse().forEach((entry) => {
      const li = el('li', { class: 'ask-entry' });
      li.appendChild(el('p', { class: 'ask-q' }, [
        el('span', { class: 'ask-q-label', text: 'Q: ' }),
        document.createTextNode(entry.query),
      ]));
      const ansWrap = el('div', { class: 'ask-a' });
      if (entry.status === 'done' && entry.html) {
        ansWrap.innerHTML = entry.html;
        wireCitations(ansWrap);
        // I6 answer feedback: thumbs up/down + optional comment on a completed
        // answer.
        ansWrap.appendChild(feedbackBar(entry));
      } else if (entry.status === 'error') {
        ansWrap.innerHTML = entry.html || '';
      } else {
        // pending/running — show busy chip + elapsed timer, and (L4) the
        // passages-first preview once retrieval passages have arrived.
        ansWrap.appendChild(busyChip(entry));
        if (Array.isArray(entry._passages) && entry._passages.length) {
          ansWrap.appendChild(passagesPane(entry._passages, entry._passagesRefused));
        }
      }
      li.appendChild(ansWrap);
      historyList.appendChild(li);
    });
  }

  function busyLabel(entry) {
    const queued =
      typeof entry.queue_position === 'number' && entry.queue_position > 0;
    return queued ? `Queued (position ${entry.queue_position + 1})` : 'Thinking';
  }

  function busyChip(entry) {
    const chip = el('span', { class: 'ask-busy', 'aria-hidden': 'false' });
    const labelEl = el('span', { class: 'ask-busy-label', text: busyLabel(entry) });
    // Seed the elapsed clock from the ORIGINAL submit time so a re-render (or a
    // resume after refresh) shows the true elapsed seconds, never a reset to 0s.
    const startedAt = entry.startedAt || Date.now();
    entry.startedAt = startedAt;
    const timeEl = el('span', {
      class: 'ask-elapsed',
      text: fmtElapsed(Date.now() - startedAt),
    });
    chip.appendChild(el('span', { class: 'ask-spin', 'aria-hidden': 'true', text: '⏳ ' }));
    chip.appendChild(labelEl);
    chip.appendChild(document.createTextNode(' · '));
    chip.appendChild(timeEl);
    // Tick the SAME timer node every second; one interval per in-flight entry.
    // Clear any prior interval the entry held (defensive — a re-render should no
    // longer recreate the chip, but resume paths must not leak intervals).
    if (entry._timer) clearTimer(entry._timer);
    const t = setInterval(() => {
      timeEl.textContent = fmtElapsed(Date.now() - startedAt);
    }, 1000);
    timers.add(t);
    entry._timer = t;
    // Stash the live nodes so a poll tick can update the queue label in place
    // WITHOUT a full renderHistory() rebuild (which would reset this clock).
    entry._busyLabelEl = labelEl;
    return chip;
  }

  // Update only the in-flight entry's busy chip (queue label) without
  // rebuilding the history DOM — keeps the elapsed clock + its single interval
  // alive across poll ticks. Falls back to a full render only if the live node
  // is gone (e.g. the entry has not been rendered yet).
  function updateBusyChip(entry) {
    if (entry._busyLabelEl && entry._busyLabelEl.isConnected) {
      entry._busyLabelEl.textContent = busyLabel(entry);
      return;
    }
    renderHistory();
  }

  // Stop an entry's elapsed-timer interval once it settles (done/error) so the
  // clock stops ticking and the interval doesn't leak past the final render.
  function stopElapsed(entry) {
    if (entry._timer) {
      clearTimer(entry._timer);
      entry._timer = null;
    }
    entry._busyLabelEl = null;
  }

  // --- L4 passages-first preview -----------------------------------------
  // While the LLM composes (35-50s), disclose the passages retrieval already
  // found so the learner sees progress. The pane is a polite live region so a
  // screen reader hears "found passages" once, without per-second chatter.
  function passagesPane(passages, refused) {
    const pane = el('div', {
      class: 'ask-passages',
      role: 'region',
      'aria-live': 'polite',
      'aria-label': refused
        ? 'Passages found (no confident answer yet)'
        : 'Passages found while composing an answer',
    });
    pane.appendChild(el('p', {
      class: 'ask-passages-head muted',
      text: refused
        ? 'Found these passages, but not a confident answer yet…'
        : 'Found these passages — composing an answer…',
    }));
    const list = el('ol', { class: 'ask-passages-list' });
    passages.forEach((p) => {
      const li = el('li', { class: 'ask-passage' });
      const src = ((p && (p.section_heading || p.course_slug)) || '').trim();
      if (src) li.appendChild(el('p', { class: 'ask-passage-src', text: src }));
      li.appendChild(el('p', { class: 'ask-passage-text', text: (p && p.snippet) || '' }));
      list.appendChild(li);
    });
    pane.appendChild(list);
    return pane;
  }

  // --- I6 answer feedback -------------------------------------------------
  function feedbackBar(entry) {
    const wrap = el('div', { class: 'ask-feedback' });
    if (entry._feedbackDone) {
      wrap.appendChild(el('p', {
        class: 'ask-feedback-thanks',
        role: 'status',
        text: 'Thanks for your feedback.',
      }));
      return wrap;
    }
    const labelId = uid('ask-fb-l');
    wrap.appendChild(el('p', { id: labelId, class: 'ask-feedback-label', text: 'Was this answer helpful?' }));
    const group = el('div', {
      class: 'ask-feedback-btns',
      role: 'group',
      'aria-labelledby': labelId,
    });
    const up = el('button', {
      type: 'button',
      class: 'ask-feedback-up',
      'aria-label': 'Yes, this answer was helpful',
      text: '👍 Helpful',
    });
    const down = el('button', {
      type: 'button',
      class: 'ask-feedback-down',
      'aria-label': 'No, this answer was not helpful',
      text: '👎 Not helpful',
    });
    group.appendChild(up);
    group.appendChild(down);
    const commentId = uid('ask-fb-c');
    const commentLabel = el('label', {
      class: 'ask-feedback-comment-label',
      for: commentId,
      text: 'Add a comment (optional)',
    });
    const comment = el('textarea', {
      id: commentId,
      class: 'ask-feedback-comment',
      rows: '2',
    });
    up.addEventListener('click', () => sendFeedback(entry, 'up', comment.value));
    down.addEventListener('click', () => sendFeedback(entry, 'down', comment.value));
    wrap.appendChild(group);
    wrap.appendChild(commentLabel);
    wrap.appendChild(comment);
    return wrap;
  }

  async function sendFeedback(entry, verdict, comment) {
    // Optimistic: mark submitted so a double-click can't double-post; the
    // re-render swaps the controls for the thank-you note.
    entry._feedbackDone = true;
    renderHistory();
    try {
      await apiJSON('/api/learn/feedback', 'POST', {
        slug,
        ask_id: entry.ask_id || null,
        verdict,
        comment: (comment || '').trim(),
      });
      announce('Thanks — your feedback was recorded.');
    } catch (_) {
      // Roll back so the learner can retry; announce the failure politely.
      entry._feedbackDone = false;
      renderHistory();
      announce('Sorry, we could not record your feedback. Please try again.');
    }
  }

  // --- citation interception ---------------------------------------------
  function wireCitations(scope) {
    scope.querySelectorAll('a[href^="/api/learn/source/"]').forEach((a) => {
      // Turn the "View in course" source link into an in-context button: parse
      // item_path + fragment from the server-built href and route to the
      // content pane instead of navigating away. The visible text/role stays
      // link-like but we override activation so answer + source show
      // simultaneously.
      a.setAttribute('role', 'button');
      a.classList.add('ask-cite');
      const onActivate = (e) => {
        e.preventDefault();
        const { item, fragment } = parseSourceHref(a.getAttribute('href'));
        if (item && typeof loadCitation === 'function') loadCitation(item, fragment);
      };
      a.addEventListener('click', onActivate);
      a.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') onActivate(e);
      });
    });
    // Source-side (DART) main citation links point at the anchored
    // source-doc URL (lands on the cited block). Route them into the
    // content pane like course-page citations — the iframe honours the
    // #dart-<block> fragment — instead of letting the <a> navigate the SPA.
    // Provenance links (target=_blank) are excluded: they keep opening in a
    // new tab.
    scope.querySelectorAll('a[href*="/source-doc?"]:not([target])').forEach((a) => {
      a.setAttribute('role', 'button');
      a.classList.add('ask-cite');
      const onActivate = (e) => {
        e.preventDefault();
        if (typeof loadSourceDoc === 'function') loadSourceDoc(a.getAttribute('href'));
      };
      a.addEventListener('click', onActivate);
      a.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') onActivate(e);
      });
    });
    // B4 provenance disclosure: each citation may carry a "Provenance"
    // <button aria-controls=...> over a hidden <ul> with the source block id +
    // one PDF-page deep-link per page. Wire the disclosure toggle. The PDF
    // links are server-built with target=_blank rel=noopener — we deliberately
    // do NOT intercept them (the original PDF page opens in a new tab, simpler
    // and keeps the content pane on the course page).
    wireProvenanceToggles(scope);
  }

  function wireProvenanceToggles(scope) {
    scope.querySelectorAll('.src-detail-toggle').forEach((btn) => {
      const panelId = btn.getAttribute('aria-controls');
      const panel = panelId ? scope.querySelector(`#${cssEscape(panelId)}`) : null;
      const toggle = () => {
        const expanded = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', expanded ? 'false' : 'true');
        if (panel) panel.hidden = expanded;
      };
      btn.addEventListener('click', (e) => { e.preventDefault(); toggle(); });
    });
  }

  // --- submit + poll ------------------------------------------------------
  async function onSubmit(e) {
    if (e) e.preventDefault();
    const query = input.value.trim();
    if (!query || busy) return;
    busy = true;
    submitBtn.disabled = true;

    const entry = { query, status: 'pending', startedAt: Date.now() };
    history.push(entry);
    saveHistory(slug, history);
    renderHistory();
    input.value = '';
    announce('Question submitted. Working on an answer…');

    // L3: when the "search all courses" toggle is visible, its state is an
    // EXPLICIT override that wins over the env default (checked → true, unchecked
    // → false). When hidden (single-course library) send nothing (env resolves).
    const body = { slug, query };
    if (!libWideWrap.hidden) body.library_wide = libWideInput.checked;

    let job;
    try {
      job = await apiJSON('/api/learn/ask-jobs', 'POST', body);
    } catch (err) {
      finishError(entry, err);
      busy = false;
      submitBtn.disabled = false;
      return;
    }
    entry.ask_id = job.ask_id;
    entry.queue_position = job.queue_position;
    saveHistory(slug, history);
    renderHistory();
    pollJob(entry);
  }

  function pollJob(entry) {
    if (!entry.ask_id) return;
    const t = setInterval(async () => {
      let rec;
      try {
        rec = await api(`/api/learn/ask-jobs/${encodeURIComponent(entry.ask_id)}`);
      } catch (err) {
        clearTimer(t);
        finishError(entry, err);
        busy = false;
        submitBtn.disabled = false;
        return;
      }
      if (rec.status === 'pending' || rec.status === 'running') {
        entry.queue_position = rec.queue_position;
        // L4: the first tick that carries retrieved passages does ONE full
        // render to surface the passages-first pane (transient `_passages` — not
        // persisted; re-fetched on resume). Subsequent ticks update only the
        // busy chip in place so the elapsed clock never resets (the flicker bug).
        if (
          Array.isArray(rec.passages) && rec.passages.length && !entry._passagesShown
        ) {
          entry._passages = rec.passages;
          entry._passagesRefused = !!rec.passages_refused;
          entry._passagesShown = true;
          renderHistory();
          announce('Found relevant passages. Composing an answer…');
        } else {
          // Update ONLY the busy chip's queue label in place — do NOT call
          // renderHistory(), which would destroy the chip and reset its elapsed
          // clock to 0s on every 1.5s tick (the timer-flicker bug).
          updateBusyChip(entry);
        }
        return;
      }
      clearTimer(t);
      stopElapsed(entry);
      if (rec.status === 'done') {
        entry.status = 'done';
        entry.html = rec.html;
        entry.answer = rec.answer;
        announce('Answer ready.');
      } else {
        entry.status = 'error';
        entry.html = rec.html || '';
        entry.error = rec.error;
        announce('We could not answer that question.');
      }
      saveHistory(slug, history);
      renderHistory();
      busy = false;
      submitBtn.disabled = false;
    }, POLL_MS);
    timers.add(t);
  }

  function finishError(entry, err) {
    stopElapsed(entry);
    entry.status = 'error';
    entry.error = err instanceof ApiError ? err.error : 'ask_failed';
    entry.html = (err && err.html) || '';
    saveHistory(slug, history);
    renderHistory();
    announce('We could not answer that question.');
  }

  // Resume any still-pending jobs after a navigation / refresh.
  function resumePending() {
    let resumed = false;
    history.forEach((entry) => {
      if ((entry.status === 'pending' || entry.status === 'running') && entry.ask_id) {
        resumed = true;
        busy = true;
        submitBtn.disabled = true;
        pollJob(entry);
      }
    });
    if (resumed) announce('Resuming a question in progress…');
  }

  function announce(msg) { live.textContent = msg; }

  function clearTimer(t) { clearInterval(t); timers.delete(t); }

  // --- collapse / focus ---------------------------------------------------
  function setCollapsed(on) {
    collapsed = on;
    root.classList.toggle('collapsed', on);
    body.hidden = on;
    toggleBtn.setAttribute('aria-expanded', on ? 'false' : 'true');
    toggleBtn.textContent = on ? 'Show Ask panel' : 'Hide Ask panel';
  }

  toggleBtn.addEventListener('click', () => {
    setCollapsed(!collapsed);
    if (!collapsed) input.focus();
  });
  form.addEventListener('submit', onSubmit);
  submitBtn.addEventListener('click', onSubmit);
  // Cmd/Ctrl+Enter submits from the textarea (keeps Enter for newlines).
  input.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') onSubmit(e);
  });

  // --- boot ---------------------------------------------------------------
  renderHistory();
  applyGate();
  resumePending();

  return {
    root,
    focus() { if (!collapsed) input.focus(); else toggleBtn.focus(); },
    destroy() {
      timers.forEach((t) => clearInterval(t));
      timers.clear();
    },
  };
}

/** Escape an id for use in a CSS attribute selector (CSS.escape fallback). */
function cssEscape(id) {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') return CSS.escape(id);
  // Panel ids are minted server-side from [a-zA-Z0-9_-]; a conservative
  // fallback escapes anything outside that set.
  return String(id).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

/** Parse item_path + heading fragment from a /api/learn/source/<slug>?... href. */
export function parseSourceHref(href) {
  let item = '';
  let fragment = '';
  try {
    // Relative href → resolve against current origin for URL parsing.
    const u = new URL(href, window.location.origin);
    item = u.searchParams.get('item_path') || '';
    fragment = (u.hash || '').replace(/^#/, '');
  } catch (_) {
    // Fallback hand-parse (no URL support / odd input).
    const m = /item_path=([^&#]*)/.exec(href || '');
    if (m) { try { item = decodeURIComponent(m[1]); } catch (_) { item = m[1]; } }
    const h = /#(.+)$/.exec(href || '');
    if (h) fragment = h[1];
  }
  return { item, fragment };
}
