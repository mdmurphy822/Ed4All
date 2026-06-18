/* Ed4All Studio — persona switcher (GUI redesign Phase 2 follow-up §4.3/§8).
 *
 * The header nav that makes Learner + Advanced discoverable from the Author
 * shell: "Author · Learner · Advanced".
 *
 *   - Author   → the current Studio shell (#/). Marked aria-current="page" while
 *                the Author surface is active (always — Studio IS Author).
 *   - Learner  → /learn/  (the learner surface; a real same-origin navigation).
 *   - Advanced → /advanced/ (the gated operator SPA), with a graceful-401
 *                explainer: we PROBE /advanced/ first; on 401 we DON'T dump the
 *                author into a raw browser 401 — we show a small in-Studio
 *                explainer dialog ("Advanced mode … is protected — enter the
 *                operator token to continue") whose primary action proceeds to
 *                /advanced/, where the operator SPA's OWN token overlay takes
 *                over. On 200 (no token / loopback default) we navigate straight
 *                through.
 *
 * The switcher markup lives in index.html (server-rendered, so it's keyboard-
 * reachable before JS boots + visible in the static a11y gate). This module only
 * UPGRADES the Advanced link: it intercepts the click to probe + explain.
 *
 * A11y: the switcher is a <nav aria-label="Switch view"> of links with
 * aria-current on the active persona. The explainer is a focus-managed
 * role=dialog (aria-modal + labelled + ESC/close + focus trap), built exactly
 * like studio.js::openDeleteDialog so the established modal pattern is reused.
 */

import { el, uid } from '/shared/dom.js';

/**
 * Wire the (server-rendered) persona switcher's Advanced link.
 * @param {Document|HTMLElement} [root=document]
 */
export function initPersonaSwitcher(root = document) {
  const advanced = root.querySelector('.persona-switcher [data-persona="advanced"]');
  if (!advanced) return;
  advanced.addEventListener('click', (e) => {
    // Plain left-click only — honor modifier/middle clicks (open-in-new-tab).
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    probeAndGo(advanced.getAttribute('href') || '/advanced/');
  });
}

/** Probe the Advanced surface; 401 → explainer, otherwise navigate through. */
async function probeAndGo(href) {
  let status = 0;
  try {
    const resp = await fetch(href, { method: 'GET', credentials: 'same-origin', redirect: 'manual' });
    status = resp.status;
  } catch (_) {
    // Network/opaque error → fall through to a straight navigation (the SPA's
    // own overlay still handles auth; never strand the author on a probe error).
    window.location.assign(href);
    return;
  }
  if (status === 401 || status === 403) {
    openAdvancedExplainer(href);
    return;
  }
  window.location.assign(href);
}

/**
 * The graceful-401 explainer: a labelled, focus-trapped modal dialog whose
 * primary action proceeds to /advanced/ (where the operator SPA's token overlay
 * takes over). Mirrors the studio.js delete-dialog modal pattern.
 */
export function openAdvancedExplainer(href) {
  const titleId = uid('adv-title');
  const descId = uid('adv-desc');

  const proceedBtn = el('a', {
    class: 'btn primary',
    href,
    text: 'Continue to Advanced',
  });
  const cancelBtn = el('button', { type: 'button', class: 'btn' }, ['Stay in Author']);

  const dialog = el('div', {
    class: 'modal-dialog',
    role: 'dialog',
    'aria-modal': 'true',
    'aria-labelledby': titleId,
    'aria-describedby': descId,
  }, [
    el('h2', { id: titleId, text: 'Advanced mode is protected' }),
    el('p', { id: descId }, [
      'Advanced mode manages API keys, model routing, and run launching, and is ',
      'protected — enter the operator token to continue. You’ll be taken to ',
      'Advanced, where you can enter the token.',
    ]),
    el('div', { class: 'modal-actions' }, [cancelBtn, proceedBtn]),
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

  // Focus trap + ESC cancel (capture phase so it wins over view handlers).
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
  // Proceeding is a real same-origin navigation; close the dialog first so a
  // back-navigation doesn't leave a stale overlay.
  proceedBtn.addEventListener('click', () => { close(); });
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });

  document.addEventListener('keydown', onKeydown, true);
  document.body.appendChild(overlay);
  cancelBtn.focus();
  return { close };
}

export default initPersonaSwitcher;
