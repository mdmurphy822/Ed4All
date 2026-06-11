/* Shared toast + error-toast — vanilla ES module, NO build step.
 *
 * Extracted from gui/static/app.js. Renders into a `#toasts` container (created
 * lazily if absent so a surface can drop a single <div id="toasts"> or rely on
 * the auto-created one). The container is an aria-live=polite region so a
 * screen reader announces transient status without stealing focus.
 */

import { el } from './dom.js';
import { ApiError } from './api.js';

function _container() {
  let box = document.getElementById('toasts');
  if (!box) {
    box = el('div', { id: 'toasts', class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.appendChild(box);
  }
  return box;
}

/** Show a transient toast. kind: info|success|error. ttl=0 keeps it sticky. */
export function toast(title, detail, kind = 'info', ttl = 6000) {
  const box = _container();
  const t = el('div', { class: `toast ${kind}` }, [
    el('div', { class: 'ttitle', text: title }),
    detail ? el('div', { class: 'tdetail', text: detail }) : null,
  ]);
  t.addEventListener('click', () => t.remove());
  box.appendChild(t);
  if (ttl) setTimeout(() => t.remove(), ttl);
  return t;
}

/** Surface an error (ApiError or any throwable) as an error toast. */
export function toastErr(e, context) {
  if (e instanceof ApiError) {
    toast(context || e.error, e.detail || `(${e.status})`, 'error', 9000);
  } else {
    toast(context || 'Error', String((e && e.message) || e), 'error', 9000);
  }
  // eslint-disable-next-line no-console
  console.error(context || 'error', e);
}
