/* Shared DOM helpers — vanilla ES module, NO build step.
 *
 * Extracted from gui/static/app.js (the operator SPA) so the Studio + Dev +
 * Learn surfaces share one tiny DOM toolkit. app.js keeps its own copies for
 * now (the full /dev/ port is a later task); these are the canonical versions
 * new surfaces import. Keep the `el` attribute semantics identical to app.js
 * so a future app.js port is a drop-in.
 */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Create an element. attrs: class|html|text|dataset|on<Event>|<attr>. */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, '');
    else if (v === false) { /* skip */ }
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return node;
}

/** Remove all children of `node`; returns `node`. */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

let _uid = 0;
/** A stable-per-session unique id (for label/control pairing). */
export function uid(prefix = 'f') {
  return `${prefix}-${Date.now().toString(36)}-${(_uid++).toString(36)}`;
}
