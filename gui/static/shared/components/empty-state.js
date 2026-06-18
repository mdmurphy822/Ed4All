/* emptyState — message + a REQUIRED CTA slot. Vanilla ES module, NO build.
 *
 * Closes the verified "empty state with no next step" gaps: an empty zone
 * (Library, Retrieval, Recent builds) always offers a single clear next action.
 * The CTA is REQUIRED — calling without one throws, so an empty state can never
 * ship a dead end.
 *
 * PURE PRESENTATION: takes data as args, calls no API. The CTA's behavior is
 * supplied by the host (an href OR an onClick), never an endpoint in this kit.
 */

import { el } from '../dom.js';

/**
 * Build an empty-state block.
 * @param {Object} opts
 * @param {string} opts.title           heading (rendered as <h2>).
 * @param {string} [opts.message]        supporting line.
 * @param {Object} opts.cta             REQUIRED call-to-action.
 * @param {string} opts.cta.label       button/link text.
 * @param {string} [opts.cta.href]      hash/route — renders an <a class="btn primary">.
 * @param {Function} [opts.cta.onClick] click handler — renders a <button class="btn primary">.
 * @returns {HTMLElement} a <section class="empty-state">.
 * @throws if no CTA (label + href|onClick) is supplied.
 */
export function emptyState({ title = 'Nothing here yet', message = '', cta } = {}) {
  if (!cta || !cta.label || (!cta.href && typeof cta.onClick !== 'function')) {
    throw new Error('emptyState requires a CTA with a label and an href or onClick');
  }
  const action = cta.href
    ? el('a', { class: 'btn primary empty-cta', href: cta.href, text: cta.label })
    : el('button', { type: 'button', class: 'btn primary empty-cta', text: cta.label, onclick: cta.onClick });

  return el('section', { class: 'empty-state' }, [
    el('h2', { class: 'empty-title', text: title }),
    message ? el('p', { class: 'empty-message', text: message }) : null,
    action,
  ]);
}

export default emptyState;
