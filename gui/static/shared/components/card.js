/* card — courseCard() / runCard(). Vanilla ES module, NO build, PURE PRESENTATION.
 *
 * The two dashboard cards: a course library entry and a run-history entry.
 * Each card's title is an <h3> (so the host view keeps its single <h1>); a card
 * that links somewhere is an <a class="card"> (mirroring the studio library
 * grid), otherwise a plain <article>. Status is shown via the shared statusPill
 * (glyph + text, never color-only).
 *
 * PURE PRESENTATION: takes data as args, calls no API. Action endpoints are
 * wired by the HOST page (the security contract), never by this kit.
 */

import { el } from '../dom.js';
import { statusPill } from './pill.js';

/**
 * A course library card.
 * @param {Object} course
 * @param {string} course.title
 * @param {string} [course.href]   viewer hash route (omit for a non-link card).
 * @param {string} [course.meta]   e.g. "12 pages · 12.3 MB".
 * @param {boolean} [course.askReady]
 * @returns {HTMLElement}
 */
export function courseCard(course = {}) {
  const title = course.title || 'Untitled course';
  const kids = [
    el('h3', { class: 'card-title', text: title }),
    course.meta ? el('p', { class: 'card-meta', text: course.meta }) : null,
    course.askReady ? el('span', { class: 'card-badge', text: 'Ask-ready' }) : null,
  ];
  if (course.href) {
    return el('a', { class: 'card course-card', href: course.href }, kids);
  }
  return el('article', { class: 'card course-card' }, kids);
}

/**
 * A run-history / live-run card.
 * @param {Object} run
 * @param {string} run.title         course name or run id.
 * @param {string} run.status        a run status (queued|running|completed|…).
 * @param {string} [run.href]        re-open route.
 * @param {string} [run.meta]        e.g. "textbook_to_course · 28m".
 * @returns {HTMLElement}
 */
export function runCard(run = {}) {
  const title = run.title || run.run_id || 'Run';
  const head = el('div', { class: 'card-head' }, [
    el('h3', { class: 'card-title', text: title }),
    statusPill(`run:${run.status}`),
  ]);
  const kids = [
    head,
    run.meta ? el('p', { class: 'card-meta', text: run.meta }) : null,
  ];
  if (run.href) {
    return el('a', { class: 'card run-card', href: run.href }, kids);
  }
  return el('article', { class: 'card run-card' }, kids);
}

export default { courseCard, runCard };
