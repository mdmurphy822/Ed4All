/* Learner quiz-panel integration (Q6 + L1).
 *
 * Wires the tested quiz.js front-end module into the learner page's Quizzes
 * section. This module owns ONLY the panel plumbing — the quiz-item rendering,
 * XSS-escaping, and server-side grading round-trip all live in quiz.js (which is
 * covered at the service layer). Contract exercised here:
 *   - GET /api/learn/quiz/{slug}                  (listQuizzes)  -> [{assessment_id, title, question_count}]
 *   - one-call mount (loadQuiz + renderQuiz + wireSubmit + last-attempt)  (mountQuiz)
 *
 * The panel reuses the ask form's #course-sel as the single source of truth for
 * the selected course (no duplicate course fetch). "Show quizzes" reads the
 * current selection; switching course clears the stale panel.
 *
 * Empty-list and fetch-error states are designed, not accidental: an empty list
 * shows an explicit "No quizzes available…" line, a list-fetch failure and a
 * quiz-open failure each show plain-language recovery copy. #quiz-list-region is
 * aria-live="polite" (NOT role=status) so it never becomes a second role=status
 * live region alongside the ask #status region.
 *
 * Vanilla ES module — no framework, no build step.
 */
import { listQuizzes, mountQuiz } from "/learn/quiz.js";

// (ES modules are strict-mode by default.)

// Mirror Python html.escape(quote=True): & < > " '. All dynamic quiz-list text
// (titles, ids, counts) is escaped before it touches innerHTML.
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

function $(id) {
  return document.getElementById(id);
}

// Render the quiz list for `slug` into #quiz-list-region. Empty + error states
// are explicit. Clears any previously-mounted quiz in #quiz-host.
async function loadList(slug) {
  const region = $("quiz-list-region");
  const host = $("quiz-host");
  if (!region) return;
  if (host) host.innerHTML = "";
  if (!slug) {
    region.textContent = "Choose a course above to see its quizzes.";
    return;
  }
  region.textContent = "Loading quizzes…";
  let quizzes;
  try {
    quizzes = await listQuizzes(slug);
  } catch (_err) {
    region.textContent =
      "Could not load quizzes for this course. Please tell the session facilitator.";
    return;
  }
  if (!Array.isArray(quizzes) || quizzes.length === 0) {
    region.textContent = "No quizzes available for this course yet.";
    return;
  }
  const items = quizzes
    .map(function (q) {
      const aid = escapeHtml(q.assessment_id);
      const title = escapeHtml(q.title || q.assessment_id);
      const count = Number(q.question_count) || 0;
      const noun = count === 1 ? "question" : "questions";
      return (
        '<li class="quiz-pick-item">' +
        '<button type="button" class="quiz-pick" data-aid="' +
        aid +
        '">' +
        title +
        ' <span class="quiz-count">(' +
        escapeHtml(String(count)) +
        " " +
        noun +
        ")</span>" +
        "</button>" +
        "</li>"
      );
    })
    .join("");
  region.innerHTML =
    '<p class="quiz-list-label" id="quiz-list-label">Select a quiz:</p>' +
    '<ul class="quiz-picker" aria-labelledby="quiz-list-label">' +
    items +
    "</ul>";
  region.querySelectorAll(".quiz-pick").forEach(function (btn) {
    btn.addEventListener("click", function () {
      openQuiz(slug, btn.getAttribute("data-aid"));
    });
  });
}

// Mount one quiz into #quiz-host. mountQuiz owns render + submit-wiring + the
// server-side grade round-trip; a load failure shows plain-language copy.
async function openQuiz(slug, assessmentId) {
  const host = $("quiz-host");
  if (!host) return;
  host.innerHTML = '<p class="quiz-loading">Loading quiz…</p>';
  try {
    await mountQuiz(host, slug, assessmentId);
  } catch (_err) {
    host.innerHTML = "";
    host.textContent =
      "Could not open this quiz. Please tell the session facilitator.";
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const btn = $("quiz-load-btn");
  const sel = $("course-sel");
  if (btn) {
    btn.addEventListener("click", function () {
      loadList(sel ? sel.value : "");
    });
  }
  // Switching course invalidates the shown list + any open quiz.
  if (sel) {
    sel.addEventListener("change", function () {
      const region = $("quiz-list-region");
      const host = $("quiz-host");
      if (region) region.textContent = "";
      if (host) host.innerHTML = "";
    });
  }
});
