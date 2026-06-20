/* Learner quiz-taking module (W10 Phase 3, spec §5.1).
 *
 * Progressive-enhancement, vanilla ES module — no framework, no build step.
 *
 * Contract (learner-open, no operator token):
 *   - GET  /api/learn/quiz/{slug}                       -> [{assessment_id, title, question_count}]
 *   - GET  /api/learn/quiz/{slug}/{assessment_id}       -> {assessment_id, title, questions:[...]}  (NO answer key)
 *   - POST /api/learn/quiz/{slug}/{assessment_id}/grade -> {score, max_score, per_item:[{question_id, correct, feedback}]}
 *
 * Anti-XSS: the quiz JSON carries RAW stem/choice text (the service does no
 * escaping — see quiz_service docstring). This module escapes EVERY dynamic
 * string with escapeHtml() before it touches innerHTML. The stem is never
 * inserted as markup.
 *
 * NO <form action> POST: selections are collected from inputs and fetch()'d to
 * the same-origin grade endpoint, so no real form submission leaves the page and
 * no answer key is ever shipped to the browser (grading is server-side).
 *
 * W10 §10 ATTEMPT PERSISTENCE: the grade request carries the X-Learner-Id owner
 * key (via learnerFetch), so a graded attempt is persisted server-side and the
 * learner's most recent attempt + score are shown on (re)mount.
 */

import { learnerFetch } from "/learn/learner_id.js";

// (ES modules are strict-mode by default.)

function escapeHtml(s) {
  // Mirror Python html.escape(quote=True): & < > " '.
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

// Single-select (mc / true_false) uses radios; multiple_response uses checkboxes.
function isMultiSelect(type) {
  return type === "multiple_response";
}
// A free-text input (numeric fill-in / short answer) has no choices.
function isTextInput(q) {
  return !q.choices || q.choices.length === 0;
}

export async function listQuizzes(slug) {
  const res = await fetch("/api/learn/quiz/" + encodeURIComponent(slug), {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error("quiz list " + res.status);
  return res.json();
}

export async function loadQuiz(slug, assessmentId) {
  const res = await fetch(
    "/api/learn/quiz/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(assessmentId),
    { headers: { Accept: "application/json" } }
  );
  if (!res.ok) throw new Error("quiz load " + res.status);
  return res.json();
}

export async function gradeQuiz(slug, assessmentId, answers) {
  // learnerFetch attaches the X-Learner-Id owner key so the attempt persists.
  const res = await learnerFetch(
    "/api/learn/quiz/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(assessmentId) +
      "/grade",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ answers: answers }),
    }
  );
  if (!res.ok) throw new Error("quiz grade " + res.status);
  return res.json();
}

// Fetch the caller's own persisted attempts (oldest first). Owner-scoped via
// the X-Learner-Id header; returns [] for a learner with no history.
export async function loadAttempts(slug, assessmentId) {
  const res = await learnerFetch(
    "/api/learn/quiz/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(assessmentId) +
      "/attempts",
    { headers: { Accept: "application/json" } }
  );
  if (!res.ok) return [];
  const body = await res.json().catch(function () {
    return [];
  });
  return Array.isArray(body) ? body : [];
}

// Build the markup for one question. ALL dynamic text is escapeHtml'd.
function renderQuestion(q, index) {
  const name = "q-" + escapeHtml(q.question_id);
  let body;
  if (isTextInput(q)) {
    body =
      '<input type="text" class="quiz-text" ' +
      'name="' +
      name +
      '" ' +
      'aria-label="Your answer" autocomplete="off">';
  } else {
    const inputType = isMultiSelect(q.type) ? "checkbox" : "radio";
    const choices = q.choices
      .map(function (c, i) {
        const cid = escapeHtml(c.id);
        const optId = name + "-opt-" + i;
        return (
          '<li class="quiz-choice">' +
          '<input type="' +
          inputType +
          '" id="' +
          optId +
          '" name="' +
          name +
          '" value="' +
          cid +
          '">' +
          '<label for="' +
          optId +
          '">' +
          escapeHtml(c.text) +
          "</label>" +
          "</li>"
        );
      })
      .join("");
    body = '<ul class="quiz-choices">' + choices + "</ul>";
  }
  return (
    '<li class="quiz-item" data-qid="' +
    escapeHtml(q.question_id) +
    '">' +
    '<fieldset>' +
    "<legend>" +
    (index + 1) +
    ". " +
    escapeHtml(q.stem) +
    "</legend>" +
    body +
    "</fieldset>" +
    "</li>"
  );
}

// Render the whole quiz into `host`. Returns the rendered questions for collect().
export function renderQuiz(host, quiz) {
  const items = (quiz.questions || [])
    .map(function (q, i) {
      return renderQuestion(q, i);
    })
    .join("");
  host.innerHTML =
    '<section class="quiz" aria-labelledby="quiz-h">' +
    '<h2 id="quiz-h" tabindex="-1">' +
    escapeHtml(quiz.title || "Quiz") +
    "</h2>" +
    '<ol class="quiz-list">' +
    items +
    "</ol>" +
    '<button type="button" class="quiz-submit">Submit answers</button>' +
    '<div class="quiz-result" role="status" aria-live="polite"></div>' +
    "</section>";
  const h = host.querySelector("#quiz-h");
  if (h) h.focus();
  return quiz.questions || [];
}

// Collect selections from the rendered inputs into {question_id: selection}.
export function collectAnswers(host, questions) {
  const answers = {};
  (questions || []).forEach(function (q) {
    const name = "q-" + q.question_id;
    if (isTextInput(q)) {
      const el = host.querySelector('input[name="' + cssEscape(name) + '"]');
      answers[q.question_id] = el ? el.value : "";
    } else if (isMultiSelect(q.type)) {
      const els = host.querySelectorAll(
        'input[name="' + cssEscape(name) + '"]:checked'
      );
      answers[q.question_id] = Array.prototype.map.call(els, function (e) {
        return e.value;
      });
    } else {
      const el = host.querySelector(
        'input[name="' + cssEscape(name) + '"]:checked'
      );
      answers[q.question_id] = el ? el.value : "";
    }
  });
  return answers;
}

// Minimal attribute-value escaper for the querySelector name (ids are slug-safe
// from the server, but defend against a stray quote).
function cssEscape(v) {
  return String(v).replace(/"/g, '\\"');
}

// Render the grade result (per-item correctness) below the quiz.
export function renderResult(host, result) {
  const region = host.querySelector(".quiz-result");
  if (!region) return;
  const rows = (result.per_item || [])
    .map(function (item) {
      const mark = item.correct ? "Correct" : "Incorrect";
      return (
        '<li class="quiz-result-item" data-correct="' +
        (item.correct ? "true" : "false") +
        '">' +
        "<strong>" +
        escapeHtml(mark) +
        "</strong> — " +
        escapeHtml(item.feedback || "") +
        "</li>"
      );
    })
    .join("");
  region.innerHTML =
    '<p class="quiz-score">Score: ' +
    escapeHtml(String(result.score)) +
    " / " +
    escapeHtml(String(result.max_score)) +
    "</p>" +
    '<ul class="quiz-result-list">' +
    rows +
    "</ul>";
}

// Wire a fully-rendered quiz: clicking submit collects + grades + renders result.
export function wireSubmit(host, slug, quiz, questions) {
  const btn = host.querySelector(".quiz-submit");
  if (!btn) return;
  btn.addEventListener("click", async function () {
    btn.disabled = true;
    const prev = btn.textContent;
    btn.textContent = "Grading…";
    try {
      const answers = collectAnswers(host, questions);
      const result = await gradeQuiz(slug, quiz.assessment_id, answers);
      renderResult(host, result);
    } catch (err) {
      const region = host.querySelector(".quiz-result");
      if (region) {
        region.textContent =
          "Could not grade your answers. Please tell the session facilitator.";
      }
    } finally {
      btn.disabled = false;
      btn.textContent = prev;
    }
  });
}

// Show the learner's most recent persisted attempt + score (W10 §10). All
// dynamic text is escapeHtml'd. No-op when the learner has no history.
export function renderLastAttempt(host, attempts) {
  const region = host.querySelector(".quiz-result");
  if (!region) return;
  if (!attempts || !attempts.length) return;
  const last = attempts[attempts.length - 1];
  region.innerHTML =
    '<p class="quiz-last-attempt">Your last attempt: ' +
    escapeHtml(String(last.score)) +
    " / " +
    escapeHtml(String(last.max_score)) +
    (last.ts ? " (" + escapeHtml(String(last.ts)) + ")" : "") +
    "</p>";
}

// One-call convenience: load a quiz, render it, wire submit, show last attempt.
export async function mountQuiz(host, slug, assessmentId) {
  const quiz = await loadQuiz(slug, assessmentId);
  const questions = renderQuiz(host, quiz);
  wireSubmit(host, slug, quiz, questions);
  try {
    const attempts = await loadAttempts(slug, assessmentId);
    renderLastAttempt(host, attempts);
  } catch (_e) {
    /* history is best-effort — a fetch failure never blocks the quiz */
  }
  return quiz;
}
