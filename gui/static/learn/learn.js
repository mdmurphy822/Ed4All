/* Learner answer surface — progressive enhancement only.
 *
 * Contract (frozen, plan §3 / §4.5):
 *  - GET  /api/learn/courses -> [{slug, chunk_count}]
 *  - POST /api/learn/ask     -> {answer, html}      (200 for ALL six statuses)
 *                            -> {error, detail, html} (typed-error responses)
 *
 * The server renders the answer HTML; this script does fetch + fragment swap +
 * focus management ONLY. It NEVER templates answer content from JSON — the
 * `html` field from the response is inserted verbatim into #answer-region so a
 * learner sees exactly what the WCAG gate validated. The only client-authored
 * copy is the network/abort fallback below, mirrored verbatim from the
 * server's STATUS_COPY `error_generic` entry.
 *
 * Vanilla JS, no dependencies.
 */
(function () {
  "use strict";

  var ABORT_MS = 150000; // > backend 120s timeout (ED4ALL_ANSWER_TIMEOUT_SECONDS)

  // Live-region copy (mirrors server STATUS_COPY headings; ONE announcement
  // per state transition, per D6).
  var MSG = {
    busy: "Searching the course materials. This can take up to a minute.",
    answered: "Answer ready.",
    no_answer: "No answer found.",
    error: "The course assistant hit a problem.",
  };

  // The ONLY client-authored answer fragment — used when the network fails or
  // the request is aborted (no server response to render). Mirrored verbatim
  // from the server's STATUS_COPY `error_generic` so copy stays single-sourced.
  var CLIENT_ERROR_HTML =
    '<section class="answer" data-status="error_generic" aria-labelledby="answer-h">' +
    '<h2 id="answer-h" tabindex="-1">Something went wrong</h2>' +
    "<p>The course assistant hit a problem and couldn&#x27;t answer. " +
    "Please tell the session facilitator.</p>" +
    "</section>";

  function $(id) {
    return document.getElementById(id);
  }

  function setStatus(text) {
    // Write ONE message to the polite live region.
    $("status").textContent = text;
  }

  function setBusy(on) {
    var region = $("answer-region");
    region.setAttribute("aria-busy", on ? "true" : "false");
    var btn = $("ask-btn");
    btn.disabled = on;
    // Not color-only: the label changes too.
    btn.textContent = on ? "Asking…" : "Ask";
  }

  // Visual-only elapsed counter — OUTSIDE the live region (aria-hidden), so SR
  // users get no per-second chatter.
  var elapsedTimer = null;
  function startElapsed() {
    var start = Date.now();
    var el = $("elapsed");
    el.textContent = "0s elapsed";
    elapsedTimer = window.setInterval(function () {
      var secs = Math.round((Date.now() - start) / 1000);
      el.textContent = secs + "s elapsed";
    }, 1000);
  }
  function stopElapsed() {
    if (elapsedTimer !== null) {
      window.clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
    $("elapsed").textContent = "";
  }

  // Swap the server-rendered fragment in, then move focus to the answer h2.
  function renderFragment(html, liveMessage) {
    var region = $("answer-region");
    region.innerHTML = html; // server-rendered + html.escape'd; gate-validated
    setStatus(liveMessage);
    var heading = region.querySelector("#answer-h");
    if (heading) {
      heading.focus(); // h2 carries tabindex="-1"
    }
  }

  function liveMessageForStatus(status) {
    if (status === "answered" || status === "answered_with_warnings") {
      return MSG.answered;
    }
    if (
      status === "refused_low_confidence" ||
      status === "refused_not_in_course" ||
      status === "blocked_invalid_citation" ||
      status === "blocked_citation_gate"
    ) {
      return MSG.no_answer;
    }
    return MSG.error;
  }

  async function loadCourses() {
    var sel = $("course-sel");
    try {
      var res = await fetch("/api/learn/courses", {
        headers: { Accept: "application/json" },
      });
      if (!res.ok) throw new Error("courses " + res.status);
      var list = await res.json();
      sel.innerHTML = "";
      if (!Array.isArray(list) || list.length === 0) {
        var none = document.createElement("option");
        none.value = "";
        none.textContent = "No courses available";
        sel.appendChild(none);
        sel.disabled = true;
        return;
      }
      list.forEach(function (c) {
        var opt = document.createElement("option");
        opt.value = c.slug;
        opt.textContent = c.slug;
        sel.appendChild(opt);
      });
    } catch (err) {
      sel.innerHTML = "";
      var opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Could not load courses";
      sel.appendChild(opt);
      sel.disabled = true;
    }
  }

  async function onSubmit(evt) {
    evt.preventDefault();
    var slug = $("course-sel").value;
    var query = $("q-in").value.trim();
    if (!slug || !query) {
      setStatus("Please choose a course and type a question.");
      return;
    }

    setBusy(true);
    setStatus(MSG.busy);
    startElapsed();

    var controller = new AbortController();
    var abortTimer = window.setTimeout(function () {
      controller.abort();
    }, ABORT_MS);

    try {
      var res = await fetch("/api/learn/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ slug: slug, query: query, engine: "auto" }),
        signal: controller.signal,
      });

      var body = null;
      try {
        body = await res.json();
      } catch (parseErr) {
        body = null;
      }

      // Uniform swap path: the response (ok OR typed-error) carries `html`.
      if (body && typeof body.html === "string") {
        var status =
          body.answer && body.answer.status
            ? body.answer.status
            : body.error || "error_generic";
        renderFragment(body.html, liveMessageForStatus(status));
      } else {
        // No renderable fragment came back — fall back to client error copy.
        renderFragment(CLIENT_ERROR_HTML, MSG.error);
      }
    } catch (err) {
      // Network failure or 150s abort — render the client-authored error copy.
      renderFragment(CLIENT_ERROR_HTML, MSG.error);
    } finally {
      window.clearTimeout(abortTimer);
      stopElapsed();
      setBusy(false);
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    loadCourses();
    $("ask-form").addEventListener("submit", onSubmit);
  });
})();
