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
 * Vanilla ES module. The only import is the PRESENTATION-ONLY progress ring
 * from the shared kit (it takes args, calls no API, wires no endpoint — the
 * locked-down learner bundle structurally cannot pull in a run-control route).
 */
import { progressRing } from "/shared/components/ring.js";

// (ES modules are strict-mode by default — no "use strict" pragma needed.)

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

  // HTTP QUERY method for the safe + idempotent, body-carrying ask request.
  // QUERY is standardized as RFC 10008 ("The HTTP QUERY Method", Standards Track,
  // June 2026 — graduated from draft-ietf-httpbis-safe-method-w-body). The server
  // keeps POST as a deprecated alias; we send QUERY and downgrade ONCE to POST for
  // the session if a client/proxy rejects the method (405/501, or a network-layer
  // method block that throws). An intentional abort (the 150s timeout) is never
  // treated as a rejection — it bubbles unchanged.
  var _queryMethodOk = true;
  async function askFetch(payload, signal) {
    var methods = _queryMethodOk ? ["QUERY", "POST"] : ["POST"];
    for (var i = 0; i < methods.length; i++) {
      var method = methods[i];
      try {
        var res = await fetch("/api/learn/ask", {
          method: method,
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(payload),
          signal: signal,
        });
        if (method === "QUERY" && (res.status === 405 || res.status === 501)) {
          _queryMethodOk = false; // proxy/server rejects QUERY → downgrade + retry
          continue;
        }
        return res;
      } catch (err) {
        if (err && err.name === "AbortError") throw err; // intentional timeout
        if (method === "QUERY") {
          _queryMethodOk = false; // network-layer method block → downgrade + retry
          continue;
        }
        throw err;
      }
    }
  }

  function $(id) {
    return document.getElementById(id);
  }

  // --- iframe-embed widget wiring (LMS embed contract) ----------------------
  // The learner page doubles as an embeddable "Ask the Course" widget. The host
  // page (e.g. an OpenOLAT course element) frames `/learn/?course=<slug>&embed=1`:
  //   - `?course=<slug>` pins the widget to ONE course: the course <select> is
  //     pre-selected + disabled (a learner can't switch it), so the widget shows
  //     a single-course ask view. The pinned slug is passed to /api/learn/ask as
  //     AskRequest.slug exactly as a manual pick would be.
  //   - `?embed=1` (truthy) is COMPACT widget mode: the course picker field and
  //     the Quizzes panel are hidden (CSS), leaving a pure ask box for the frame.
  // Both are additive + presentational; with neither param the page is
  // byte-identical to the standalone learner surface.
  function getParam(name) {
    try {
      return new URLSearchParams(window.location.search).get(name);
    } catch (_e) {
      return null;
    }
  }
  function isTruthy(v) {
    if (!v) return false;
    return ["1", "true", "yes", "on"].indexOf(String(v).toLowerCase()) !== -1;
  }

  // Pin the course <select> to `slug`: ensure the option exists, select it, and
  // disable the control so the widget stays single-course. Runs AFTER loadCourses
  // resolves so it wins over the fetched option list (and over the empty/error
  // placeholder) — the server still validates the slug on /ask, so a stale pin
  // surfaces an honest 404 fragment rather than a silently-wrong course.
  function applyCoursePin(slug) {
    var sel = $("course-sel");
    if (!sel) return;
    var found = false;
    for (var i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === slug) {
        found = true;
        break;
      }
    }
    if (!found) {
      var opt = document.createElement("option");
      opt.value = slug;
      opt.textContent = slug;
      sel.appendChild(opt);
    }
    sel.value = slug;
    sel.disabled = true;
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

  // Visual "thinking" reassurance during the ask window — an indeterminate
  // progress ring + an elapsed counter, BOTH aria-hidden decoration (OUTSIDE
  // the live region), so SR users get no per-second chatter. The ring adds NO
  // live-region text: the single polite status announcement (busy on start,
  // arrival on finish) is the only thing a screen reader hears.
  var elapsedTimer = null;
  function startElapsed() {
    var start = Date.now();
    var el = $("elapsed");
    el.textContent = "0s elapsed";
    // Indeterminate "running" ring (no label → no visually-hidden text, so it
    // stays pure decoration; the host page owns the one live region). Motion
    // is reduced-motion-safe via tokens.css's @keyframes spin freeze.
    var ringHost = $("thinking-ring");
    if (ringHost) {
      ringHost.textContent = "";
      ringHost.appendChild(progressRing({ state: "running", size: 24 }));
    }
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
    var ringHost = $("thinking-ring");
    if (ringHost) {
      ringHost.textContent = "";
    }
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
      var res = await askFetch(
        { slug: slug, query: query, engine: "auto" },
        controller.signal
      );

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
    var pinnedSlug = getParam("course");
    if (isTruthy(getParam("embed"))) {
      document.body.classList.add("embed"); // compact widget: hides picker + quizzes
    }
    // loadCourses() is async; apply the course pin only after it resolves so the
    // pinned selection wins over the fetched option list.
    Promise.resolve(loadCourses()).then(function () {
      if (pinnedSlug) applyCoursePin(pinnedSlug);
    });
    $("ask-form").addEventListener("submit", onSubmit);
  });
