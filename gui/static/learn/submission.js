/* Learner assignment-submission module (W10 §10 — assignment SUBMISSION).
 *
 * Adds a submit box (text + optional single file) to the read-only assignment
 * view (tasks.js), and shows the learner's own current submission.
 *
 * Contract (learner-open, owner-scoped via X-Learner-Id):
 *   - POST /api/learn/assignment/{slug}/{assignment_id}/submit
 *       multipart/form-data: text=<...> [file=<single upload>]
 *   - GET  /api/learn/assignment/{slug}/{assignment_id}/submission -> the caller's own record (404 if none)
 *
 * SECURITY: text is sanitized SERVER-SIDE on store; the file is validated
 * server-side (extension allowlist .pdf/.txt/.md/.docx, 5MB cap, traversal-safe
 * opaque blob). This module escapeHtml()'s every dynamic string before innerHTML.
 */

import { learnerFetch } from "/learn/learner_id.js";

// Mirror the server allowlist for a friendlier client-side pre-check (the server
// is the authority — this only improves the UX, never the security boundary).
const ALLOWED_EXT = [".pdf", ".txt", ".md", ".docx"];
const MAX_BYTES = 5 * 1024 * 1024;

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

export async function loadSubmission(slug, assignmentId) {
  const res = await learnerFetch(
    "/api/learn/assignment/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(assignmentId) +
      "/submission",
    { headers: { Accept: "application/json" } }
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error("submission load " + res.status);
  return res.json();
}

export async function submitAssignment(slug, assignmentId, text, file) {
  const fd = new FormData();
  fd.append("text", text || "");
  if (file) fd.append("file", file, file.name);
  const res = await learnerFetch(
    "/api/learn/assignment/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(assignmentId) +
      "/submit",
    { method: "POST", body: fd }
  );
  if (!res.ok) {
    const body = await res.json().catch(function () {
      return null;
    });
    throw new Error((body && body.detail) || "submit " + res.status);
  }
  return res.json();
}

function extOf(name) {
  const i = String(name || "").lastIndexOf(".");
  return i >= 0 ? String(name).slice(i).toLowerCase() : "";
}

// Render the submit box + the learner's current submission (if any).
export function renderSubmissionBox(host, current) {
  const have = current && (current.text_sanitized || current.file);
  const fileLine =
    have && current.file
      ? '<p class="sub-file">Attached: ' +
        escapeHtml(current.file.original_filename || current.file.stored_name || "") +
        "</p>"
      : "";
  const currentBlock = have
    ? '<div class="sub-current"><p class="sub-current-h">Your current submission' +
      (current.ts ? " (" + escapeHtml(String(current.ts)) + ")" : "") +
      ":</p><p class="sub-current-text">' +
      escapeHtml(stripTags(current.text_sanitized || "")) +
      "</p>" +
      fileLine +
      "</div>"
    : "";
  host.innerHTML =
    '<section class="submission" aria-labelledby="sub-h">' +
    '<h3 id="sub-h" tabindex="-1">Your submission</h3>' +
    currentBlock +
    '<label for="sub-text">Your response</label>' +
    '<textarea id="sub-text" class="sub-input" rows="5"></textarea>' +
    '<label for="sub-file">Optional file (.pdf, .txt, .md, .docx; max 5 MB)</label>' +
    '<input type="file" id="sub-file" accept=".pdf,.txt,.md,.docx">' +
    '<button type="button" class="sub-submit">Submit</button>' +
    '<div class="sub-status" role="status" aria-live="polite"></div>' +
    "</section>";
}

function stripTags(html) {
  return String(html || "").replace(/<[^>]*>/g, "");
}

const _wired = new WeakSet();
export function wireSubmission(host, slug, assignmentId) {
  if (_wired.has(host)) return;
  _wired.add(host);
  host.addEventListener("click", async function (ev) {
    const btn = ev.target;
    if (!btn || !btn.classList || !btn.classList.contains("sub-submit")) return;
    const textEl = host.querySelector("#sub-text");
    const fileEl = host.querySelector("#sub-file");
    const status = host.querySelector(".sub-status");
    const text = textEl ? textEl.value.trim() : "";
    const file = fileEl && fileEl.files && fileEl.files[0] ? fileEl.files[0] : null;
    if (!text && !file) {
      if (status) status.textContent = "Enter a response before submitting.";
      return;
    }
    // Client-side pre-check (UX only; the server re-validates authoritatively).
    if (file) {
      if (ALLOWED_EXT.indexOf(extOf(file.name)) < 0) {
        if (status) status.textContent = "File type not allowed.";
        return;
      }
      if (file.size > MAX_BYTES) {
        if (status) status.textContent = "File is larger than 5 MB.";
        return;
      }
    }
    btn.disabled = true;
    try {
      await submitAssignment(slug, assignmentId, text, file);
      await refreshSubmission(host, slug, assignmentId);
      const s2 = host.querySelector(".sub-status");
      if (s2) s2.textContent = "Submitted.";
    } catch (err) {
      if (status) status.textContent = "Could not submit: " + escapeHtml(err.message || "");
      btn.disabled = false;
    }
  });
}

export async function refreshSubmission(host, slug, assignmentId) {
  let current = null;
  try {
    current = await loadSubmission(slug, assignmentId);
  } catch (_e) {
    current = null;
  }
  renderSubmissionBox(host, current);
  return current;
}

export async function mountSubmission(host, slug, assignmentId) {
  wireSubmission(host, slug, assignmentId);
  return refreshSubmission(host, slug, assignmentId);
}
