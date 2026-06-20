/* Learner assignment + discussion module (W10 Phase 4, spec §5.2).
 *
 * Read-only first cut (no submission, no threading — §10). The SERVER renders
 * the sanitized fragment (title + objective + sanitized cartridge body), so this
 * module does fetch + verbatim fragment swap ONLY — it never templates the task
 * body from JSON. The server-side sanitize ran the assessment-path scrub
 * (strip_form_actions=True), so the swapped HTML carries no off-origin <form>
 * action and no active content.
 *
 * Contract (learner-open, no operator token):
 *   - GET /api/learn/assignment/{slug}/{assignment_id} -> {assignment_id, title, objective_id, html}
 *   - GET /api/learn/discussion/{slug}/{topic_id}      -> {topic_id, title, objective_id, html}
 */

async function fetchFragment(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  const body = await res.json().catch(function () {
    return null;
  });
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : "request failed";
    throw new Error(detail);
  }
  return body;
}

export async function loadAssignment(slug, assignmentId) {
  return fetchFragment(
    "/api/learn/assignment/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(assignmentId)
  );
}

export async function loadDiscussion(slug, topicId) {
  return fetchFragment(
    "/api/learn/discussion/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(topicId)
  );
}

// Swap a server-rendered task fragment into `host`, then focus its heading.
// `payload.html` is server-sanitized (assessment-path scrub) — inserted verbatim.
export function renderTask(host, payload) {
  host.innerHTML = payload && payload.html ? payload.html : "";
  const heading = host.querySelector("#task-h");
  if (heading) heading.focus();
}

export async function mountAssignment(host, slug, assignmentId) {
  const payload = await loadAssignment(slug, assignmentId);
  renderTask(host, payload);
  return payload;
}

export async function mountDiscussion(host, slug, topicId) {
  const payload = await loadDiscussion(slug, topicId);
  renderTask(host, payload);
  return payload;
}
