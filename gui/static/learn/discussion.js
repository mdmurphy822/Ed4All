/* Learner discussion-thread module (W10 §10 — discussion THREADING).
 *
 * Adds posting + one-level replies to the read-only discussion view (tasks.js).
 *
 * Contract (learner-open):
 *   - GET  /api/learn/discussion/{slug}/{topic_id}/posts -> [{post_id, learner_id, body_sanitized, parent_post_id, ts}]
 *   - POST /api/learn/discussion/{slug}/{topic_id}/posts  body {body, parent_post_id?}
 *
 * SECURITY: post bodies are sanitized SERVER-SIDE on store (strip_form_actions=True)
 * and the field is named body_sanitized. Even so, this module escapeHtml()'s
 * EVERY dynamic string before it touches innerHTML — the sanitized body is shown
 * as TEXT, not re-injected as live markup (defense in depth; the thread view
 * never renders another learner's HTML as markup).
 */

import { learnerFetch, getLearnerId } from "/learn/learner_id.js";

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

export async function loadPosts(slug, topicId) {
  const res = await fetch(
    "/api/learn/discussion/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(topicId) +
      "/posts",
    { headers: { Accept: "application/json" } }
  );
  if (!res.ok) throw new Error("posts load " + res.status);
  const body = await res.json().catch(function () {
    return [];
  });
  return Array.isArray(body) ? body : [];
}

export async function createPost(slug, topicId, body, parentPostId) {
  const payload = { body: body };
  if (parentPostId) payload.parent_post_id = parentPostId;
  const res = await learnerFetch(
    "/api/learn/discussion/" +
      encodeURIComponent(slug) +
      "/" +
      encodeURIComponent(topicId) +
      "/posts",
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!res.ok) throw new Error("post create " + res.status);
  return res.json();
}

// Shorten an opaque learner id for display (privacy: never show the full uuid of
// another learner; "you" for the caller's own posts).
function authorLabel(post, myId) {
  if (post.learner_id === myId) return "You";
  const id = String(post.learner_id || "");
  return "Learner " + escapeHtml(id.slice(0, 8));
}

// Render the thread: top-level posts, each followed by its one-level replies.
export function renderThread(host, posts) {
  const myId = getLearnerId();
  const tops = posts.filter(function (p) {
    return !p.parent_post_id;
  });
  const repliesOf = {};
  posts.forEach(function (p) {
    if (p.parent_post_id) {
      (repliesOf[p.parent_post_id] = repliesOf[p.parent_post_id] || []).push(p);
    }
  });

  function postLi(p, isReply) {
    const replies = (repliesOf[p.post_id] || [])
      .map(function (r) {
        return postLi(r, true);
      })
      .join("");
    return (
      '<li class="disc-post' +
      (isReply ? " disc-reply" : "") +
      '" data-post-id="' +
      escapeHtml(p.post_id) +
      '">' +
      '<p class="disc-meta"><strong>' +
      authorLabel(p, myId) +
      "</strong> " +
      '<span class="disc-ts">' +
      escapeHtml(String(p.ts || "")) +
      "</span></p>" +
      // body_sanitized is server-sanitized; shown as TEXT (escaped) here too.
      '<p class="disc-body">' +
      escapeHtml(stripTags(p.body_sanitized || "")) +
      "</p>" +
      (replies ? '<ul class="disc-replies">' + replies + "</ul>" : "") +
      (isReply ? "" : replyForm(p.post_id)) +
      "</li>"
    );
  }

  const items = tops
    .map(function (p) {
      return postLi(p, false);
    })
    .join("");
  host.innerHTML =
    '<section class="discussion" aria-labelledby="disc-h">' +
    '<h3 id="disc-h" tabindex="-1">Discussion</h3>' +
    '<ul class="disc-list">' +
    items +
    "</ul>" +
    rootForm() +
    '<div class="disc-status" role="status" aria-live="polite"></div>' +
    "</section>";
}

// A bs4-sanitized body may still carry safe inline markup (e.g. <strong>); we
// render posts as plain text in the thread, so flatten any remaining tags. (The
// server already removed all ACTIVE content; this is a display choice.)
function stripTags(html) {
  return String(html || "").replace(/<[^>]*>/g, "");
}

function rootForm() {
  return (
    '<div class="disc-new">' +
    '<label for="disc-new-body">Add a post</label>' +
    '<textarea id="disc-new-body" class="disc-input" rows="3"></textarea>' +
    '<button type="button" class="disc-submit" data-parent="">Post</button>' +
    "</div>"
  );
}

function replyForm(postId) {
  const pid = escapeHtml(postId);
  return (
    '<div class="disc-reply-new">' +
    '<textarea class="disc-input disc-reply-input" rows="2" ' +
    'aria-label="Reply"></textarea>' +
    '<button type="button" class="disc-submit disc-reply-submit" ' +
    'data-parent="' +
    pid +
    '">Reply</button>' +
    "</div>"
  );
}

// Wire post/reply buttons ONCE via event delegation on `host`. A WeakSet guards
// against stacking duplicate listeners across re-renders (renderThread replaces
// the innerHTML but the delegated listener on `host` survives, so we attach it
// exactly once per host).
const _wired = new WeakSet();
export function wireThread(host, slug, topicId) {
  if (_wired.has(host)) return;
  _wired.add(host);
  host.addEventListener("click", async function (ev) {
    const btn = ev.target;
    if (!btn || !btn.classList || !btn.classList.contains("disc-submit")) return;
    const parent = btn.getAttribute("data-parent") || "";
    const container = btn.closest(".disc-new, .disc-reply-new");
    const input = container ? container.querySelector(".disc-input") : null;
    const text = input ? input.value.trim() : "";
    const status = host.querySelector(".disc-status");
    if (!text) {
      if (status) status.textContent = "Write something before posting.";
      return;
    }
    btn.disabled = true;
    try {
      await createPost(slug, topicId, text, parent || null);
      await refreshThread(host, slug, topicId);
    } catch (_err) {
      if (status) status.textContent = "Could not post. Please try again.";
      btn.disabled = false;
    }
  });
}

// Re-fetch + re-render the thread WITHOUT re-wiring (the delegated listener
// persists on `host`).
export async function refreshThread(host, slug, topicId) {
  const posts = await loadPosts(slug, topicId);
  renderThread(host, posts);
  return posts;
}

export async function mountThread(host, slug, topicId) {
  wireThread(host, slug, topicId);
  return refreshThread(host, slug, topicId);
}
