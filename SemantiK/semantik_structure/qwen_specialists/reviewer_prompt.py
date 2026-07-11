"""Stage-5d structure-REVIEWER prompt builder (text-preserving 70B).

This module builds the prompt for the 70B **structure reviewer** — a
distinct concern from the Stage-6 content GENERATOR. Per the design
(`plans/finegrain/semantik-70b-structure-reviewer-2026-06-22.md` §3, §5,
§6) the reviewer:

* **NEVER** rewrites / paraphrases / summarizes / adds / removes source
  TEXT. It touches STRUCTURE / ROLE only (kind / heading-level / doc-role
  / WCAG wrapper). The source words are preserved verbatim.
* Receives an already-classified block + neighbor context and returns a
  **parseable JSON verdict** (one object per block), NOT an HTML
  fragment. It INVERTS the Stage-6 generation contract.
* Is given a per-region-kind WCAG rule block (§5) that mirrors exactly
  what the Stage-7 gates check, so the model's structural verdict cannot
  fail the very gates it sits upstream of.

Why a SEPARATE prompt builder (not `endpoint_runtime.split_specialist_prompt`)
--------------------------------------------------------------------------
`split_specialist_prompt` ALWAYS prepends ``_ENVELOPE_DIRECTIVE``
("Convert … into ONE accessible HTML5 fragment") — a content-AUTHORING
framing that would make the 70B paraphrase, a direct verbatim-mandate
violation (Risk R1). The reviewer needs a self-contained ``SYSTEM:`` line
that NEVER carries that envelope. ``reviewer.py`` therefore drives the
endpoint via a reviewer-aware split (``split_reviewer_prompt`` below)
that does NOT touch ``_ENVELOPE_DIRECTIVE`` and reuses
``OpenAICompatibleRuntime.generate_batch`` / ``_one_completion``
unchanged.

The JSON-only mandate is a CONTRACT, not the safety net — the tolerant
extractor in ``reviewer.py`` is the safety net (§3 M1).
"""

from __future__ import annotations

import json
from typing import Any

# Keep in lock-step with structure_graph.REGION_KINDS (11 members).
# ``_get_signal`` / ``_top1`` (CouncilState lookup) and ``_running_header_norm``
# / ``_detect_running_headers`` (deterministic furniture recurrence) are reused
# verbatim from ``structure_graph`` — that module is a dependency of this one
# (no cycle), so the import is safe at module top. ``_joined_source_text`` /
# ``resolve_block_review_edge_tokens`` live in ``reviewer`` (which imports THIS
# module), so they are imported lazily inside ``build_edge_input`` to break the
# import cycle (doc §7 pins ``_joined_source_text`` as the SOLE verbatim-source
# accessor — it is reused, never reimplemented).
from semantik_structure.structure_graph import (
    REGION_KINDS,
    Region,
    _detect_running_headers,
    _get_signal,
    _running_header_norm,
    _top1,
)
from semantik_structure.types import FeatureBlock

# ---------------------------------------------------------------------------
# Neighbor-snippet truncation (§3 — ~120 chars/neighbor; m4: the window
# SIZE is ±2; the 120-char truncation is the MECHANISM, confirmed adequate
# in Phase 4 on the real answer-key fixture).
# ---------------------------------------------------------------------------

NEIGHBOR_SNIPPET_CHARS = 120


# ---------------------------------------------------------------------------
# System directive — structure-only, no-rewrite, JSON-only.
#
# This string MUST NOT contain the generation envelope ("convert … into a
# fragment"); a regression test asserts its absence (§6, R1).
# ---------------------------------------------------------------------------

_SYSTEM_REVIEWER = (
    "You are a DART document-STRUCTURE REVIEWER. You are NOT a content "
    "author or generator. Your single job is to review the logical and "
    "semantic conformance of one already-classified document block and, "
    "where it is mis-structured, correct its ROLE / KIND / heading-LEVEL "
    "only.\n"
    "\n"
    "HARD CONSTRAINT — VERBATIM TEXT. You MUST NEVER rewrite, paraphrase, "
    "summarize, translate, add, or remove any of the source TEXT. You "
    "touch STRUCTURE and ROLE only; the source words are preserved "
    "verbatim. You do NOT emit HTML. You do NOT produce a fragment.\n"
    "\n"
    "POSTURE — CONSERVATIVE + ADDITIVE. Your DEFAULT is to KEEP the block "
    "as-is (verdict \"ok\"). Re-tag a block ONLY when the TEXT ITSELF makes "
    "clear it is not a real heading. Genuine section / chapter / subsection "
    "headings are KEPT as headings — front-matter and table-of-contents "
    "handling is done by a separate DETERMINISTIC pass, NOT by you. When in "
    "doubt, return \"ok\".\n"
    "\n"
    "What you correct (only the clearly-not-a-heading cases):\n"
    "  1. A block whose TEXT is clearly NOT a real heading — an answer-key "
    "fragment (e.g. \"3.2: 14, 17, 20\"), a bare page number, an operator- "
    "or-numeral-led noise string, a running header / page furniture, or other "
    "obvious non-heading content that was mis-promoted to a heading. Re-role "
    "it: kind 'heading' -> 'paragraph' when the words are real body content, "
    "or kind 'heading' -> 'metadata_drop' when the text is genuine "
    "non-content furniture / answer-key. PREFER 'paragraph' for real words; "
    "reserve 'metadata_drop' for true furniture. Judge the TEXT, not the "
    "company it keeps: do NOT demote a heading merely because it sits in a "
    "run of similar headings; only re-tag when the text itself is clearly "
    "not heading content.\n"
    "  2. A real heading at the wrong depth — re-level it by setting "
    "corrected_level (an integer 1-6). Express the level ONLY through "
    "corrected_level; the assembler owns final absolute h1..h6 "
    "normalization, so a level you write is an INPUT to normalization, not "
    "a competing authority.\n"
    "  3. A real heading the classifier missed — promote kind 'paragraph' "
    "-> 'heading' and set corrected_level. The block's existing verbatim "
    "text is kept as the heading text; you NEVER invent heading words.\n"
    "\n"
    "Use the prev/next neighbor snippets to judge document flow, but they are "
    "context, not a trigger: a heading whose own TEXT is a plausible section "
    "title stays a heading even when its neighbors look similar.\n"
    "\n"
    "STRUCTURAL CLUSTER SIGNALS — INFORMATIONAL CONTEXT ONLY. Each block "
    "carries four deterministic structural measurements computed over the "
    "whole document: same_level_run_len (how many consecutive same-level "
    "headings this heading sits in, with NO body content between them), "
    "run_position (this heading's place in that run), content_blocks_following "
    "(how many body blocks — paragraph/list/table — immediately follow this "
    "heading before the next heading), and trailing_pagenum (whether the text "
    "ends in a bare page number). These are measurements to inform your read "
    "of the document — they are NOT a rule. In particular, do NOT demote a "
    "heading just because it is part of a long same-level run or has "
    "content_blocks_following == 0; many real chapter / section openers are "
    "back-to-back in a well-structured document. Re-tag ONLY on the basis of "
    "the block's OWN text being clearly non-heading content (correction 1 "
    "above). A trailing_pagenum == true is a hint that the TEXT may be a "
    "table-of-contents / index line, but confirm against the text itself "
    "before re-tagging.\n"
    "\n"
    "OUTPUT — JSON ONLY. Return EXACTLY ONE JSON object and nothing else. "
    "NO Markdown, NO code fences, NO commentary, NO surrounding prose. The "
    "object's keys are: block_id (int, echo the input), verdict (one of "
    "\"ok\", \"corrected\", \"drop_injected_header\"), corrected_kind (a "
    "RegionKind string or null), corrected_level (int 1-6 or null), "
    "corrected_doc_role (string or null), review_note (a short string "
    "stating what you found and why), and OPTIONALLY ambiguous (a boolean — "
    "set true ONLY when the head/tail edge you were shown is genuinely "
    "insufficient to decide the kind, so a fuller-text re-read is warranted; "
    "omit it or set false when you are confident). When the block is already "
    "correct, return verdict \"ok\" with corrected_kind/level/doc_role "
    "unchanged or null."
)


# ---------------------------------------------------------------------------
# Global WCAG header — stated ONCE (not per kind). §5 compactness rule 2.
# ---------------------------------------------------------------------------

_GLOBAL_WCAG_HEADER = (
    "WCAG / HTML contract (global, applies to every kind):\n"
    "  - Preserve every source word verbatim — never rewrite/paraphrase/"
    "add/remove (the verbatim invariant above).\n"
    "  - Emit NO color/font/spacing CSS — contrast/resize/focus/spacing "
    "(SC 1.4.3/1.4.4/2.4.7) are owned by the DART page CSS template, not "
    "by block content.\n"
    "  - Never an empty image source (<img src=''>) and never math-as-"
    "image.\n"
    "  - The corrected structure must pass axe-core WCAG 2.2 AA with no "
    "serious/critical violations."
)


# ---------------------------------------------------------------------------
# Conformant exemplars — anchored on the deterministic emitter shapes the
# Stage-7 gates accept (mirrors `ontology_map.py` element choices: text-only
# <figure><figcaption>, <th scope=...>, <math alttext=...>). §5 rule 3.
# ---------------------------------------------------------------------------

_EXEMPLARS: dict[str, str] = {
    "heading": "<h2>Section heading text</h2>",
    "table": (
        "<table><caption>...</caption><thead><tr>"
        "<th scope=\"col\">Col</th></tr></thead>"
        "<tbody><tr><td>cell</td></tr></tbody></table>"
    ),
    "math": (
        "<math xmlns=\"http://www.w3.org/1998/Math/MathML\" "
        "alttext=\"x equals 2\"><mi>x</mi><mo>=</mo><mn>2</mn></math>"
    ),
    "figure": "<figure><figcaption>Caption text</figcaption></figure>",
    "list": "<ul><li>item<ul><li>nested</li></ul></li></ul>",
    "code_block": "<pre><code>verbatim code</code></pre>",
    "blockquote": "<blockquote><p>quoted text</p></blockquote>",
    "paragraph": "<p>body text</p>",
    "definition_list": "<dl><dt>term</dt><dd>definition</dd></dl>",
}


# ---------------------------------------------------------------------------
# Per-region-kind WCAG rule blocks (§5 — the reviewer restates ONLY the
# HARD-gated rules; soft/template-scope criteria are out of scope). Per-kind
# DISPATCH: each call carries only its kind's contract (a heading never sees
# the table H43 text), mirroring `check_region`'s kind-conditional dispatch.
# ---------------------------------------------------------------------------

_RULE_BLOCKS: dict[str, str] = {
    "heading": (
        "Rule (heading, SC 2.4.6 / 1.3.1): the text MUST be REAL heading "
        "text, not injected content (no TOC/answer-key/noise lines). "
        "Headings render as <h1>..<h6>, levels strictly 1..6, no level "
        "skips on descent, the first heading is <h1>. Hierarchy is gated "
        "at the DOCUMENT level, so a wrong level can sink the whole "
        "document — use the running heading-depth context. Express level "
        "ONLY via corrected_level."
    ),
    "table": (
        "Rule (table, SC 1.3.1): a simple table — every <th> carries "
        "scope in {col,row,colgroup,rowgroup} or a non-empty id. A complex "
        "table (>=2 header rows / dual-axis / spanned <th>) needs full H43 "
        "id/headers pairing. A 2x2+ table with NO <th> hard-fails. Every "
        "source cell must survive (coverage 0.95). Emit clean <th>/<td> "
        "structure and let the deterministic H43 enrichment generate "
        "id/headers — the grid is the authority, do NOT author them."
    ),
    "math": (
        "Rule (math, SC 1.1.1 / 1.4.5): ship MathML, NEVER an image. "
        "Well-formed XML; root <math> (MathML namespace or bare "
        "no-namespace); non-empty alttext; presentation elements only "
        "(50-element allowlist); every source glyph/token preserved "
        "(coverage 0.90)."
    ),
    "figure": (
        "Rule (figure, SC 1.1.1): <figure> + <figcaption>. If an <img> is "
        "present it MUST carry alt; NEVER <img src=''>; when there is no "
        "real source, emit a text-only <figure><figcaption>. Do NOT "
        "fabricate descriptive alt beyond the source text."
    ),
    "list": (
        "Rule (list): a native semantic list element; a nested <ul> lives "
        "INSIDE an <li>, not as a sibling of other <li>s. No structural "
        "gate beyond HTML5 well-formedness + text-preserve + axe."
    ),
    "definition_list": (
        "Rule (definition list): a native <dl> with <dt>/<dd> pairs. No "
        "structural gate beyond HTML5 well-formedness + text-preserve + "
        "axe."
    ),
    "code_block": (
        "Rule (code): <pre><code> with escaped, verbatim contents. No "
        "structural gate beyond HTML5 well-formedness + text-preserve + "
        "axe."
    ),
    "blockquote": (
        "Rule (blockquote): <blockquote><p>...</p></blockquote>. No "
        "structural gate beyond HTML5 well-formedness + text-preserve + "
        "axe."
    ),
    "paragraph": (
        "Rule (paragraph): a native <p>. No structural gate beyond HTML5 "
        "well-formedness + text-preserve + axe."
    ),
    "form": (
        "Rule (form): native form controls with associated <label>s. No "
        "dedicated structural gate beyond HTML5 well-formedness + "
        "text-preserve + axe."
    ),
    "metadata_drop": (
        "Rule (metadata_drop): page furniture / answer-key / running "
        "header — non-content. It emits empty HTML by design and is "
        "filtered from the document body. Re-role a block here ONLY when "
        "the text is genuine non-content furniture."
    ),
}


def _rule_block_for(kind: str) -> str:
    """Return the per-kind rule block (falls back to the paragraph rule for
    any unexpected kind so the prompt is never empty)."""
    return _RULE_BLOCKS.get(kind, _RULE_BLOCKS["paragraph"])


def _exemplar_for(kind: str) -> str:
    """Return the conformant exemplar for ``kind`` (paragraph fallback)."""
    return _EXEMPLARS.get(kind, _EXEMPLARS["paragraph"])


# ---------------------------------------------------------------------------
# Input-JSON assembly (§3 input contract).
# ---------------------------------------------------------------------------


def _snippet(text: str | None) -> str:
    """Hard-truncate a neighbor snippet to ~120 chars (collapse newlines)."""
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat[:NEIGHBOR_SNIPPET_CHARS]


def _neighbor_obj(neighbor: Region | None) -> dict[str, Any] | None:
    """Build the ±1 neighbor snippet object from a Region (or None)."""
    if neighbor is None:
        return None
    payload = neighbor.payload or {}
    return {
        "kind": neighbor.kind,
        "level": payload.get("level_hint"),
        "text_snippet": _snippet(payload.get("text") or ""),
    }


def _cluster_signal_fields(cluster_signals: Any | None) -> dict[str, Any]:
    """Project a :class:`ClusterSignals` (duck-typed, or None) into the four
    input-JSON fields. Accepts the dataclass from ``reviewer.py`` without an
    import (avoids a circular import); falls back to inert defaults when None.
    """
    if cluster_signals is None:
        return {
            "same_level_run_len": 0,
            "run_position": 0,
            "content_blocks_following": 0,
            "trailing_pagenum": False,
        }
    return {
        "same_level_run_len": getattr(cluster_signals, "same_level_run_len", 0),
        "run_position": getattr(cluster_signals, "run_position", 0),
        "content_blocks_following": getattr(
            cluster_signals, "content_blocks_following", 0
        ),
        "trailing_pagenum": bool(getattr(cluster_signals, "trailing_pagenum", False)),
    }


def build_reviewer_input_json(
    region: Region,
    *,
    block_id: int,
    text: str,
    prev_block: Region | None = None,
    next_block: Region | None = None,
    cluster_signals: Any | None = None,
) -> dict[str, Any]:
    """Assemble the §3 input object for one block.

    ``text`` is the caller-resolved verbatim source text (the m2 rule lives
    in ``reviewer.py::_resolve_region_text`` so this builder stays a pure
    formatter). ``prev_block`` / ``next_block`` are the ±1 neighbor Regions
    (the ±2 window is threaded by the caller passing the right neighbors;
    this object carries the ±1 snippets the design names).

    ``cluster_signals`` carries the four deterministic CLUSTER-LEVEL signals
    (``same_level_run_len`` / ``run_position`` / ``content_blocks_following``
    / ``trailing_pagenum``) computed in ``reviewer.py::compute_cluster_signals``
    so the model can distinguish a phantom-TOC/index cluster from real
    content. When None (a caller that did not compute them), the four fields
    are emitted with inert defaults.
    """
    payload = region.payload or {}
    heading_conf = payload.get("confidence")
    obj: dict[str, Any] = {
        "block_id": block_id,
        "current_kind": region.kind,
        "current_level": payload.get("level_hint"),
        "current_doc_role": payload.get("doc_role"),
        "text": text,
        "heading_confidence": heading_conf,
        "prev_block": _neighbor_obj(prev_block),
        "next_block": _neighbor_obj(next_block),
    }
    obj.update(_cluster_signal_fields(cluster_signals))
    return obj


def build_reviewer_request(
    region: Region,
    neighbors: tuple[Region | None, Region | None],
    index: int,
    *,
    text: str | None = None,
    cluster_signals: Any | None = None,
    feedback_by_idx: dict[int, str] | None = None,
) -> str:
    """Emit a ``SYSTEM:\\nUSER:`` reviewer prompt for one block.

    The SYSTEM turn carries the structure-only / no-rewrite / JSON-only
    mandate, the global WCAG header, the per-kind rule block, and the
    per-kind exemplar. The USER turn is the §3 input JSON (one line).

    ``index`` is the region's position in the structure_regions list and is
    used as ``block_id`` (the stable in-batch id). ``text`` is the
    caller-resolved verbatim source text (m2); when omitted it falls back to
    the region's payload ``text`` (the caller in ``reviewer.py`` always
    passes the m2-resolved value).
    """
    prev_block, next_block = neighbors
    resolved_text = text if text is not None else ((region.payload or {}).get("text") or "")
    input_obj = build_reviewer_input_json(
        region,
        block_id=index,
        text=resolved_text,
        prev_block=prev_block,
        next_block=next_block,
        cluster_signals=cluster_signals,
    )
    user_json = json.dumps(input_obj, ensure_ascii=False, separators=(",", ":"))

    system_parts = [
        _SYSTEM_REVIEWER,
        _GLOBAL_WCAG_HEADER,
        _rule_block_for(region.kind),
        f"Conformant shape for this kind: {_exemplar_for(region.kind)}",
    ]
    # CONTENT branch only: a non-heading (content) block additionally gets the
    # content-type re-typing directive so the model can correct a mis-typed
    # KIND (e.g. a 'TRY IT' exercise the council labeled code_block -> paragraph).
    # The HEADING single-block prompt is left byte-identical (the legacy
    # byte-stable path its tests assert against).
    if str(region.kind) != "heading":
        system_parts.append(_CONTENT_RETYPE_DIRECTIVE)
        # Phase 5 (flag-gated): offer the gold-component menu + the optional
        # semantic_class key. Appended as an EXTRA part ONLY when the flag is
        # on, so the flag-off content prompt is byte-identical (and the HEADING
        # path never reaches here, so its legacy byte-stable prompt is
        # untouched either way).
        if _semantic_class_enabled():
            system_parts.append(_semantic_class_directive())
    # Phase 5 (targeted re-drive): inject the Pass-2 verifier's per-block
    # feedback when THIS index was flagged. Gated on a non-empty dict carrying an
    # entry for this index -> round-1 / no-feedback prompts (single-object
    # contract, BOTH heading and content) are byte-identical.
    if feedback_by_idx:
        hint = feedback_by_idx.get(index)
        if hint:
            system_parts.append(_render_verifier_feedback(hint))
    system = "\n\n".join(system_parts)
    return f"SYSTEM: {system}\nUSER: {user_json}"


# ---------------------------------------------------------------------------
# Windowed block-review prompt (Phase 4 — one prompt per window of M members).
#
# Packs many Phase-1 edge records into ONE prompt and asks for an idx-keyed
# op-LIST out (a JSON array of ops, one per block the model wants to change),
# mirroring the endpoint runtime's BATCHED delimited-envelope precedent
# (``endpoint_runtime._BATCH_ENVELOPE_DIRECTIVE`` / ``generate_multi``). Reuses
# the conservative structure-only / verbatim-text mandate of _SYSTEM_REVIEWER;
# only the OUTPUT CARDINALITY contract differs (an array keyed by idx instead
# of one object keyed by block_id). A window of a SINGLE member does NOT use
# this builder — the driver degenerates to the byte-stable single-block
# ``build_reviewer_request`` for size-1 windows (Phase-3 byte-stability).
# ---------------------------------------------------------------------------

_WINDOW_OPLIST_DIRECTIVE = (
    "You are reviewing MULTIPLE document blocks in ONE response. The USER "
    "message is a JSON ARRAY of compact EDGE RECORDS — each carries its idx, "
    "council_kind, role, page, n_tokens, and EITHER the full verbatim text "
    "(short blocks) OR head/tail token edges (long blocks). Apply the SAME "
    "conservative, structure-only review to EACH block independently.\n"
    "\n"
    "OUTPUT — JSON ARRAY ONLY. Return EXACTLY ONE JSON array and nothing "
    "else (NO Markdown, NO code fences, NO commentary, NO surrounding prose). "
    "Emit ONE op object per block you want to CHANGE; OMIT any block you "
    "would leave \"ok\". Each op object's keys are: idx (int — ECHO the "
    "block's idx; this is how your op is matched back to the block, and an "
    "idx NOT present in this window is DROPPED), verdict (one of \"ok\", "
    "\"corrected\", \"drop_injected_header\"), corrected_kind (a RegionKind "
    "string or null), corrected_level (int 1-6 or null), corrected_doc_role "
    "(string or null), review_note (a short string), and OPTIONALLY ambiguous "
    "(a boolean — true ONLY when the head/tail edge you were shown is "
    "genuinely insufficient to decide the kind, so a fuller-text re-read is "
    "warranted). You touch kind/role/level ONLY — NEVER rewrite the source "
    "text."
)


# ---------------------------------------------------------------------------
# Content-type RE-TYPING directive (the Phase-6 root-cause fix). The heading-
# oriented _SYSTEM_REVIEWER only ever asks "is this a heading?" — it makes the
# model set corrected_doc_role but NEVER corrected_kind, so a mis-typed CONTENT
# block (a 'TRY IT' exercise the council labeled code_block) is never re-typed.
# This directive asks the model, for a CONTENT block, to judge whether its
# council_kind is the correct CONTENT TYPE and return corrected_kind (a real
# REGION_KINDS value) when it is wrong. It is appended ONLY to the content /
# windowed prompts; the heading single-block prompt is left byte-identical so
# the legacy heading path (and its byte-stability tests) is unaffected.
# ---------------------------------------------------------------------------

_CONTENT_RETYPE_DIRECTIVE = (
    "CONTENT-TYPE RE-TYPING (content blocks). Each block is labeled with a "
    "council_kind (its current content type). If that council_kind is the WRONG "
    "type for the block's ACTUAL content, return corrected_kind set to the "
    "correct kind (and verdict \"corrected\"). This is a CONTENT-TYPE judgment, "
    "NOT only a heading / doc-role judgment: populate corrected_kind whenever "
    "the TYPE is wrong — you still NEVER rewrite, add, or remove the source "
    "text, you only change the structural KIND label.\n"
    "Use ONLY these kinds for corrected_kind (there is NO 'exercise' kind): "
    "paragraph, heading, list, definition_list, table, math, code_block, "
    "blockquote, figure, form, metadata_drop.\n"
    "Common council errors to correct:\n"
    "  - A practice exercise (e.g. a \"TRY IT : : 1.27 ...\" item) or ordinary "
    "running prose mislabeled as code_block -> return corrected_kind "
    "\"paragraph\" (an exercise / prose is body text, not code; there is no "
    "exercise kind, so it re-types to paragraph).\n"
    "  - A definition / term list mislabeled as table -> return the correct "
    "list kind: corrected_kind \"definition_list\" for term/definition pairs, "
    "or \"list\" / \"paragraph\" as the content warrants (a table is a real "
    "row/column grid, not a definition line).\n"
    "Example: council_kind \"code_block\", text \"TRY IT : : 1.27 Simplify "
    "3 + 7\" -> corrected_kind \"paragraph\", verdict \"corrected\" (a worked "
    "exercise, not code).\n"
    "Example: council_kind \"table\", text \"Commutative Property: a + b = "
    "b + a\" -> corrected_kind \"paragraph\", verdict \"corrected\" (a "
    "definition line, not a tabular grid).\n"
    "When the council_kind is ALREADY the correct content type, leave it "
    "(verdict \"ok\", corrected_kind null)."
)


# ITEM6 — council top-k distribution directive. Appended (windowed prompt ONLY)
# when any record carries ``council_top_k`` so the model reads the ranked
# alternatives as re-typing evidence.
_COUNCIL_TOPK_DIRECTIVE = (
    "Some records carry council_top_k: the classifier's ranked alternative "
    "content types with probabilities. A close runner-up means the printed "
    "council_kind is uncertain. When you re-type, prefer a listed alternative "
    "over an unlisted kind unless the text plainly demands otherwise."
)


# ---------------------------------------------------------------------------
# Gold accessible semantic-CLASS menu + assign directive (Phase 5, flag-gated
# on ``SEMANTIK_SEMANTIC_CLASS`` via ``resolve_semantic_class_mode``).
#
# This adds a SECOND, orthogonal axis to the content / windowed review: in
# addition to ``corrected_kind`` (the 11-member REGION_KINDS HTML structure)
# the model MAY return an optional ``semantic_class`` (a gold accessible-
# component token from the in-tree catalog). The menu is rendered from the
# SemantiK-local ``semantic_catalog`` accessors (NEVER ``lib.*`` — the reviewer
# runs out-of-process in SemantiK's bridge venv), mirroring how
# ``lib/generation/block_planner._build_prompt`` renders its block catalog as
# ``- {token}: {use_when}`` lines.
#
# HARD INVARIANT — flag-off byte-identical: the menu + directive are appended
# to the content / windowed system turn ONLY when ``resolve_semantic_class_mode``
# is True. Flag-off, neither builder appends anything, so every legacy prompt
# (incl. the heading single-block path, which never reaches the content branch)
# is byte-identical. The contract-key mention is carried WHOLLY inside this
# flag-gated block (it documents the optional key against BOTH the single-object
# contract of ``_SYSTEM_REVIEWER`` and the op-list contract of
# ``_WINDOW_OPLIST_DIRECTIVE``), so the module-level constants are NOT edited and
# stay byte-stable for the flag-off / heading paths.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Verifier-feedback directive (Phase 5 — targeted Pass-1 re-drive).
#
# When the Pass-2 verifier FLAGS a region, the Phase-6 orchestrator bounces that
# index back through the Pass-1 reviewer with the verifier's per-block
# ``fix_hint`` injected into the prompt. This directive is appended to the
# SINGLE-block ``build_reviewer_request`` system turn ONLY when feedback for that
# index exists; the windowed path instead travels the feedback in the USER JSON
# (the ``review_feedback`` record key). Gated on a non-empty entry -> round-1 /
# no-feedback prompts are byte-identical (mirrors the ``_semantic_class_directive``
# / ``_CONTENT_RETYPE_DIRECTIVE`` flag-gated append idiom).
# ---------------------------------------------------------------------------

_VERIFIER_FEEDBACK_DIRECTIVE = (
    "VERIFIER FEEDBACK (targeted re-review). A Pass-2 verifier judged the "
    "ASSEMBLED document and flagged THIS block for re-typing. Treat the hint "
    "below as the authoritative reason this block needs correction, and emit "
    "the corrected kind / semantic_class accordingly — you still NEVER rewrite, "
    "add, or remove the source text. Verifier hint: "
)


def _render_verifier_feedback(hint: str) -> str:
    """Render the verifier-feedback system part (constant header + the hint)."""
    return _VERIFIER_FEEDBACK_DIRECTIVE + " ".join(str(hint).split())


def _semantic_class_enabled() -> bool:
    """Return ``resolve_semantic_class_mode()`` (lazy import to break the
    ``reviewer`` <-> ``reviewer_prompt`` cycle — ``reviewer`` imports THIS
    module, mirroring ``build_edge_input``'s lazy ``from .reviewer import``)."""
    from .reviewer import resolve_semantic_class_mode

    return resolve_semantic_class_mode()


def _semantic_component_menu() -> str:
    """Render the gold-component menu — one terse ``- {component}: {use_when}``
    line per catalog entry (mirrors ``block_planner._build_prompt``'s catalog
    menu idiom). SemantiK-local accessors only; never imports ``lib.*``."""
    from semantik_structure.assembler.semantic_catalog import load_semantic_catalog

    lines = [
        "AVAILABLE SEMANTIC COMPONENTS (choose at most ONE token as semantic_class):"
    ]
    for entry in load_semantic_catalog():
        component = str(entry.get("component") or "").strip()
        if not component:
            continue
        use_when = " ".join(str(entry.get("use_when") or "").split())
        lines.append(f"  - {component}: {use_when}")
    return "\n".join(lines)


def _semantic_class_directive() -> str:
    """The flag-gated semantic-class assign instruction + EXAMPLE-is-a-component
    rule + the rendered component menu, appended to the content / windowed
    system turn when ``resolve_semantic_class_mode`` is True. Rendered at call
    time (the catalog read is cheap + lru-cached)."""
    return (
        "SEMANTIC CLASS (additive — ORTHOGONAL to corrected_kind). For EVERY "
        "content block you review or re-type you MUST assign a gold accessible-"
        "component semantic_class: the single CLOSEST component token from the "
        "AVAILABLE SEMANTIC COMPONENTS menu below. Set it to null ONLY when "
        "truly NO component fits. corrected_kind is the block's HTML STRUCTURE "
        "(paragraph/list/table/...); semantic_class is its PEDAGOGICAL / "
        "STRUCTURAL role (worked_example, definition_region, callout_note, "
        "exercise, ...) — the two are independent. ALWAYS choose the closest "
        "component, ESPECIALLY for the dispatched code_block / table blocks (the "
        "council's known-weak kinds — these are usually worked_example / "
        "exercise / definition_region, NOT real code or data tables). The "
        "semantic_class JSON key stays optional for backward compatibility (an "
        "absent key is treated as null), but you SHOULD populate it on every "
        "block — add the semantic_class string (from the menu, or null) to the "
        "single object's keys (single-block review) or to each op object's keys "
        "(windowed review). You still NEVER rewrite, add, or remove the source "
        "text.\n"
        "RULE — AN EXAMPLE IS A COMPONENT, NOT A HEADING. A labeled EXAMPLE / "
        "worked instance is BODY content carrying a semantic_class; it is NEVER "
        "a heading. EXAMPLE 1.3: a block whose text begins \"EXAMPLE 1.3 "
        "Simplify 3 + 7\" -> corrected_kind \"paragraph\", semantic_class "
        "\"worked_example\", verdict \"corrected\" — NEVER corrected_kind "
        "\"heading\" (an EXAMPLE label is a component, not a section heading).\n"
        + _semantic_component_menu()
    )


def build_windowed_reviewer_request(
    records: list[dict[str, Any]],
    *,
    cluster_signals_by_idx: dict[int, Any] | None = None,
    feedback_by_idx: dict[int, str] | None = None,
) -> str:
    """Emit ONE windowed block-review prompt packing many edge records.

    Each member is the Phase-1 ``build_edge_input`` record (idx-keyed). The
    SYSTEM turn carries the conservative structure-only mandate + the global
    WCAG header + the idx-keyed op-LIST output contract; the USER turn is the
    JSON array of edge records (one per member), optionally enriched with each
    member's four cluster signals. Pure formatter — no LLM, no mutation, no
    env side effect. Used only for windows of >= 2 members (the driver routes
    a single-member window to the byte-stable single-block prompt).
    """
    enriched: list[dict[str, Any]] = []
    for rec in records:
        item = dict(rec)
        if cluster_signals_by_idx is not None:
            item.update(_cluster_signal_fields(cluster_signals_by_idx.get(rec.get("idx"))))
        # Phase 5 (targeted re-drive): stamp the Pass-2 verifier feedback onto
        # the record under the DISTINCT key ``review_feedback`` (NOT a reserved
        # edge-record key — idx/council_kind/role/confidence/page/n_tokens/
        # dup_count/text/head/tail/pages) so it travels in the USER JSON next to
        # that block's idx. Gated on a non-empty entry -> round-1 prompts are
        # byte-identical.
        if feedback_by_idx:
            fb = feedback_by_idx.get(rec.get("idx"))
            if fb:
                item["review_feedback"] = " ".join(str(fb).split())
        enriched.append(item)
    user_json = json.dumps(enriched, ensure_ascii=False, separators=(",", ":"))
    system_parts = [
        _SYSTEM_REVIEWER,
        _GLOBAL_WCAG_HEADER,
        _WINDOW_OPLIST_DIRECTIVE,
        _CONTENT_RETYPE_DIRECTIVE,
    ]
    # Phase 5 (flag-gated): append the gold-component menu + optional
    # semantic_class key ONLY when the flag is on, so the flag-off windowed
    # prompt is byte-identical to the legacy assembly.
    if _semantic_class_enabled():
        system_parts.append(_semantic_class_directive())
    # ITEM6 (windowed ONLY): append the council-top-k directive when any record
    # carries the additive ``council_top_k`` evidence key. Gated on presence so
    # a window with no stamped records (mock state + unstamped regions) is
    # byte-identical to the legacy assembly; the single-block heading path never
    # reaches this builder, so it stays byte-identical too.
    if any("council_top_k" in r for r in enriched):
        system_parts.append(_COUNCIL_TOPK_DIRECTIVE)
    system = "\n\n".join(system_parts)
    return f"SYSTEM: {system}\nUSER: {user_json}"


# ---------------------------------------------------------------------------
# Pass-2 VERIFIER prompt + parser (Phase 3, flag-gated on SEMANTIK_SECOND_PASS).
#
# A SECOND, orthogonal reviewer ROLE: it judges the ASSEMBLED-document digest
# (heading outline + per-region {kind, semantic_class, aria_wrapper, head/tail})
# and emits per-flagged-region judgments keyed by ``region_index`` (the
# ``region_provenance`` capped-index VALUE), NEVER text. JSON-ARRAY-ONLY out,
# echo region_index. The shipped Pass-1 constants (``_SYSTEM_REVIEWER`` /
# ``_WINDOW_OPLIST_DIRECTIVE`` / ``_CONTENT_RETYPE_DIRECTIVE``) are NOT touched —
# this is a NEW directive, built ONLY when ``_second_pass_enabled()`` so the
# flag-off path is byte-identical.
# ---------------------------------------------------------------------------


def _second_pass_enabled() -> bool:
    """Return ``resolve_second_pass_mode()`` (lazy import to break the
    ``reviewer`` <-> ``reviewer_prompt`` cycle — mirrors
    :func:`_semantic_class_enabled`)."""
    from .reviewer import resolve_second_pass_mode

    return resolve_second_pass_mode()


_SECOND_PASS_VERIFY_DIRECTIVE = (
    "You are a Pass-2 VERIFIER judging an ALREADY-ASSEMBLED document. The USER "
    "message is a JSON object with a ``digest`` (a heading OUTLINE plus one "
    "terse entry per emitted region — idx-keyed ``region_index``, council "
    "``kind``, ``semantic_class``, ``aria_wrapper``, and head/tail token edges) "
    "and, on a re-ask, a ``spot_html`` map of a flagged section's REAL assembled "
    "HTML. You judge STRUCTURE, semantic_class, and reading ORDER ONLY — you "
    "NEVER rewrite, add, or remove source text.\n"
    "\n"
    "OUTPUT — JSON ARRAY ONLY. Return EXACTLY ONE JSON array and nothing else "
    "(NO Markdown, NO code fences, NO commentary, NO prose, NO rewritten text). "
    "An EMPTY array ``[]`` means the document PASSES (no defects). Otherwise emit "
    "ONE object per FLAGGED region. Each object's keys are: region_index (int — "
    "ECHO the digest entry's region_index; this is how your flag is matched back "
    "to the region, and a region_index NOT present in the digest is DROPPED), "
    "failure_mode (one of the vocabulary below), fix_hint (a SHORT directive for "
    "the re-typing reviewer — what the block should become), fixable (a boolean; "
    "copy the FIXABLE column below), and OPTIONALLY proposed_regroup_run (a JSON "
    "array of region_index ints — populate ONLY for section_no_body / "
    "example_misordered_from_body).\n"
    "\n"
    "FAILURE-MODE VOCABULARY (failure_mode -> fixable):\n"
    "  - example_as_heading (fixable=true): a labeled EXAMPLE / TRY IT appears "
    "as a heading_outline node; it is BODY content, re-type to a worked_example "
    "component.\n"
    "  - mistyped_component (fixable=true): the council kind disagrees with the "
    "assembled shape (a code_block that is really an exercise; a mis-routed "
    "table); re-type the kind.\n"
    "  - wrong_semantic_class (fixable=true): a worked-example/definition-shaped "
    "region has a wrong/absent aria_wrapper (bare-<p> fall-through); re-stamp the "
    "semantic_class.\n"
    "  - under_firing (fixable=true): a known-weak code_block/table the Pass-1 "
    "confidence gate skipped but the assembled shape needs re-typing.\n"
    "  - no_context_answer_fragment_heading (fixable=true ONLY when the fix is a "
    "per-block re-type to metadata_drop/paragraph; set fixable=false when it "
    "needs a MERGE into a neighbor).\n"
    "  - section_no_body (fixable=false for the re-typing reviewer; supply "
    "proposed_regroup_run — a move+merge channel handles it, and the run may be "
    "non-contiguous): a heading node with zero following content blocks.\n"
    "  - example_misordered_from_body (fixable=false for the re-typing reviewer; "
    "supply proposed_regroup_run — a move+merge channel handles it, and the run "
    "may be non-contiguous): a worked-example region separated from its body in "
    "reading order.\n"
    "\n"
    "SPOT-HTML RE-ASK. When a head/tail edge is genuinely insufficient to decide "
    "and you must see a region's real assembled HTML before failing it, emit an "
    "object with failure_mode \"need_spot_html\" and that region_index (no "
    "fix_hint needed) INSTEAD of a flag; you will be re-asked once with that "
    "section's HTML in ``spot_html``."
)


def build_second_pass_verify_request(
    digest: dict[str, Any],
    *,
    spot_html_by_idx: dict[int, str] | None = None,
) -> str:
    """Emit the Pass-2 verifier prompt (digest-in / index-keyed-judgments-out).

    Builds NOTHING (returns ``""``) when ``_second_pass_enabled()`` is False —
    the flag-off byte-stable contract. The SYSTEM turn carries the verifier
    directive + the global WCAG header; the USER turn is the JSON digest, plus
    an optional ``spot_html`` map (region_index -> real assembled section HTML)
    on the single bounded spot-HTML re-ask. Pure formatter — no LLM, no
    mutation, no env side effect beyond the gate read.
    """
    if not _second_pass_enabled():
        return ""
    system = "\n\n".join([_SECOND_PASS_VERIFY_DIRECTIVE, _GLOBAL_WCAG_HEADER])
    # ``allowed_region_indices`` is a WINDOWING hint, not part of the digest the
    # verifier reads — strip it from the serialized digest so a single-window
    # sub-digest (which carries the full regions + full outline) is byte-
    # identical to the non-windowed digest prompt.
    serial_digest = {k: v for k, v in digest.items() if k != "allowed_region_indices"}
    user_payload: dict[str, Any] = {"digest": serial_digest}
    # Per-window allow-list: the shared full-doc outline is repeated in every
    # window for context, so on a TRUE sub-window the model can SEE heading
    # region_indices outside the window. Pin the allow-list ONLY when the
    # outline references regions this window does not own — so the verifier
    # cannot flag an id outside its window. When the outline references only
    # in-window regions (the single-window / fits-one-window case), no allow-
    # list is emitted and the prompt stays byte-identical to today.
    entry_ids = {
        e.get("region_index")
        for e in serial_digest.get("regions", [])
        if isinstance(e, dict)
    }
    entry_ids.discard(None)
    allowed_raw = digest.get("allowed_region_indices")
    allowed = set(allowed_raw) if allowed_raw is not None else set(entry_ids)
    outline_ids = {
        n.get("region_index")
        for n in serial_digest.get("heading_outline", [])
        if isinstance(n, dict)
    }
    outline_ids.discard(None)
    if outline_ids - allowed:
        user_payload["allowed_region_indices"] = sorted(allowed)
    if spot_html_by_idx:
        user_payload["spot_html"] = {str(k): v for k, v in spot_html_by_idx.items()}
    user_json = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))
    return f"SYSTEM: {system}\nUSER: {user_json}"


def _extract_json_array(raw: str | None) -> list[Any] | None:
    """Tolerant fail-soft extraction of a JSON ARRAY (fences/commentary stripped).

    ``None`` on empty/whitespace/garbage (no raise), so a broken verifier never
    crashes the loop."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text[3:]
        nl = text.find("\n")
        if nl != -1 and text[:nl].strip().isalpha():
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except (ValueError, TypeError):
        pass
    i = text.find("[")
    j = text.rfind("]")
    if i != -1 and j != -1 and j > i:
        try:
            val = json.loads(text[i : j + 1])
            if isinstance(val, list):
                return val
        except (ValueError, TypeError):
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    """Tolerant int coercion (rejects bool, which is an ``int`` subclass)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def parse_verifier_response(raw: str | None):
    """Parse a verifier raw response into a ``VerifierVerdict`` (fail-soft None).

    Returns ``None`` on unparseable / empty / non-array input (mirroring the
    windowed parse's fail-soft posture). An empty array ``[]`` is a WELL-FORMED
    PASS (``passed=True``). A ``need_spot_html`` element adds its region_index to
    ``spot_html_requested`` (not a flag); every other element is a
    :class:`FlaggedBlock`. ``passed`` is True iff there are no flags AND no
    spot-HTML requests.
    """
    from .reviewer import FlaggedBlock, VerifierVerdict  # lazy: break cycle

    data = _extract_json_array(raw)
    if data is None:
        return None
    flagged: list[Any] = []
    spot: list[int] = []
    for el in data:
        if not isinstance(el, dict):
            continue
        ri = _coerce_int(el.get("region_index"))
        if ri is None:
            continue
        mode = str(el.get("failure_mode") or "").strip()
        if mode == "need_spot_html":
            spot.append(ri)
            continue
        if not mode:
            continue
        fix_hint = str(el.get("fix_hint") or "")
        fixable = bool(el.get("fixable", False))
        prr_raw = el.get("proposed_regroup_run") or ()
        prr_list: list[int] = []
        if isinstance(prr_raw, (list, tuple)):
            for x in prr_raw:
                xi = _coerce_int(x)
                if xi is not None:
                    prr_list.append(xi)
        flagged.append(
            FlaggedBlock(
                region_index=ri,
                failure_mode=mode,
                fix_hint=fix_hint,
                fixable=fixable,
                proposed_regroup_run=tuple(prr_list),
            )
        )
    passed = not flagged and not spot
    return VerifierVerdict(
        passed=passed,
        flagged=tuple(flagged),
        spot_html_requested=tuple(spot),
    )


# ---------------------------------------------------------------------------
# Edge-input builder (Phase 1 — design §3 record).
#
# Turns ONE region + its council signal into the head/tail-biased "edge"
# record the full-block reviewer windows over. Pure + GPU-free: no LLM, no
# region mutation, no env side effect beyond reading ``_EDGE_TOKENS`` (the
# Phase-0 resolver). NOT yet wired into any dispatch — dead-but-callable
# (Phase 4 wires it). The record is the design §3 shape:
#
#     {idx, council_kind, role, confidence, page, n_tokens, dup_count}
#
# plus EXACTLY ONE verbatim-text representation:
#   * short block (``n_tokens <= 2 * edge_tokens``): a ``text`` key carrying
#     the FULL verbatim joined source — ``head`` / ``tail`` are OMITTED (a
#     short block has no "middle" to elide, so the full text is the edge);
#   * long block: ``head`` (first N tokens) + ``tail`` (last N tokens) as
#     space-joined verbatim strings — ``text`` is OMITTED.
# The furniture-dedup helper additionally stamps an optional ``pages`` list.
# ---------------------------------------------------------------------------


def _fb_page(feature_blocks: list[FeatureBlock], idx: int | None) -> int | None:
    """The 1-indexed source page of FeatureBlock ``idx`` (``raw.page``)."""
    if idx is None:
        return None
    try:
        fb = feature_blocks[idx]
    except (IndexError, TypeError):
        return None
    return getattr(getattr(fb, "raw", None), "page", None)


def _fb_raw_text(feature_blocks: list[FeatureBlock], idx: int | None) -> str:
    """One FeatureBlock's stripped raw text (``raw.text``)."""
    if idx is None:
        return ""
    try:
        fb = feature_blocks[idx]
    except (IndexError, TypeError):
        return ""
    return (getattr(getattr(fb, "raw", None), "text", "") or "").strip()


def _edge_role_confidence(
    region: Region,
    council_state: Any | None,
    fb_idx: int | None,
) -> tuple[str | None, float | None]:
    """Resolve ``(role, confidence)`` for the edge record.

    A ``heading`` region carries its is_heading ``confidence`` on the payload
    (``structure_graph.py:1214``), so a heading reads it directly and its role
    is ``"heading"``. Content blocks do NOT carry ``payload['confidence']`` —
    the structural role is RE-DERIVED from CouncilState via Structure's
    ``structural_role`` head (``_get_signal`` -> ``_top1``, mirroring the
    structure_graph content-role sites). When the signal / state is missing,
    role falls back to the region's council kind and confidence is ``None``.
    """
    payload = region.payload or {}
    if region.kind == "heading":
        return "heading", payload.get("confidence")
    if council_state is None or fb_idx is None:
        return region.kind, None
    label, conf = _top1(_get_signal(council_state, "structure", "structural_role", fb_idx))
    return (label or region.kind), conf


def build_edge_input(
    region: Region,
    *,
    block_id: int,
    feature_blocks: list[FeatureBlock],
    council_state: Any | None = None,
    edge_tokens: int | None = None,
    dup_count: int = 1,
    pages: list[int] | None = None,
) -> dict[str, Any]:
    """Build the design §3 edge record for ONE region (pure read, no LLM).

    ``edge_tokens`` defaults to ``resolve_block_review_edge_tokens()`` (the
    Phase-0 ``SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS`` resolver) so the builder
    reads ``_EDGE_TOKENS`` by default; an explicit positive value overrides it.

    Verbatim source rides ``_joined_source_text`` (the SOLE FB-text accessor,
    doc §7) tokenized with the whitespace ``.split()`` convention — case is
    PRESERVED (unlike the lower-cased token-conservation multiset) because the
    head/tail edges are read by the model verbatim. ``n_tokens`` is the full
    token count; the head/tail (or full ``text``) is the only window kept.

    The builder NEVER mutates ``region`` or ``feature_blocks`` — it only views
    the edges (the Phase-1 pure-read contract).
    """
    # Lazy import — ``reviewer`` imports this module, so importing from it at
    # module top would cycle. ``_joined_source_text`` is reused verbatim (the
    # design's SOLE verbatim-source accessor); the edge-tokens resolver is the
    # Phase-0 ``_EDGE_TOKENS`` read.
    from .reviewer import _joined_source_text, resolve_block_review_edge_tokens

    if edge_tokens is None:
        edge_tokens = resolve_block_review_edge_tokens()
    n = int(edge_tokens)
    if n <= 0:
        n = resolve_block_review_edge_tokens()

    source = _joined_source_text(region, feature_blocks)
    tokens = source.split()
    n_tokens = len(tokens)

    fb_indices = region.feature_block_indices or ()
    first_idx = fb_indices[0] if fb_indices else None
    role, confidence = _edge_role_confidence(region, council_state, first_idx)

    record: dict[str, Any] = {
        "idx": block_id,
        "council_kind": region.kind,
        "role": role,
        "confidence": confidence,
        "page": _fb_page(feature_blocks, first_idx),
        "n_tokens": n_tokens,
        "dup_count": dup_count,
    }
    # ITEM6 — additive council role-distribution evidence (k=3, probs @2dp for
    # the prompt copy). Prefer the live council_state; fall back to the
    # persisted region.provenance['role_top_k'] on the post-run re-drive path
    # (state absent). Absent (byte-stable record) without either source.
    sig = (
        _get_signal(council_state, "structure", "structural_role", first_idx)
        if council_state is not None and first_idx is not None
        else None
    )
    labels = list(getattr(sig, "top_k_labels", None) or [])
    confs = list(getattr(sig, "top_k_confidences", None) or [])
    if labels and confs:
        record["council_top_k"] = [
            [str(lab), round(float(c), 2)] for lab, c in list(zip(labels, confs))[:3]
        ]
    else:
        prov_top_k = (getattr(region, "provenance", {}) or {}).get("role_top_k")
        if prov_top_k:
            record["council_top_k"] = [
                [str(lab), round(float(c), 2)] for lab, c in prov_top_k[:3]
            ]
    if n_tokens <= 2 * n:
        # Short block — the full verbatim text IS the edge (head/tail omitted).
        record["text"] = source
    else:
        record["head"] = " ".join(tokens[:n])
        record["tail"] = " ".join(tokens[-n:])
    if pages is not None:
        record["pages"] = list(pages)
    return record


def dedup_furniture_records(
    regions: list[Region],
    *,
    feature_blocks: list[FeatureBlock],
    council_state: Any | None = None,
    edge_tokens: int | None = None,
    block_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build edge records for ``regions``, collapsing repeated furniture.

    A running header / footer recurs verbatim (modulo its page number) on many
    pages; the design §3 furniture-dedup keeps ONE record + a page-list + a
    ``dup_count`` instead of one record per page. Recurrence detection is
    REUSED, not rebuilt: ``_detect_running_headers`` flags the furniture FBs
    (the same detector that ``metadata_drop``'s them at
    ``structure_graph.py:986``) and ``_running_header_norm`` supplies the
    cross-page grouping key. The first region of each furniture group keeps its
    edge record (``dup_count`` incremented + ``pages`` appended per repeat);
    every non-furniture region passes through unchanged (``dup_count`` 1).
    """
    header_fbs = _detect_running_headers(feature_blocks)
    records: list[dict[str, Any]] = []
    group_pos: dict[str, int] = {}  # norm key -> index into ``records``
    for pos, region in enumerate(regions):
        block_id = block_ids[pos] if block_ids is not None else pos
        fb_indices = region.feature_block_indices or ()
        first_idx = fb_indices[0] if fb_indices else None
        is_furniture = first_idx is not None and first_idx in header_fbs
        if is_furniture:
            norm = _running_header_norm(_fb_raw_text(feature_blocks, first_idx))
            page = _fb_page(feature_blocks, first_idx)
            if norm in group_pos:
                rec = records[group_pos[norm]]
                rec["dup_count"] += 1
                if page is not None and page not in rec["pages"]:
                    rec["pages"].append(page)
                continue
            rec = build_edge_input(
                region,
                block_id=block_id,
                feature_blocks=feature_blocks,
                council_state=council_state,
                edge_tokens=edge_tokens,
                dup_count=1,
                pages=[page] if page is not None else [],
            )
            group_pos[norm] = len(records)
            records.append(rec)
            continue
        records.append(
            build_edge_input(
                region,
                block_id=block_id,
                feature_blocks=feature_blocks,
                council_state=council_state,
                edge_tokens=edge_tokens,
            )
        )
    return records


__all__ = [
    "NEIGHBOR_SNIPPET_CHARS",
    "REGION_KINDS",
    "build_edge_input",
    "build_reviewer_input_json",
    "build_reviewer_request",
    "build_second_pass_verify_request",
    "dedup_furniture_records",
    "parse_verifier_response",
]
