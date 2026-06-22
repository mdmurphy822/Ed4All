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
from dart_semantic.structure_graph import REGION_KINDS, Region

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
    "stating what you found and why). When the block is already correct, "
    "return verdict \"ok\" with corrected_kind/level/doc_role unchanged or "
    "null."
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

    system = "\n\n".join(
        [
            _SYSTEM_REVIEWER,
            _GLOBAL_WCAG_HEADER,
            _rule_block_for(region.kind),
            f"Conformant shape for this kind: {_exemplar_for(region.kind)}",
        ]
    )
    return f"SYSTEM: {system}\nUSER: {user_json}"


__all__ = [
    "NEIGHBOR_SNIPPET_CHARS",
    "REGION_KINDS",
    "build_reviewer_input_json",
    "build_reviewer_request",
]
