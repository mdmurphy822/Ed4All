"""Phase 3 inter-tier gate adapters (Subtask 50, extended in Phase 3.5
Subtasks 6-9).

Bridges the outline-tier ``Block`` list emitted by
:class:`Courseforge.router.router.CourseforgeRouter` into the existing
``lib.validators`` surface so the workflow runner can reuse those
validators on outline-tier Blocks (which carry ``content`` as a dict
of ``{curies, key_claims, content_type, ...}``) without duplicating
the underlying check logic.

Each adapter implements the standard validator surface:

    class _Adapter:
        name: str
        version: str
        def validate(self, inputs: Dict[str, Any]) -> GateResult: ...

The router passes the Block list through ``inputs["blocks"]`` (mirroring
the ``inputs`` dict shape that legacy validators consume — see
``lib/validators/page_objectives.py:69``). Each adapter pulls the Block
list out, runs the per-block check, and aggregates the per-block
outcomes into a single ``GateResult``.

GateResult.action contract (Phase 3 Subtask 46 / Phase 4 §1):
- regenerate: outline-tier semantic miss the rewrite tier could fix on
  a re-roll. Used for content-side validators (``curie_anchoring``,
  ``content_type``).
- block: structural miss — re-rolling won't help because the outline
  references something that doesn't exist downstream (e.g. an
  ``objective_id`` not in the canonical objectives, a ``sourceId``
  not in the staging manifest). The router escalates instead of
  regenerating.

Phase 3.5 (Subtasks 6-9) extended these adapters with shape
discrimination so they also handle rewrite-tier blocks where
``block.content`` is an HTML string. Each adapter dispatches on
``isinstance(block.content, dict | str)`` via per-validator
``_extract_*`` helpers; the dict path preserves the legacy outline-tier
contract byte-stable while the str path scrapes the rewrite-tier HTML
for the same signal (CURIEs in text, ``data-cf-content-type`` /
``data-cf-objective-id`` / ``data-cf-source-ids`` attributes).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.curie_extraction import extract_curies as _extract_curies
from lib.validators.content_type import get_valid_chunk_types

logger = logging.getLogger(__name__)

# ``blocks.py`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# router's import bridge so ``from blocks import Block`` resolves
# regardless of how this module is loaded.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

# Cap the number of per-block issues each adapter emits so a uniformly
# broken outline batch doesn't drown the gate report. The router still
# sees ``passed=False`` regardless of issue count.
_ISSUE_LIST_CAP = 50

# Canonical sourceId pattern (kept in sync with
# lib/validators/source_refs.py:34 / source_reference.schema.json).
import re

_SOURCE_ID_RE = re.compile(r"^dart:[a-z0-9_-]+#[a-z0-9_-]+$")

# Scaffolding block types that carry no chunk-level ``content_type`` label.
# An ``objective`` block's canonical emit (``Block._objective_attrs`` /
# ``generate_course.py::_render_objectives``) is a bare ``<li
# data-cf-objective-id=... data-cf-bloom-level=...>`` — it deliberately does
# NOT stamp ``data-cf-content-type`` (objectives are not a ChunkType member;
# see ``lib/validators/content_type.py::get_valid_chunk_types``). ``chrome`` /
# ``recap`` are template scaffolding with no pedagogical content_type either.
# The content_type gate must skip these block types or it 100%-fails any
# objective-only / scaffolding-heavy page. Mirrors the skip set the grounding
# validators already use (``lib/validators/claim_support.py::_SKIPPED_BLOCK_TYPES``
# minus ``assessment_item`` — which DOES carry a content_type) and the
# scaffolding split in ``lib/validators/manifest_completeness.py``.
_CONTENT_TYPE_SCAFFOLDING_TYPES: frozenset = frozenset(
    {"objective", "chrome", "recap"}
)


def _coerce_blocks(inputs: Dict[str, Any]) -> Tuple[List[Block], Optional[GateIssue]]:
    """Pull a ``List[Block]`` out of ``inputs["blocks"]``.

    Returns ``(blocks, error_issue)``. ``error_issue`` is non-None
    when the input shape is wrong; the caller wraps it into a
    ``passed=False`` ``GateResult`` and skips the per-block walk.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _is_unshipped_escalation_tombstone(block: Block) -> bool:
    """Return True only for a marker-bearing block that never ships.

    The skip predicate is the EXACT ship-exclusion predicate the
    packager applies at HTML-emit time
    (``MCP/tools/pipeline_tools.py::_run_content_generation_rewrite``,
    the per-page emit filter at ``pipeline_tools.py:5226``):

        ``escalation_marker is not None and not (content or "").strip()``

    Two marker-bearing classes exist and the predicate must distinguish
    them — the original ``_is_escalated`` (which skipped on the marker
    alone) was wrong on the second:

    * **Marker + empty content** — a *tombstone*. The router stamped the
      marker (outline budget exhausted, validator consensus failure,
      structural-unfixable, dispatch error, per-claim attribution
      unfixable, …) and the block has no surviving content. The packager
      filters it OUT of the shipped IMSCC at HTML emit. Re-auditing it at
      the inter-tier / post-rewrite seam is a guaranteed false positive —
      it never ships, so this predicate returns True and the caller skips
      it (no audit, not counted toward ``audited``, no issue emitted).
    * **Marker + non-empty content** — a *salvaged* block. The
      escalated-rewrite salvage path
      (``Courseforge/generators/_rewrite_provider.py::_apply_rewrite_touch``
      via ``dataclasses.replace``, marker preserved) produces a block
      that carries a marker BUT also real content; per the design comment
      at ``pipeline_tools.py:5135`` it DOES ship and therefore MUST be
      audited by post_rewrite_validation. This predicate returns False
      for it so the caller audits it like any other block.

    Keep this predicate in lockstep with the packaging emit filter at
    ``pipeline_tools.py:5226`` — if one changes, the other must change
    with it, or escalated-with-content blocks will ship unaudited (or
    clean blocks will be skipped).

    The IMSCC-side backstop is ``lib/validators/imscc.py::
    IMSCCValidator._check_escalated_blocks_absent`` (code
    ``ESCALATED_BLOCK_IN_IMSCC``), which confirms no tombstone leaked
    into shipped HTML; in `textbook_to_course` that check is wired at
    warning severity, and in fully orchestrated runs it is input-starved
    (it needs ``blocks_final_path`` + the shipped HTML threaded in), so
    it is NOT a critical backstop — the lockstep skip predicate here is
    what keeps salvaged-with-content blocks on the audit path.
    """
    marker = getattr(block, "escalation_marker", None)
    if marker is None or marker == "":
        return False
    content = getattr(block, "content", None)
    if isinstance(content, str):
        body = content
    elif content is None:
        body = ""
    else:
        # dict-shaped outline content is never "empty" in the
        # ship-exclusion sense — it's pre-rewrite, so it cannot be a
        # tombstone (only the str-content rewrite tier ships HTML).
        return False
    return not body.strip()


def _outline_dict(block: Block) -> Optional[Dict[str, Any]]:
    """Return ``block.content`` if it is the outline-tier dict shape.

    Outline-tier Blocks carry a dict in ``content``; rewrite-tier
    Blocks carry an HTML string. Phase-3.5 shape-dispatching helpers
    (``_extract_curies`` / ``_extract_content_type`` /
    ``_extract_objective_refs`` / ``_extract_source_refs``) cover the
    str path; this helper remains the canonical dict-side accessor.
    """
    content = block.content
    if isinstance(content, dict):
        return content
    return None


# --------------------------------------------------------------------------- #
# H3 W1: per-block decision capture (Pattern A, borrowed from
# lib/validators/rewrite_source_grounding.py:268-311). One emit per
# validate() call per audited Block; never raises — capture failures
# log at DEBUG and pass through. ``inputs.get("decision_capture")`` is
# the canonical injection seam (S0.5 commit ``8914fce``); ``inputs.get
# ("capture")`` is honoured as a back-compat alias.
# --------------------------------------------------------------------------- #


def _resolve_capture(inputs: Dict[str, Any]) -> Any:
    """Pull the DecisionCapture instance from gate-runner inputs.

    Honours both keys S0.5 wired up: ``decision_capture`` (canonical)
    and ``capture`` (alias). Returns None when neither is present so
    the per-block emit helpers no-op silently.
    """
    capture = inputs.get("decision_capture")
    if capture is None:
        capture = inputs.get("capture")
    return capture


def _emit_block_decision(
    capture: Any,
    *,
    decision_type: str,
    block: Block,
    passed: bool,
    code: Optional[str],
    signals: Dict[str, Any],
) -> None:
    """Emit one Pattern A decision-capture event for a Block audit.

    `signals` carries validator-specific dynamic interpolations the
    rationale interpolates verbatim (block_id, gate counters, threshold
    values per H3 plan §3 W1 "Per-block dynamic signals"). The
    rationale is constructed once here so all four W1 validators share
    the same emit shape — sole differentiator is `decision_type` +
    the per-validator signals dict.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    # Render signals as `key=value` pairs for the rationale tail. Sort
    # by key so the rendered string is deterministic across runs (the
    # H3 regression suite asserts rationale-length floor; keeping the
    # render stable lets a future suite assert exact strings.).
    rendered_signals = ", ".join(
        f"{k}={signals[k]!r}" for k in sorted(signals.keys())
    )
    rationale = (
        f"Outline/rewrite-tier {decision_type} on Block "
        f"{block.block_id!r}: block_type={block.block_type}, "
        f"content_type={block.content_type_label or 'n/a'}, "
        f"failure_code={code or 'none'}, "
        f"signals: {rendered_signals}."
    )
    try:
        capture.log_decision(
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture failure must not abort the gate
        logger.debug(
            "DecisionCapture.log_decision raised on %s for %s: %s",
            decision_type,
            block.block_id,
            exc,
        )


# --------------------------------------------------------------------------- #
# Phase 3.5: shape-discriminating extraction helpers
# --------------------------------------------------------------------------- #

# Cheap HTML-to-text helper: strip tags + collapse whitespace. Avoids
# pulling in BeautifulSoup so the validator stays import-light. The
# CURIE regex in lib.ontology.curie_extraction is robust to surrounding
# punctuation, so a tag-strip is sufficient — we don't need a real DOM
# walk for the curie / objective / source extraction surfaces.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Used by the rewrite-tier (str-path) extractors. Not a full DOM
    parser — just enough to surface text content for regex matching.
    """
    if not html:
        return ""
    text = _HTML_TAG_RE.sub(" ", html)
    return _WHITESPACE_RE.sub(" ", text).strip()


# Attribute extractors. Each matches the canonical Courseforge emit
# pattern from Courseforge/scripts/blocks.py + generate_course.py:
#   data-cf-content-type="<chunk-type>"
#   data-cf-objective-id="<TO-NN>"  (per-element, may repeat)
#   data-cf-source-ids="dart:slug#blk[,dart:slug#blk2]"
# Quotes are normalised to double quotes by the renderer; we accept
# both for forward-compat with future emit changes.
_DATA_CF_CONTENT_TYPE_RE = re.compile(
    r'data-cf-content-type=["\']([^"\']+)["\']'
)
_DATA_CF_OBJECTIVE_ID_RE = re.compile(
    r'data-cf-objective-id=["\']([^"\']+)["\']'
)
_DATA_CF_SOURCE_IDS_RE = re.compile(
    r'data-cf-source-ids=["\']([^"\']+)["\']'
)


def _source_chunk_text_for_block(
    block: Block,
    source_chunks_by_block_id: Optional[Dict[str, Any]],
) -> str:
    """Concatenate the grounded source-chunk text for one block (FIX 1b).

    ``source_chunks_by_block_id`` is the ``{block_id: [{id, text, ...}]}``
    universe threaded into ``inputs`` by the caller (rehydrated from the
    outline tier's ``outline_chunks.json`` sidecar). A block's source
    chunks ARE its grounded provenance and ARE tagged with the domain
    vocabulary, so a minted-CURIE surface form present in this text is a
    legitimate anchor for that block.

    Only entries carrying a real ``text`` field count — a topic-paragraph
    fallback entry (no ``text`` key) contributes nothing, so an UNGROUNDED
    block gains no source-chunk surface and still fails closed when it
    carries no anchorable CURIE.

    Returns an empty string when the block has no resolved source chunks.
    """
    if not source_chunks_by_block_id:
        return ""
    entries = source_chunks_by_block_id.get(
        getattr(block, "block_id", ""), []
    ) or []
    parts: List[str] = []
    for e in entries:
        if isinstance(e, dict):
            txt = e.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    return "\n".join(parts)


def _extract_curies_from_block(block: Block) -> List[str]:
    """Shape-discriminating CURIE extractor for Subtask 6.

    Dict path (outline tier): returns ``block.content["curies"]``.
    Str path (rewrite tier): strips HTML and uses
    ``lib.ontology.curie_extraction.extract_curies`` over the surface
    text. Returns a list (sorted, deduplicated) so the gate's anchoring
    walk has a stable order regardless of dict/string source shape.
    """
    content = block.content
    if isinstance(content, dict):
        raw = content.get("curies") or []
        return [c for c in raw if isinstance(c, str) and c]
    if isinstance(content, str):
        text = _strip_html(content)
        return sorted(_extract_curies(text))
    return []


def _extract_content_type_from_block(block: Block) -> Optional[str]:
    """Shape-discriminating content_type extractor for Subtask 7.

    Dict path: returns ``block.content["content_type"]``.
    Str path: regex-matches the first ``data-cf-content-type``
    attribute in the HTML (canonical emit per
    Courseforge/CLAUDE.md § "HTML Data Attributes").
    """
    content = block.content
    if isinstance(content, dict):
        ctype = content.get("content_type")
        if isinstance(ctype, str) and ctype:
            return ctype
        return None
    if isinstance(content, str):
        match = _DATA_CF_CONTENT_TYPE_RE.search(content)
        if match:
            return match.group(1)
        return None
    return None


def _extract_objective_refs_from_block(block: Block) -> List[str]:
    """Shape-discriminating objective_id extractor for Subtask 8.

    Dict path: unions the refs from ``block.objective_ids`` (the
    structural field), ``content["objective_ids"]`` AND
    ``content["objective_refs"]``. The 7B outline tier puts its real
    refs in ``content["objective_refs"]`` (e.g. ``["CO-24","TO-01"]``),
    so consulting only the structural field / ``objective_ids`` read
    those blocks as "no objective refs". This mirrors how the router's
    own ``Courseforge/router/router.py::_candidate_covers_objective``
    reads the surface, keeping the validator and the router consistent.
    Str path: scrapes every ``data-cf-objective-id`` attribute from
    the rewrite-tier HTML. Multiple occurrences (one per ``<li>``) are
    expected and deduplicated.

    Returns a list preserving discovery order so the gate's "first
    miss" diagnostic stays readable.
    """
    structural = list(block.objective_ids or ())
    content = block.content
    if isinstance(content, dict):
        # Union all three surfaces, preserving discovery order. Mirrors
        # router.py::_candidate_covers_objective (objective_ids +
        # content["objective_refs"]/content["objective_ids"]).
        seen: List[str] = []
        seen_set: Set[str] = set()
        for oid in structural:
            if isinstance(oid, str) and oid and oid not in seen_set:
                seen.append(oid)
                seen_set.add(oid)
        for key in ("objective_refs", "objective_ids"):
            for oid in content.get(key) or []:
                if isinstance(oid, str) and oid and oid not in seen_set:
                    seen.append(oid)
                    seen_set.add(oid)
        return seen
    if isinstance(content, str):
        # Rewrite-tier: prefer the structural field when populated
        # (the rewrite provider preserves it on the immutable Block);
        # fall back to scraping the HTML for stand-alone callers.
        if structural:
            return structural
        seen = []
        seen_set = set()
        for match in _DATA_CF_OBJECTIVE_ID_RE.finditer(content):
            raw = match.group(1)
            if not raw:
                continue
            # The emit contract puts one LO id per attribute, but the 7B
            # rewrite tier sometimes packs multiple ids into a single
            # ``data-cf-objective-id="TO-04, CO-03, CO-07"`` (or
            # space-separated ``"CO-13 CO-20 CO-71"``) attribute. Split on
            # commas AND whitespace (mirrors the source-ids str-path split)
            # so each id resolves individually instead of the whole joined
            # string failing as one unknown id.
            for oid in re.split(r"[,\s]+", raw.strip()):
                oid = oid.strip()
                if oid and oid not in seen_set:
                    seen.append(oid)
                    seen_set.add(oid)
        return seen
    return structural


def _extract_source_refs_from_block(block: Block) -> List[str]:
    """Shape-discriminating sourceId extractor for Subtask 8.

    Dict path: harvests both ``block.source_references`` (preferred —
    canonical post-Wave-35 shape) and ``block.source_ids`` (legacy
    tuple).
    Str path: scrapes every ``data-cf-source-ids`` attribute, splitting
    on comma per the Courseforge emit contract (multiple ids on a
    single block separated by ``,``). Falls back to the structural
    fields when the HTML carries none (e.g. blocks with deferred
    source attribution).

    Returns a list preserving discovery order; the gate dedupes when
    walking the validation universe.
    """
    structural: List[str] = []
    for ref in block.source_references or ():
        if isinstance(ref, dict):
            sid = ref.get("sourceId")
            if isinstance(sid, str) and sid:
                structural.append(sid)
    for sid in block.source_ids or ():
        if isinstance(sid, str) and sid:
            structural.append(sid)

    content = block.content
    if isinstance(content, dict):
        return structural
    if isinstance(content, str):
        scraped: List[str] = []
        for match in _DATA_CF_SOURCE_IDS_RE.finditer(content):
            for sid in match.group(1).split(","):
                sid = sid.strip()
                if sid:
                    scraped.append(sid)
        # Prefer the union of structural + scraped: rewrite-tier blocks
        # may carry source_ids on either surface. Deduplicate while
        # preserving order.
        seen: Set[str] = set()
        merged: List[str] = []
        for sid in structural + scraped:
            if sid not in seen:
                seen.add(sid)
                merged.append(sid)
        return merged
    return structural


# --------------------------------------------------------------------------- #
# 1. CURIE anchoring
# --------------------------------------------------------------------------- #


# Common-word stoplist for the minted-CURIE surface-form anchoring
# guard (R4). A surface form that is a bare article / preposition /
# copula / conjunction must NOT be allowed to anchor a minted CURIE —
# such a "form" matches almost any English prose and vacuously passes
# the gate. This is a *small* curated set (NOT a length cut): legit
# short domain terms like "pH" / "DNA" / "RNA" survive because they are
# not in the list. 1-character surface forms are dropped outright.
# Kept local rather than reusing concept_tagging.NON_CONCEPT_TAGS —
# that set is Bloom verbs + course logistics, not function words.
_CURIE_ANCHOR_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "for", "yet", "so",
    "of", "to", "in", "on", "at", "by", "as", "is", "are", "was",
    "were", "be", "been", "being", "am", "it", "its", "this", "that",
    "these", "those", "with", "from", "into", "than", "then", "if",
    "no", "not", "we", "you", "he", "she", "they", "i",
})


def _surface_form_can_anchor(surface_form: str) -> bool:
    """Return True when ``surface_form`` is allowed to anchor a CURIE.

    R4 short/common-token guard: rejects 1-character forms and bare
    common English function words (``_CURIE_ANCHOR_STOPWORDS``). Applied
    here in ``_curie_anchored`` — the cleanest spot — rather than in
    ``build_minted_curie_map``: keeping the filter at match time means
    ``build_minted_curie_map`` stays a pure data transform whose
    ``surface_forms`` list other consumers (e.g.
    ``minted_curie_by_canonical``) still see in full.
    """
    if not isinstance(surface_form, str):
        return False
    sf = surface_form.strip()
    if len(sf) < 2:
        return False
    if sf.lower() in _CURIE_ANCHOR_STOPWORDS:
        return False
    return True


def _curie_anchored(
    curie: str,
    surface_curies: "Set[str]",
    text_blob: str,
    minted_curie_map: Optional[Dict[str, Any]],
    source_chunk_text: str = "",
) -> bool:
    """Return True when ``curie`` is anchored in the block's surface.

    Two anchoring contracts, dispatched on whether the CURIE is a
    minted (corpus-derived) CURIE or an RDF CURIE:

    * **Minted CURIE** — ``curie`` is a key in ``minted_curie_map``. It
      is "anchored" when EITHER the literal CURIE token appears in
      ``surface_curies`` (after the R1 force-injection fix the minted
      token lands in the str-path text, so force-injected blocks pass
      legitimately) OR any of that CURIE's vocabulary ``surface_forms``
      (the concept's canonical name + aliases) appears as a
      **word-boundary** match in the block's ``text_blob`` (the
      ``key_claims`` prose) OR in the block's ``source_chunk_text``
      (its grounded provenance — the source chunks ARE tagged with the
      domain vocabulary, so a concept present there is genuinely the
      block's subject; this is provenance-aware anchoring, NOT a
      relaxation). Surface forms that are too short or are common
      English function words are filtered out (see
      :func:`_surface_form_can_anchor`) so a form like ``"ion"`` cannot
      vacuously match inside "definition".
    * **RDF CURIE** — ``curie`` is NOT in the map. The legacy literal-
      token check is preserved verbatim: the CURIE counts as anchored
      only when the literal token appears in ``surface_curies`` (the
      CURIEs ``extract_curies`` scraped from the surface text). RDF
      CURIEs never consult ``source_chunk_text``.

    When ``minted_curie_map`` is ``None`` / empty, every CURIE takes the
    RDF arm — byte-identical to the pre-minting contract.
    """
    if minted_curie_map and curie in minted_curie_map:
        # Literal-token arm: after the R1 force-injection fix the minted
        # CURIE token survives into the str-path text, so a literal
        # match is a legitimate pass for force-injected blocks.
        if curie in surface_curies:
            return True
        entry = minted_curie_map.get(curie) or {}
        surface_forms = entry.get("surface_forms") if isinstance(entry, dict) else None
        if not surface_forms:
            # Minted CURIE with no surface forms — literal check only
            # (already failed above) rather than auto-passing.
            return False
        # Provenance-aware anchoring: the concept counts as discussed
        # when its surface form appears in the block's key_claims prose
        # OR in the block's grounded source-chunk text.
        blob = "\n".join(b for b in (text_blob, source_chunk_text) if b)
        for sf in surface_forms:
            if not _surface_form_can_anchor(sf):
                continue
            if re.search(
                r"\b" + re.escape(sf.strip()) + r"\b", blob, re.IGNORECASE
            ):
                return True
        return False
    return curie in surface_curies


class BlockCurieAnchoringValidator:
    """Outline-tier CURIE-anchoring gate.

    Ports the per-pair anchoring check from
    ``lib/validators/curie_anchoring.py`` to the Block-list shape:
    every outline-tier Block's ``content["curies"]`` must be
    non-empty, AND at least one of those CURIEs must appear in the
    block's textual surface (``content["key_claims"]``). A miss is
    a content-side semantic problem the rewrite tier could fix on a
    re-roll, so the gate emits ``action="regenerate"``.

    Concept-aware anchoring (v0.3.0): when ``inputs["minted_curie_map"]``
    is supplied — a ``{minted_curie: {"canonical", "surface_forms"}}``
    dict built from the per-course domain-concept vocabulary — a CURIE
    that is a key in that map is "anchored" when any of its
    ``surface_forms`` appears (case-insensitive substring) in the block
    text blob, instead of requiring the synthetic CURIE token literally.
    RDF CURIEs (not in the map) keep the legacy literal-token check.
    When no ``minted_curie_map`` is supplied, behaviour is byte-identical
    to the pre-minting contract.
    """

    name = "outline_curie_anchoring"
    version = "1.2.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = _resolve_capture(inputs)
        minted_curie_map = inputs.get("minted_curie_map") or None
        if not isinstance(minted_curie_map, dict):
            minted_curie_map = None
        # FIX 1b: per-block grounded source-chunk universe. A minted CURIE
        # whose surface form appears in a block's source-chunk text is
        # legitimately anchored (provenance-aware). No-op when absent
        # (legacy callers) — anchoring then runs over key_claims only,
        # byte-identical to the pre-fix contract.
        source_chunks_by_block_id = inputs.get("source_chunks_by_block_id")
        if not isinstance(source_chunks_by_block_id, dict):
            source_chunks_by_block_id = None
        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            # Skip only an UNSHIPPED escalation tombstone (marker +
            # empty content): the packager filters it out of the IMSCC,
            # so auditing it is a false positive. A salvaged block
            # (marker + content) DOES ship and MUST stay on the audit
            # path. Predicate mirrors pipeline_tools.py:5226.
            if _is_unshipped_escalation_tombstone(block):
                continue
            content = block.content
            # Phase 3.5: shape-dispatch. Dict and str paths share the
            # CURIE-anchoring contract (declared CURIEs must appear in
            # the textual surface) but pull the surface from different
            # shapes.
            if isinstance(content, dict):
                audited += 1
                curies = _extract_curies_from_block(block)
                # Wave 1.5 W1.5.A: ``key_claims`` is now a back-compat
                # ``oneOf``: legacy ``List[str]`` OR structured
                # ``List[{claim, source_chunk_ids[]}]``. Pull the
                # surface text via shape-dispatch so both arms feed
                # the CURIE-anchoring extractor identically. Defensive:
                # ignore non-str / non-dict entries (the schema rejects
                # them upstream, but the validator never crashes on a
                # malformed Block).
                claims_raw = content.get("key_claims") or []
                text_blob_parts: List[str] = []
                for c in claims_raw:
                    if isinstance(c, str):
                        text_blob_parts.append(c)
                    elif isinstance(c, dict):
                        claim_text = c.get("claim", "")
                        if isinstance(claim_text, str):
                            text_blob_parts.append(claim_text)
                text_blob = "\n".join(text_blob_parts)
                surface_curies = _extract_curies(text_blob)
            elif isinstance(content, str):
                audited += 1
                curies = _extract_curies_from_block(block)
                # Rewrite-tier: declared CURIEs == surfaced CURIEs by
                # construction (the extractor scrapes them from the
                # HTML body). The "miss" condition collapses to "no
                # CURIEs in the HTML body at all", caught by the
                # ``if not curies`` branch below. Anchoring is
                # tautologically satisfied when curies is non-empty.
                surface_curies = set(curies)
                # Stripped-HTML surface for minted-CURIE surface-form
                # anchoring. The rewrite tier carries the minted CURIE
                # token verbatim (force-injected — see
                # ``_rewrite_provider``), so the literal check already
                # passes; the prose surface is still threaded so a
                # surface-form fallback is available when supplied.
                text_blob = _strip_html(content)
            else:
                # Non-dict / non-str content — nothing to audit. No
                # capture emit either; the "block" wasn't actually a
                # member of the validator's input universe.
                continue

            # FIX 1b: only a GROUNDED block (real source chunks carrying
            # text) contributes a source-chunk anchoring surface. An
            # ungrounded block gets an empty string here and still fails
            # closed when it carries no anchorable CURIE (no fabrication).
            source_chunk_text = _source_chunk_text_for_block(
                block, source_chunks_by_block_id
            )
            anchored_count = sum(
                1
                for c in curies
                if _curie_anchored(
                    c, surface_curies, text_blob, minted_curie_map,
                    source_chunk_text,
                )
            )
            anchoring_rate = (
                (anchored_count / len(curies)) if curies else 0.0
            )
            # Signal whether this block carried any minted (corpus-
            # derived) CURIE — lets a postmortem distinguish a prose
            # corpus's surface-form anchoring from an RDF corpus's
            # literal-token anchoring.
            minted_curie_count = (
                sum(1 for c in curies if c in minted_curie_map)
                if minted_curie_map
                else 0
            )

            if not curies:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_MISSING_CURIES",
                        message=(
                            f"Block {block.block_id!r} carries no CURIEs "
                            f"(dict path: empty content['curies']; str path: "
                            f"no CURIEs detected in HTML surface). Phase 3 "
                            f"contract requires at least one anchoring CURIE "
                            f"per block."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Re-roll the outline-tier provider; ensure the "
                            "system prompt requests at least one CURIE per "
                            "Block and the local-model JSON-schema directive "
                            "marks ``curies`` as a non-empty array."
                        ),
                    ))
                _emit_block_decision(
                    capture,
                    decision_type="block_curie_anchoring_check",
                    block=block,
                    passed=False,
                    code="OUTLINE_BLOCK_MISSING_CURIES",
                    signals={
                        "curies_count": 0,
                        "anchored_count": 0,
                        "anchoring_rate": 0.0,
                        "min_rate_threshold": 1.0,
                        "surface_curies_count": len(surface_curies),
                        "minted_curie_count": 0,
                        "minted_anchoring_used": minted_curie_map is not None,
                    },
                )
                continue

            if anchored_count >= 1:
                passed_count += 1
                _emit_block_decision(
                    capture,
                    decision_type="block_curie_anchoring_check",
                    block=block,
                    passed=True,
                    code=None,
                    signals={
                        "curies_count": len(curies),
                        "anchored_count": anchored_count,
                        "anchoring_rate": round(anchoring_rate, 4),
                        "min_rate_threshold": 1.0,
                        "surface_curies_count": len(surface_curies),
                        "minted_curie_count": minted_curie_count,
                        "minted_anchoring_used": minted_curie_map is not None,
                    },
                )
            else:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_CURIE_NOT_ANCHORED",
                        message=(
                            f"Block {block.block_id!r} declares CURIEs "
                            f"{curies!r} but none of them appear in the "
                            f"block's textual surface (dict path: "
                            f"content['key_claims']; str path: rendered "
                            f"HTML body)."
                        ),
                        location=block.block_id,
                    ))
                _emit_block_decision(
                    capture,
                    decision_type="block_curie_anchoring_check",
                    block=block,
                    passed=False,
                    code="OUTLINE_BLOCK_CURIE_NOT_ANCHORED",
                    signals={
                        "curies_count": len(curies),
                        "anchored_count": 0,
                        "anchoring_rate": 0.0,
                        "min_rate_threshold": 1.0,
                        "surface_curies_count": len(surface_curies),
                        "minted_curie_count": minted_curie_count,
                        "minted_anchoring_used": minted_curie_map is not None,
                    },
                )

        passed = len(issues) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


# --------------------------------------------------------------------------- #
# 2. Content-type taxonomy
# --------------------------------------------------------------------------- #


class BlockContentTypeValidator:
    """Outline-tier content-type taxonomy gate.

    Wraps ``lib/validators/content_type.py::get_valid_chunk_types`` so
    every outline-tier Block's ``content["content_type"]`` must be a
    member of the canonical taxonomy (``schemas/taxonomies/content_type.json``).
    A miss is a content-side typo / hallucination the rewrite tier
    could fix on a re-roll → ``action="regenerate"``.
    """

    name = "outline_content_type"
    version = "1.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = _resolve_capture(inputs)
        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        valid_types: Set[str] = set(get_valid_chunk_types())
        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            # Skip unshipped tombstone only; salvaged-with-content stays
            # on the audit path (predicate mirrors pipeline_tools.py:5226).
            if _is_unshipped_escalation_tombstone(block):
                continue
            # Skip scaffolding block types (objective / chrome / recap):
            # their canonical emit carries no data-cf-content-type and they
            # are not ChunkType members, so auditing them for a content_type
            # is a guaranteed false positive (an objective-only page would
            # 100%-fail this gate). NOT counted toward ``audited``.
            if block.block_type in _CONTENT_TYPE_SCAFFOLDING_TYPES:
                continue
            content = block.content
            # Phase 3.5: shape-dispatch — dict and str paths both
            # extract a single content_type label, then validate it
            # against the canonical chunk-type taxonomy (Trainforge-side
            # enum, NOT the section-level content-type taxonomy — see
            # Worker M's flagged inconsistency in Courseforge/CLAUDE.md
            # § "Phase 3: outline-rewrite two-pass router").
            if not isinstance(content, (dict, str)):
                continue
            audited += 1
            ctype = _extract_content_type_from_block(block)
            if not isinstance(ctype, str) or not ctype:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_MISSING_CONTENT_TYPE",
                        message=(
                            f"Block {block.block_id!r} is missing a "
                            f"content_type label (dict path: "
                            f"content['content_type']; str path: "
                            f"data-cf-content-type attribute). Got "
                            f"{ctype!r}."
                        ),
                        location=block.block_id,
                    ))
                _emit_block_decision(
                    capture,
                    decision_type="block_content_type_check",
                    block=block,
                    passed=False,
                    code="OUTLINE_BLOCK_MISSING_CONTENT_TYPE",
                    signals={
                        "declared_content_type": None,
                        "valid": False,
                        "expected_enum_size": len(valid_types),
                    },
                )
                continue
            if ctype not in valid_types:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_INVALID_CONTENT_TYPE",
                        message=(
                            f"Block {block.block_id!r} declares "
                            f"content_type={ctype!r} which is not in the "
                            f"canonical ChunkType taxonomy. Valid values: "
                            f"{sorted(valid_types)}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Re-roll the outline-tier provider with the "
                            "JSON-schema directive enumerating the valid "
                            "content_type values, or correct the rewrite-"
                            "tier emit to stamp data-cf-content-type with "
                            "a taxonomy-compliant label."
                        ),
                    ))
                _emit_block_decision(
                    capture,
                    decision_type="block_content_type_check",
                    block=block,
                    passed=False,
                    code="OUTLINE_BLOCK_INVALID_CONTENT_TYPE",
                    signals={
                        "declared_content_type": ctype,
                        "valid": False,
                        "expected_enum_size": len(valid_types),
                    },
                )
            else:
                passed_count += 1
                _emit_block_decision(
                    capture,
                    decision_type="block_content_type_check",
                    block=block,
                    passed=True,
                    code=None,
                    signals={
                        "declared_content_type": ctype,
                        "valid": True,
                        "expected_enum_size": len(valid_types),
                    },
                )

        passed = len(issues) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


# --------------------------------------------------------------------------- #
# 3. Page objective coverage
# --------------------------------------------------------------------------- #


def _harvest_objective_id(entry: Any, ids: Set[str]) -> None:
    """Add an objective ID from a flat ``{id, ...}`` dict entry."""
    if isinstance(entry, dict):
        oid = entry.get("id") or entry.get("objective_id")
        if isinstance(oid, str) and oid:
            ids.add(oid)


def _harvest_grouped_objectives(raw: Any, ids: Set[str]) -> None:
    """Harvest IDs from any ``chapter_objectives`` / ``component_objectives``
    shape (nested list-of-groups, flat list, or dict-of-lists).

    Reuses the canonical ``_normalize_chapter_objectives_to_groups``
    normalizer so this loader handles the same three shapes the rest of
    the pipeline does (Wave2b/Wave2c). The real synthesizer emits the
    nested group form
    ``[{"chapter": "Week N", "objectives": [{"id": "CO-NN", ...}]}]``;
    the legacy flat ``[{"id": "CO-NN", ...}]`` form is also accepted.
    """
    if not raw:
        return
    try:
        # Lazy import: pipeline_tools imports inter_tier_gates, so a
        # module-level import would form a cycle. The normalizer body
        # is pure-stdlib, so the lazy import is import-safe here.
        from MCP.tools.pipeline_tools import (  # noqa: PLC0415
            _normalize_chapter_objectives_to_groups,
        )

        groups = _normalize_chapter_objectives_to_groups(raw)
    except Exception:  # pragma: no cover - defensive fallback
        # Mirror the normalizer's logic if the import is unavailable.
        groups = []
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict) and isinstance(
                    entry.get("objectives"), list
                ):
                    groups.append(entry)
                elif isinstance(entry, dict):
                    _harvest_objective_id(entry, ids)
        elif isinstance(raw, dict):
            for items in raw.values():
                if isinstance(items, list):
                    groups.append({"objectives": items})
    for group in groups:
        if isinstance(group, dict):
            for obj in group.get("objectives") or []:
                _harvest_objective_id(obj, ids)


def _load_canonical_objectives(path: Path) -> Set[str]:
    """Load canonical objective IDs from a course.json /
    synthesized_objectives.json file.

    Accepts the Courseforge synthesized form, the LibV2 archive form,
    AND a bare top-level flat list of ``{id, ...}`` entries (some
    sidecars like ``01_outline/outline_objectives.json`` ship a flat
    70-item list). The grouped ``chapter_objectives`` /
    ``component_objectives`` keys are parsed through the canonical
    ``_normalize_chapter_objectives_to_groups`` normalizer so the
    nested ``[{"chapter": ..., "objectives": [{"id": ...}]}]`` group
    shape, the flat-list shape, and the dict-of-lists shape are all
    harvested.

    Returns an empty set when the file is missing / unparseable
    so the caller can decide fail-vs-warn semantics.
    """
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    ids: Set[str] = set()

    # Top-level flat list (e.g. outline_objectives.json sidecar).
    if isinstance(data, list):
        for entry in data:
            _harvest_objective_id(entry, ids)
        return ids

    if not isinstance(data, dict):
        return ids

    # Courseforge synthesized + LibV2 archive flat terminal lists.
    for key in ("terminal_objectives", "terminal_outcomes"):
        for entry in data.get(key, []) or []:
            _harvest_objective_id(entry, ids)

    # Grouped chapter/component objectives (any of the three shapes).
    for key in ("chapter_objectives", "component_objectives"):
        _harvest_grouped_objectives(data.get(key), ids)

    return ids


class BlockPageObjectivesValidator:
    """Outline-tier objective coverage gate.

    Each outline-tier Block must reference one or more canonical
    learning objectives via ``block.objective_ids``. The reference
    must resolve against the canonical objectives JSON declared by
    ``inputs['objectives_path']`` (Courseforge ``synthesized_objectives.json``
    or LibV2 archive form). A reference to an unknown objective is
    a structural miss — re-rolling won't help because the underlying
    objectives JSON doesn't carry the missing ID. → ``action="block"``.
    """

    name = "outline_page_objectives"
    version = "1.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = _resolve_capture(inputs)
        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        objectives_path_raw = inputs.get("objectives_path")
        canonical_ids: Optional[Set[str]] = None
        if objectives_path_raw:
            canonical_ids = _load_canonical_objectives(Path(objectives_path_raw))

        # Allow tests / direct callers to seed the objective set.
        # Per Worker M's flag (Phase 3 review), this validator's input
        # contract uses ``valid_objective_ids`` (asymmetric with the
        # other three Block validators that take only ``blocks``).
        seeded = inputs.get("valid_objective_ids")
        if seeded is not None:
            canonical_ids = {str(o) for o in seeded}

        valid_objective_ids_count = (
            len(canonical_ids) if canonical_ids is not None else 0
        )

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            # Skip unshipped tombstone only; salvaged-with-content stays
            # on the audit path (predicate mirrors pipeline_tools.py:5226).
            if _is_unshipped_escalation_tombstone(block):
                continue
            content = block.content
            # Phase 3.5: shape-dispatch — both paths extract a list of
            # objective_id refs and validate against the canonical set.
            # The dict path historically only consulted block.objective_ids;
            # the str path falls back to scraping data-cf-objective-id
            # attributes when the structural field is empty (unlikely
            # for rewrite-tier output, but the helper is defensive).
            if not isinstance(content, (dict, str)):
                continue
            audited += 1
            obj_ids = _extract_objective_refs_from_block(block)
            if not obj_ids:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_MISSING_OBJECTIVE_REF",
                        message=(
                            f"Block {block.block_id!r} declares no "
                            f"objective_ids (dict path: empty "
                            f"block.objective_ids; str path: no "
                            f"data-cf-objective-id attributes in HTML). "
                            f"Every block must reference at least one "
                            f"canonical TO-NN/CO-NN objective."
                        ),
                        location=block.block_id,
                    ))
                _emit_block_decision(
                    capture,
                    decision_type="block_page_objectives_check",
                    block=block,
                    passed=False,
                    code="OUTLINE_BLOCK_MISSING_OBJECTIVE_REF",
                    signals={
                        "declared_objective_ids": [],
                        "unresolved_count": 0,
                        "valid_objective_ids_count": valid_objective_ids_count,
                    },
                )
                continue
            if canonical_ids is None:
                # No canonical universe to check against — count as
                # passing the structural check (a non-empty reference
                # is the only thing we can audit).
                passed_count += 1
                _emit_block_decision(
                    capture,
                    decision_type="block_page_objectives_check",
                    block=block,
                    passed=True,
                    code=None,
                    signals={
                        "declared_objective_ids": list(obj_ids),
                        "unresolved_count": 0,
                        "valid_objective_ids_count": valid_objective_ids_count,
                    },
                )
                continue
            unknown = [oid for oid in obj_ids if oid not in canonical_ids]
            if unknown:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_UNKNOWN_OBJECTIVE",
                        message=(
                            f"Block {block.block_id!r} references "
                            f"objective_ids {unknown!r} that do not resolve "
                            f"against the canonical objectives JSON at "
                            f"{objectives_path_raw!r} (input key: "
                            f"valid_objective_ids)."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Either correct the objective_id reference in "
                            "the outline tier / rewrite-tier HTML emit, or "
                            "extend synthesized_objectives.json upstream."
                        ),
                    ))
                _emit_block_decision(
                    capture,
                    decision_type="block_page_objectives_check",
                    block=block,
                    passed=False,
                    code="OUTLINE_BLOCK_UNKNOWN_OBJECTIVE",
                    signals={
                        "declared_objective_ids": list(obj_ids),
                        "unresolved_count": len(unknown),
                        "valid_objective_ids_count": valid_objective_ids_count,
                    },
                )
            else:
                passed_count += 1
                _emit_block_decision(
                    capture,
                    decision_type="block_page_objectives_check",
                    block=block,
                    passed=True,
                    code=None,
                    signals={
                        "declared_objective_ids": list(obj_ids),
                        "unresolved_count": 0,
                        "valid_objective_ids_count": valid_objective_ids_count,
                    },
                )

        passed = len(issues) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "block",
        )


# --------------------------------------------------------------------------- #
# 4. Source-ref manifest resolution
# --------------------------------------------------------------------------- #


def _resolve_against_manifest(manifest_path: Optional[Path]) -> Set[str]:
    """Harvest the valid sourceId universe from a DART staging manifest.

    Mirrors ``lib/validators/source_refs.py::_collect_valid_ids`` but
    accepts the path directly (no ``inputs`` indirection). Returns
    an empty set when the manifest is missing / unparseable; the
    caller passes ``passed=True`` with an EMPTY-MANIFEST warning.
    """
    import json

    if manifest_path is None or not manifest_path.exists():
        return set()

    valid: Set[str] = set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    # The manifest has shape: {"files": [{"path": ..., "role": ...}, ...]}
    # plus per-block sourceId entries the source-router emits. We accept
    # both: explicit ``valid_source_ids`` field at the top, or harvested
    # from provenance_sidecar entries.
    explicit = manifest.get("valid_source_ids")
    if isinstance(explicit, list):
        for sid in explicit:
            if isinstance(sid, str) and sid:
                valid.add(sid)
    files = manifest.get("files", []) or []
    if isinstance(files, list):
        # Future-compat: harvest from provenance sidecars sitting next
        # to the manifest. Keep the import shape thin to avoid pulling
        # in the full source_refs._iter_sidecar_block_ids walker; the
        # outline-tier seam doesn't need sidecar discovery in this
        # round (the manifest path direct seed is enough for the
        # blocking-action path).
        pass

    return valid


class BlockSourceRefValidator:
    """Outline-tier source-ref manifest gate.

    Every outline-tier Block's ``block.source_references`` (or the
    ``source_ids`` tuple) must resolve against the DART staging
    manifest at ``inputs['manifest_path']``. A miss is structural —
    the Block references a ``sourceId`` that doesn't exist in the
    staging manifest, so the rewrite tier has nothing to ground on.
    → ``action="block"``.

    Wave 1.5 W1.5.C extension: when ``block.content`` is the outline-
    tier dict shape carrying the structured ``key_claims`` shape
    (``List[{claim, source_chunk_ids[]}]``), every chunk_id listed in
    ``claim.source_chunk_ids[]`` MUST also appear in the block's
    top-level resolved ``source_refs[]``. Per-claim misses fire
    ``OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS`` (severity=warning) and
    map to ``action="regenerate"`` — semantically the model can re-
    roll and pick the right chunk. Block-level structural misses
    still dominate: when both fire, ``action="block"`` wins.

    Legacy ``List[str]`` ``key_claims`` shape skips the per-claim walk
    (back-compat preserved per Wave 1.5 §6.1).
    """

    name = "outline_source_refs"
    version = "1.2.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = _resolve_capture(inputs)
        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        manifest_path_raw = inputs.get("manifest_path")
        manifest_path = (
            Path(manifest_path_raw) if manifest_path_raw else None
        )
        valid_ids: Set[str] = _resolve_against_manifest(manifest_path)

        # Test seam — direct injection bypasses manifest discovery.
        seeded = inputs.get("valid_source_ids")
        if seeded is not None:
            valid_ids = {str(s) for s in seeded}

        # Per H3 W1 dynamic-signal contract: the staging-dir signal is
        # the manifest path's parent. None when no manifest is wired.
        staging_dir = (
            str(manifest_path.parent) if manifest_path is not None else None
        )

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0
        empty_manifest = manifest_path is not None and not valid_ids

        for block in blocks:
            # Skip unshipped tombstone only; salvaged-with-content stays
            # on the audit path (predicate mirrors pipeline_tools.py:5226).
            if _is_unshipped_escalation_tombstone(block):
                continue
            content = block.content
            # Phase 3.5: shape-dispatch — both paths extract a list of
            # source_id refs. Dict path harvests from
            # block.source_references + block.source_ids (legacy);
            # str path additionally scrapes data-cf-source-ids
            # attributes from the rewrite-tier HTML.
            if not isinstance(content, (dict, str)):
                continue
            audited += 1
            block_ids: List[str] = _extract_source_refs_from_block(block)

            if not block_ids:
                # No source_ids on this block — Blocks are allowed to
                # defer source attribution when no DART grounding
                # applies, so an empty list passes the structural check
                # on both tiers.
                passed_count += 1
                _emit_block_decision(
                    capture,
                    decision_type="block_source_ref_check",
                    block=block,
                    passed=True,
                    code=None,
                    signals={
                        "declared_source_ids_count": 0,
                        "unresolved_count": 0,
                        "staging_dir": staging_dir,
                        "valid_ids_universe_size": len(valid_ids),
                    },
                )
                continue

            block_passed = True
            unresolved_count = 0
            failure_code: Optional[str] = None
            # W1.5.C: track block-level vs claim-level misses separately
            # so the action resolver can prefer "block" when ANY block-
            # level structural miss fired (block-level dominates).
            block_level_miss = False
            claim_unresolved = False
            claim_misses_count = 0
            for sid in block_ids:
                if not _SOURCE_ID_RE.match(sid):
                    block_passed = False
                    block_level_miss = True
                    unresolved_count += 1
                    failure_code = "OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE"
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="critical",
                            code="OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE",
                            message=(
                                f"Outline-tier Block {block.block_id!r} "
                                f"declares sourceId {sid!r} which does not "
                                f"match the canonical dart:{{slug}}#{{block_id}} "
                                f"shape."
                            ),
                            location=block.block_id,
                        ))
                    continue
                if valid_ids and sid not in valid_ids:
                    block_passed = False
                    block_level_miss = True
                    unresolved_count += 1
                    if failure_code is None:
                        failure_code = "OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID"
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="critical",
                            code="OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID",
                            message=(
                                f"Outline-tier Block {block.block_id!r} "
                                f"declares sourceId {sid!r} which does not "
                                f"resolve against the staging manifest at "
                                f"{manifest_path_raw!r}."
                            ),
                            location=block.block_id,
                            suggestion=(
                                "Re-run stage_dart_outputs to regenerate "
                                "the manifest, or correct the source-router "
                                "binding upstream."
                            ),
                        ))

            # W1.5.C per-claim attribution consistency walk. Operates on
            # the outline-tier dict shape only; rewrite-tier (str)
            # blocks are skipped — their per-claim attribution lives
            # in the rewrite-tier per-claim citation map (W1.5.D).
            # Legacy ``List[str]`` claim shape silently skips the per-
            # claim walk (back-compat preserved per Wave 1.5 §6.1).
            claims_total_count = 0
            claims_with_attribution_count = 0
            if isinstance(content, dict):
                claims_raw = content.get("key_claims") or []
                if isinstance(claims_raw, list):
                    claims_total_count = len(claims_raw)
                    block_sids_set = set(block_ids)
                    for idx, c in enumerate(claims_raw):
                        if not isinstance(c, dict):
                            # legacy str entry — skip (back-compat).
                            continue
                        claim_sids_raw = c.get("source_chunk_ids") or []
                        if not isinstance(claim_sids_raw, list):
                            continue
                        claims_with_attribution_count += 1
                        for sid in claim_sids_raw:
                            if not isinstance(sid, str) or not sid:
                                continue
                            if sid not in block_sids_set:
                                block_passed = False
                                claim_unresolved = True
                                claim_misses_count += 1
                                if failure_code is None:
                                    failure_code = (
                                        "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS"
                                    )
                                if len(issues) < _ISSUE_LIST_CAP:
                                    issues.append(GateIssue(
                                        severity="warning",
                                        code=(
                                            "OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS"
                                        ),
                                        message=(
                                            f"Block {block.block_id!r} "
                                            f"key_claims[{idx}] cites "
                                            f"source_chunk_id {sid!r} which is "
                                            f"not in the block's top-level "
                                            f"source_refs[]."
                                        ),
                                        location=block.block_id,
                                        suggestion=(
                                            "Re-roll the outline tier so "
                                            "every claim cites a chunk_id "
                                            "already declared in the block's "
                                            "source_refs[], or extend "
                                            "source_refs[] to cover the "
                                            "claim's chunk universe."
                                        ),
                                    ))

            # Per-block decision-capture emit. W1.5.C surfaces three new
            # claim-level signals so the audit trail records per-claim
            # attribution health alongside the existing block-level
            # source-ref counts. Legacy ``List[str]`` shape contributes
            # ``claims_with_attribution_count=0``.
            block_signals = {
                "declared_source_ids_count": len(block_ids),
                "unresolved_count": unresolved_count,
                "staging_dir": staging_dir,
                "valid_ids_universe_size": len(valid_ids),
                "claim_misses_count": claim_misses_count,
                "claims_with_attribution_count": claims_with_attribution_count,
                "claims_total_count": claims_total_count,
            }
            if block_passed:
                passed_count += 1
                _emit_block_decision(
                    capture,
                    decision_type="block_source_ref_check",
                    block=block,
                    passed=True,
                    code=None,
                    signals=block_signals,
                )
            else:
                _emit_block_decision(
                    capture,
                    decision_type="block_source_ref_check",
                    block=block,
                    passed=False,
                    code=failure_code,
                    signals=block_signals,
                )

        # Empty-manifest path: when the manifest is empty / missing
        # AND no Block declared a sourceId, pass with an info note.
        if empty_manifest and not issues and audited == passed_count:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="info",
                    code="EMPTY_STAGING_MANIFEST",
                    message=(
                        f"Staging manifest at {manifest_path_raw!r} is "
                        f"empty or has no harvestable sourceIds; the "
                        f"outline tier did not declare any source_ids. "
                        f"Gate passes."
                    ),
                )],
                action=None,
            )

        passed = len(issues) == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)

        # W1.5.C action resolution: block-level structural misses
        # dominate over claim-level semantic misses. Inspect the
        # collected issue codes:
        # - any OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE /
        #   OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID  → action="block"
        # - only OUTLINE_CLAIM_SOURCE_NOT_IN_BLOCK_REFS  → "regenerate"
        # - no failures                                  → action=None
        if passed:
            action: Optional[str] = None
        else:
            block_level_codes = {
                "OUTLINE_BLOCK_INVALID_SOURCE_ID_SHAPE",
                "OUTLINE_BLOCK_UNRESOLVED_SOURCE_ID",
            }
            has_block_level = any(
                i.code in block_level_codes for i in issues
            )
            action = "block" if has_block_level else "regenerate"

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )


__all__ = [
    "BlockCurieAnchoringValidator",
    "BlockContentTypeValidator",
    "BlockPageObjectivesValidator",
    "BlockSourceRefValidator",
]
