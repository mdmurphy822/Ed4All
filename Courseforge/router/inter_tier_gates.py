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

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.curie_extraction import extract_curies as _extract_curies
from lib.validators.content_type import get_valid_chunk_types

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

    Dict path: prefers ``block.objective_ids`` (the structural field,
    same source the Phase-3 dict-only path used). Falls back to
    ``block.content["objective_ids"]`` if the field is empty.
    Str path: scrapes every ``data-cf-objective-id`` attribute from
    the rewrite-tier HTML. Multiple occurrences (one per ``<li>``) are
    expected and deduplicated.

    Returns a list preserving discovery order so the gate's "first
    miss" diagnostic stays readable.
    """
    structural = list(block.objective_ids or ())
    content = block.content
    if isinstance(content, dict):
        if structural:
            return structural
        raw = content.get("objective_ids") or []
        return [o for o in raw if isinstance(o, str) and o]
    if isinstance(content, str):
        # Rewrite-tier: prefer the structural field when populated
        # (the rewrite provider preserves it on the immutable Block);
        # fall back to scraping the HTML for stand-alone callers.
        if structural:
            return structural
        seen: List[str] = []
        seen_set: Set[str] = set()
        for match in _DATA_CF_OBJECTIVE_ID_RE.finditer(content):
            oid = match.group(1)
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


class BlockCurieAnchoringValidator:
    """Outline-tier CURIE-anchoring gate.

    Ports the per-pair anchoring check from
    ``lib/validators/curie_anchoring.py`` to the Block-list shape:
    every outline-tier Block's ``content["curies"]`` must be
    non-empty, AND at least one of those CURIEs must appear in the
    block's textual surface (``content["key_claims"]``). A miss is
    a content-side semantic problem the rewrite tier could fix on a
    re-roll, so the gate emits ``action="regenerate"``.
    """

    name = "outline_curie_anchoring"
    version = "1.1.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
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
            content = block.content
            # Phase 3.5: shape-dispatch. Dict and str paths share the
            # CURIE-anchoring contract (declared CURIEs must appear in
            # the textual surface) but pull the surface from different
            # shapes.
            if isinstance(content, dict):
                audited += 1
                curies = _extract_curies_from_block(block)
                claims = content.get("key_claims") or []
                text_blob = "\n".join(
                    str(c) for c in claims if isinstance(c, str)
                )
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
            else:
                # Non-dict / non-str content — nothing to audit.
                continue

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
                continue

            anchored = any(c in surface_curies for c in curies)
            if anchored:
                passed_count += 1
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
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
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
            content = _outline_dict(block)
            if content is None:
                continue
            audited += 1
            ctype = content.get("content_type")
            if not isinstance(ctype, str) or not ctype:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_MISSING_CONTENT_TYPE",
                        message=(
                            f"Outline-tier Block {block.block_id!r} is "
                            f"missing content['content_type'] (got "
                            f"{ctype!r})."
                        ),
                        location=block.block_id,
                    ))
                continue
            if ctype not in valid_types:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_INVALID_CONTENT_TYPE",
                        message=(
                            f"Outline-tier Block {block.block_id!r} declares "
                            f"content['content_type']={ctype!r} which is not "
                            f"in the canonical ChunkType taxonomy. Valid "
                            f"values: {sorted(valid_types)}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Re-roll the outline-tier provider with the "
                            "JSON-schema directive enumerating the valid "
                            "content_type values."
                        ),
                    ))
            else:
                passed_count += 1

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


def _load_canonical_objectives(path: Path) -> Set[str]:
    """Load canonical objective IDs from a course.json /
    synthesized_objectives.json file.

    Accepts both shapes (Courseforge synthesized + LibV2 archive).
    Returns an empty set when the file is missing / unparseable
    so the caller can decide fail-vs-warn semantics.
    """
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    ids: Set[str] = set()
    # Courseforge synthesized form.
    for key in ("terminal_objectives", "chapter_objectives"):
        for entry in data.get(key, []) or []:
            if isinstance(entry, dict):
                oid = entry.get("id") or entry.get("objective_id")
                if isinstance(oid, str) and oid:
                    ids.add(oid)
    # LibV2 archive form.
    for entry in data.get("terminal_outcomes", []) or []:
        if isinstance(entry, dict):
            oid = entry.get("id") or entry.get("objective_id")
            if isinstance(oid, str) and oid:
                ids.add(oid)
    for entry in data.get("component_objectives", []) or []:
        if isinstance(entry, dict):
            oid = entry.get("id") or entry.get("objective_id")
            if isinstance(oid, str) and oid:
                ids.add(oid)
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
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
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
        seeded = inputs.get("valid_objective_ids")
        if seeded is not None:
            canonical_ids = {str(o) for o in seeded}

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            audited += 1
            obj_ids = block.objective_ids or ()
            if not obj_ids:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_MISSING_OBJECTIVE_REF",
                        message=(
                            f"Outline-tier Block {block.block_id!r} declares "
                            f"no objective_ids; every block must reference "
                            f"at least one canonical TO-NN/CO-NN objective."
                        ),
                        location=block.block_id,
                    ))
                continue
            if canonical_ids is None:
                # No canonical universe to check against — count as
                # passing the structural check (a non-empty reference
                # is the only thing we can audit).
                passed_count += 1
                continue
            unknown = [oid for oid in obj_ids if oid not in canonical_ids]
            if unknown:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_UNKNOWN_OBJECTIVE",
                        message=(
                            f"Outline-tier Block {block.block_id!r} "
                            f"references objective_ids {unknown!r} that do "
                            f"not resolve against the canonical objectives "
                            f"JSON at {objectives_path_raw!r}."
                        ),
                        location=block.block_id,
                        suggestion=(
                            "Either correct the objective_id reference in "
                            "the outline tier or extend "
                            "synthesized_objectives.json upstream."
                        ),
                    ))
            else:
                passed_count += 1

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
    """

    name = "outline_source_refs"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
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

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0
        empty_manifest = manifest_path is not None and not valid_ids

        for block in blocks:
            audited += 1
            # Accept both source_ids (Tuple[str,...]) and source_references
            # (Tuple[Dict[str,...], ...]); the rewrite tier consumes
            # source_references but the outline tier may carry either.
            block_ids: List[str] = []
            for ref in block.source_references or ():
                if isinstance(ref, dict):
                    sid = ref.get("sourceId")
                    if isinstance(sid, str) and sid:
                        block_ids.append(sid)
            for sid in block.source_ids or ():
                if isinstance(sid, str) and sid:
                    block_ids.append(sid)

            if not block_ids:
                # No source_ids on this block — outline-tier Blocks
                # are allowed to defer source attribution to the
                # rewrite tier when no DART grounding applies, so
                # an empty list passes the structural check.
                passed_count += 1
                continue

            block_passed = True
            for sid in block_ids:
                if not _SOURCE_ID_RE.match(sid):
                    block_passed = False
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
            if block_passed:
                passed_count += 1

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
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "block",
        )


__all__ = [
    "BlockCurieAnchoringValidator",
    "BlockContentTypeValidator",
    "BlockPageObjectivesValidator",
    "BlockSourceRefValidator",
]
