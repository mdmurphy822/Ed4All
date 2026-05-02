"""Phase 3 inter-tier gate adapters (Subtask 50).

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

Phase 3.5 will extend these adapters with shape discrimination so
they also handle rewrite-tier blocks where ``block.content`` is an
HTML string. The dict-only path in this module is intentional: the
outline tier is the only consumer right now, and the rewrite-tier
extension lands as a follow-up wave with its own tests.
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
    Blocks carry an HTML string. The Phase-3 adapters only audit the
    outline tier, so any non-dict content is silently skipped (the
    Block is treated as "not auditable by this gate").
    """
    content = block.content
    if isinstance(content, dict):
        return content
    return None


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

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0

        for block in blocks:
            content = _outline_dict(block)
            if content is None:
                continue
            audited += 1
            curies_raw = content.get("curies") or []
            curies = [c for c in curies_raw if isinstance(c, str) and c]
            if not curies:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_MISSING_CURIES",
                        message=(
                            f"Outline-tier Block {block.block_id!r} carries "
                            f"an empty content['curies'] list. Phase 3 outline "
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
            # Anchoring: at least one declared CURIE must appear in
            # the block's textual surface (key_claims is the canonical
            # surface for outline-tier text per blocks.py:223-291).
            claims = content.get("key_claims") or []
            text_blob = "\n".join(
                str(c) for c in claims if isinstance(c, str)
            )
            extracted = _extract_curies(text_blob)
            anchored = any(c in extracted for c in curies)
            if anchored:
                passed_count += 1
            else:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(GateIssue(
                        severity="critical",
                        code="OUTLINE_BLOCK_CURIE_NOT_ANCHORED",
                        message=(
                            f"Outline-tier Block {block.block_id!r} declares "
                            f"CURIEs {curies!r} but none of them appear in "
                            f"content['key_claims']. The rewrite tier can't "
                            f"surface a CURIE that isn't anchored upstream."
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
