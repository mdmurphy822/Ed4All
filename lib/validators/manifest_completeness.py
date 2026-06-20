"""ManifestCompletenessValidator — W3 per-block synthesis-manifest RESOLUTION gate.

W3 lands a per-block provenance manifest: every synthesized Courseforge block
emits one ``block_synthesis_manifest.jsonl`` line carrying the chunk(s) + span(s)
it drew from, the template/slot identity, the model+params (from the Touch
chain), and the chunk-grounded ``concept_tags``. This validator asserts that
manifest is **complete and well-formed** — a structural completeness + existence
check, the exact failure family ``lib/validators/source_refs.py`` already catches,
extended to the manifest object.

RESOLUTION scope ONLY (the W3↔W4 boundary, load-bearing):

- W3 (this validator): the manifest's ``source_chunk_ids`` are non-empty (for
  substantive block types), each id RESOLVES against the DART chunk universe, a
  ``template_id`` is present, ``concept_tags`` is present, and the per-tier
  ``decision_capture_id`` is non-empty. A block can PASS W3 with fabricated prose
  if its declared ids happen to resolve — that is acceptable + intentional: W3
  guarantees the manifest is complete + well-formed, NOT that the prose is true.
- W4 (separate, owns NLI): the entailment gate verifies the block's prose is
  entailed by the cited chunk texts. **W3 does NOT run NLI, does NOT load any
  classifier, and makes NO truth claim about prose.** It is the cheap structural
  precondition that runs first.

Models on ``PageSourceRefValidator`` (``lib/validators/source_refs.py``) +
``ObjectiveSourceRefValidator`` (``lib/validators/objective_source_refs.py``); it
reuses the latter's ``_load_dart_chunks_universe`` chunk-universe loader so the
resolution universe is computed identically (chunk ``id`` ∪
``source.source_references[].sourceId``).

Schema: ``schemas/knowledge/block_synthesis_manifest.schema.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.objective_source_refs import _load_dart_chunks_universe

logger = logging.getLogger(__name__)


# §1.4 substantive / scaffolding split. The non-empty ``source_chunk_ids`` +
# ``concept_tags`` assertions apply only to SUBSTANTIVE block types. The
# scaffolding set mirrors the skip set the existing grounding validators use
# (``rewrite_source_grounding`` / ``claim_support`` skip chrome / recap /
# objective): ``objective``'s provenance is the LO synthesis path (W4
# ``objective_entailment``), not the block grounding path.
SUBSTANTIVE_BLOCK_TYPES: frozenset = frozenset(
    {
        "concept",
        "example",
        "explanation",
        "misconception",
        "summary_takeaway",
        "assessment_item",
        "self_check_question",
        "activity",
        "flip_card_grid",
        "callout",
        "reflection_prompt",
        "discussion_prompt",
        "prereq_set",
        # Wave-2 block-variety additions — all content-bearing and
        # source-grounded (their routing requires ``source_ref``), so
        # they belong in the substantive bucket alongside concept /
        # example / explanation.
        "scenario",
        "problem",
        "vocab_card",
        "formula",
        "checklist",
    }
)
SCAFFOLDING_BLOCK_TYPES: frozenset = frozenset({"objective", "chrome", "recap"})


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    total_blocks: int,
    complete_count: int,
    ungrounded_count: int,
    unresolved_id_count: int,
    no_concept_tag_count: int,
    resolution_universe_size: int,
) -> None:
    """Emit one ``manifest_completeness_check`` decision per validate() call.

    Rationale interpolates dynamic signals (block counts, ungrounded /
    unresolved / no-concept-tag counts, resolution-universe size) per the
    decision-capture replay contract — no static boilerplate.
    """
    if capture is None:
        return
    decision = (
        f"manifest_complete:{complete_count}/{total_blocks}"
        f"_grounded_fail:{ungrounded_count}"
        f"_unresolved:{unresolved_id_count}"
        if passed
        else f"failed:{code or 'unknown'}"
    )
    rationale = (
        f"Audited {total_blocks} manifest block(s) against "
        f"{resolution_universe_size} resolvable DART chunk id(s); "
        f"{complete_count} complete, {ungrounded_count} ungrounded "
        f"(empty source_chunk_ids on substantive types), "
        f"{unresolved_id_count} unresolved id(s), "
        f"{no_concept_tag_count} missing concept_tags. "
        f"RESOLUTION-only check (W3); failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="manifest_completeness_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "manifest_completeness_check: %s",
            exc,
        )


class ManifestCompletenessValidator:
    """Assert every synthesized block carries a complete, resolvable manifest.

    Inputs (``Dict[str, Any]``):
        manifest_path: path to ``block_synthesis_manifest.jsonl`` (the §1.3a
            sidecar). The single source of truth the gate consumes.
        dart_chunks_manifest_path: path to the DART chunkset ``manifest.json``
            (sibling ``chunks.jsonl`` is the resolution universe; reuses
            ``ObjectiveSourceRefValidator._load_dart_chunks_universe``).
        valid_source_ids: optional pre-seeded id set (test seam, mirrors the
            existing validators). Overrides ``dart_chunks_manifest_path``.
        content_development_dir: optional path to ``03_content_development/``;
            when it exists + is populated but ``manifest_path`` is missing, the
            anti-silent-degradation guard fires ``MANIFEST_FILE_MISSING``.
        require_concept_tags: bool (default ``False``) — when ``False`` an empty
            ``concept_tags`` on a substantive block is warning-severity; when
            ``True`` it is critical. Ships ``False`` (warning) day-1, flips to
            critical after a shaded run.
        block_routing_path: optional path to ``block_routing.yaml`` to source
            the per-block-type skip set; falls back to the hardcoded §1.4 split.
        gate_id, decision_capture / capture: standard.
    """

    name = "manifest_completeness"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "manifest_completeness")
        capture = inputs.get("decision_capture")
        if capture is None:
            capture = inputs.get("capture")
        require_concept_tags = bool(inputs.get("require_concept_tags", False))

        issues: List[GateIssue] = []

        substantive, scaffolding = self._resolve_skip_set(inputs)

        manifest_path = inputs.get("manifest_path")
        manifest_lines, manifest_present = self._load_manifest(
            manifest_path, issues
        )

        # --- Anti-silent-degradation guard: missing manifest but blocks exist.
        if not manifest_present:
            blocks_exist = self._content_development_populated(inputs)
            if blocks_exist:
                issues.append(GateIssue(
                    severity="critical",
                    code="MANIFEST_FILE_MISSING",
                    message=(
                        "block_synthesis_manifest.jsonl is absent/empty but "
                        "content_generation produced blocks (03_content_"
                        "development/ is populated). A missing manifest on a "
                        "real run is a hard failure, never a silent pass."
                    ),
                    location=str(manifest_path) if manifest_path else None,
                    suggestion=(
                        "Re-run content_generation so _generate_course_content "
                        "writes the per-block synthesis manifest sidecar."
                    ),
                ))
                _emit_decision(
                    capture,
                    passed=False,
                    code="MANIFEST_FILE_MISSING",
                    total_blocks=0,
                    complete_count=0,
                    ungrounded_count=0,
                    unresolved_id_count=0,
                    no_concept_tag_count=0,
                    resolution_universe_size=0,
                )
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    score=0.0,
                    issues=issues,
                    action="block",
                )
            # Graceful pass: no manifest AND no blocks (dry-run / pre-content).
            _emit_decision(
                capture,
                passed=True,
                code=None,
                total_blocks=0,
                complete_count=0,
                ungrounded_count=0,
                unresolved_id_count=0,
                no_concept_tag_count=0,
                resolution_universe_size=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=issues,
            )

        valid_ids = self._resolve_valid_ids(inputs)

        # --- Empty-resolution-universe trap (mirror SOURCE_REFS_MANIFEST_MISSING).
        manifest_declares_ids = any(
            isinstance(line, dict) and line.get("source_chunk_ids")
            for line in manifest_lines
        )
        universe_explicitly_seeded = inputs.get("valid_source_ids") is not None
        if (
            not universe_explicitly_seeded
            and not valid_ids
            and manifest_declares_ids
        ):
            issues.append(GateIssue(
                severity="critical",
                code="MANIFEST_RESOLUTION_UNIVERSE_EMPTY",
                message=(
                    "The manifest declares source_chunk_ids but the DART "
                    "chunkset resolution universe is empty (manifest.json "
                    "missing, sibling chunks.jsonl unreadable, or the chunking "
                    "phase didn't run). Without a resolvable universe the gate "
                    "would vacuously accept any declared chunk id."
                ),
                location=str(inputs.get("dart_chunks_manifest_path") or ""),
                suggestion=(
                    "Re-run the chunking phase to regenerate dart_chunks/"
                    "chunks.jsonl, or pass valid_source_ids explicitly when "
                    "the chunkset is intentionally absent."
                ),
            ))
            _emit_decision(
                capture,
                passed=False,
                code="MANIFEST_RESOLUTION_UNIVERSE_EMPTY",
                total_blocks=len(manifest_lines),
                complete_count=0,
                ungrounded_count=0,
                unresolved_id_count=0,
                no_concept_tag_count=0,
                resolution_universe_size=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                score=0.0,
                issues=issues,
                action="block",
            )

        schema = self._load_schema()

        total = len(manifest_lines)
        complete_count = 0
        ungrounded_count = 0
        unresolved_id_count = 0
        no_concept_tag_count = 0
        any_regenerate = False

        for idx, line in enumerate(manifest_lines):
            line_issues, signals = self._check_line(
                line,
                idx=idx,
                schema=schema,
                valid_ids=valid_ids,
                substantive=substantive,
                require_concept_tags=require_concept_tags,
            )
            issues.extend(line_issues)
            if signals.get("ungrounded"):
                ungrounded_count += 1
            if signals.get("unresolved_ids"):
                unresolved_id_count += signals["unresolved_ids"]
            if signals.get("no_concept_tags"):
                no_concept_tag_count += 1
            if signals.get("regenerate"):
                any_regenerate = True
            if not any(i.severity == "critical" for i in line_issues):
                complete_count += 1

        critical = [i for i in issues if i.severity == "critical"]
        passed = len(critical) == 0
        score = (complete_count / total) if total else 1.0

        action: Optional[str]
        if passed:
            action = None
        elif any_regenerate:
            action = "regenerate"
        else:
            action = "block"

        failure_code = critical[0].code if critical else None
        _emit_decision(
            capture,
            passed=passed,
            code=failure_code,
            total_blocks=total,
            complete_count=complete_count,
            ungrounded_count=ungrounded_count,
            unresolved_id_count=unresolved_id_count,
            no_concept_tag_count=no_concept_tag_count,
            resolution_universe_size=len(valid_ids),
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    # ------------------------------------------------------------------ #
    # Per-line checks (RESOLUTION ONLY)
    # ------------------------------------------------------------------ #

    def _check_line(
        self,
        line: Any,
        *,
        idx: int,
        schema: Optional[Dict[str, Any]],
        valid_ids: Set[str],
        substantive: frozenset,
        require_concept_tags: bool,
    ) -> tuple[List[GateIssue], Dict[str, Any]]:
        issues: List[GateIssue] = []
        signals: Dict[str, Any] = {}

        if not isinstance(line, dict):
            issues.append(GateIssue(
                severity="critical",
                code="MANIFEST_SCHEMA_INVALID",
                message=f"Manifest line {idx} is not a JSON object.",
                location=f"line {idx}",
            ))
            return issues, signals

        block_id = line.get("block_id") or f"<line {idx}>"

        # 1. Schema validity.
        if schema is not None:
            schema_err = self._schema_error(line, schema)
            if schema_err is not None:
                issues.append(GateIssue(
                    severity="critical",
                    code="MANIFEST_SCHEMA_INVALID",
                    message=(
                        f"Manifest line for block {block_id!r} failed schema "
                        f"validation: {schema_err}"
                    ),
                    location=str(block_id),
                    suggestion=(
                        "Conform to schemas/knowledge/"
                        "block_synthesis_manifest.schema.json."
                    ),
                ))
                # A schema-invalid line can't be trusted for the field checks.
                return issues, signals

        # 2. Required fields present.
        block_type = line.get("block_type")
        for field in ("block_id", "block_type", "page_id"):
            if not line.get(field):
                issues.append(GateIssue(
                    severity="critical",
                    code="MANIFEST_INCOMPLETE",
                    message=(
                        f"Manifest line for block {block_id!r} is missing "
                        f"required field {field!r}."
                    ),
                    location=str(block_id),
                ))
        tiers = (line.get("synthesis") or {}).get("tiers")
        if not isinstance(tiers, list) or not tiers:
            issues.append(GateIssue(
                severity="critical",
                code="MANIFEST_INCOMPLETE",
                message=(
                    f"Manifest line for block {block_id!r} is missing "
                    "synthesis.tiers[] (>=1 required)."
                ),
                location=str(block_id),
            ))
            tiers = []
        if "concept_tags" not in line:
            issues.append(GateIssue(
                severity="critical",
                code="MANIFEST_INCOMPLETE",
                message=(
                    f"Manifest line for block {block_id!r} is missing the "
                    "concept_tags field (may be empty, but must be present)."
                ),
                location=str(block_id),
            ))

        is_substantive = block_type in substantive
        source_chunk_ids = line.get("source_chunk_ids") or []
        if not isinstance(source_chunk_ids, list):
            source_chunk_ids = []

        # 3. source_chunk_ids non-empty for substantive block types.
        if is_substantive and not source_chunk_ids:
            signals["ungrounded"] = True
            signals["regenerate"] = True
            issues.append(GateIssue(
                severity="critical",
                code="MANIFEST_UNGROUNDED",
                message=(
                    f"Substantive block {block_id!r} (type {block_type!r}) has "
                    "an empty source_chunk_ids — no DART chunk grounds it."
                ),
                location=str(block_id),
                suggestion="regenerate",
            ))

        # 4. Every source_chunk_ids[] id RESOLVES against the chunk universe.
        if valid_ids:
            unresolved = 0
            for cid in source_chunk_ids:
                if isinstance(cid, str) and cid and cid not in valid_ids:
                    unresolved += 1
                    signals["regenerate"] = True
                    issues.append(GateIssue(
                        severity="critical",
                        code="MANIFEST_SOURCE_ID_UNRESOLVED",
                        message=(
                            f"Block {block_id!r} declares source_chunk_id "
                            f"{cid!r}, which does not resolve against the DART "
                            "chunk universe."
                        ),
                        location=str(block_id),
                        suggestion="regenerate",
                    ))
            if unresolved:
                signals["unresolved_ids"] = unresolved

        # 5. concept_tags non-empty for substantive block types.
        concept_tags = line.get("concept_tags") or []
        if is_substantive and not concept_tags:
            signals["no_concept_tags"] = True
            severity = "critical" if require_concept_tags else "warning"
            issues.append(GateIssue(
                severity=severity,
                code="MANIFEST_NO_CONCEPT_TAGS",
                message=(
                    f"Substantive block {block_id!r} (type {block_type!r}) "
                    "carries no concept_tags."
                ),
                location=str(block_id),
            ))

        # 6. template_id present + non-empty.
        if not line.get("template_id"):
            issues.append(GateIssue(
                severity="critical",
                code="MANIFEST_NO_TEMPLATE",
                message=(
                    f"Block {block_id!r} manifest is missing a non-empty "
                    "template_id."
                ),
                location=str(block_id),
            ))

        # 7. synthesis.tiers[].decision_capture_id non-empty (warning).
        for tier in tiers:
            if isinstance(tier, dict) and not tier.get("decision_capture_id"):
                issues.append(GateIssue(
                    severity="warning",
                    code="MANIFEST_NO_CAPTURE_LINK",
                    message=(
                        f"Block {block_id!r} tier {tier.get('tier')!r} carries "
                        "an empty decision_capture_id (Touch invariant broken "
                        "in the manifest projection)."
                    ),
                    location=str(block_id),
                ))

        return issues, signals

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_manifest(
        self, manifest_path: Any, issues: List[GateIssue]
    ) -> tuple[List[Any], bool]:
        """Load the manifest JSONL. Returns (lines, present).

        ``present`` is True only when the file exists AND carries >=1
        non-blank line (an empty file counts as absent for the guard).
        """
        if not manifest_path:
            return [], False
        path = Path(manifest_path)
        if not path.exists():
            return [], False
        lines: List[Any] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        lines.append(json.loads(raw))
                    except ValueError as exc:
                        issues.append(GateIssue(
                            severity="critical",
                            code="MANIFEST_SCHEMA_INVALID",
                            message=(
                                f"Manifest line {line_no} failed to parse as "
                                f"JSON: {exc}"
                            ),
                            location=f"line {line_no}",
                        ))
        except OSError as exc:
            logger.warning(
                "ManifestCompletenessValidator: failed to read %s: %s",
                path, exc,
            )
            return [], False
        # A file that parsed but yielded zero lines AND zero issues is empty.
        present = bool(lines) or any(
            i.code == "MANIFEST_SCHEMA_INVALID" for i in issues
        )
        return lines, present

    def _content_development_populated(self, inputs: Dict[str, Any]) -> bool:
        """True when 03_content_development/ exists + carries content files.

        Detects the "blocks were produced but the manifest is missing" trap.
        Accepts an explicit ``content_development_dir`` input; falls back to
        deriving it from ``manifest_path`` (the sidecar lives inside that dir).
        """
        cd_dir = inputs.get("content_development_dir")
        if not cd_dir:
            manifest_path = inputs.get("manifest_path")
            if manifest_path:
                cd_dir = Path(manifest_path).parent
        if not cd_dir:
            return False
        path = Path(cd_dir)
        if not path.exists() or not path.is_dir():
            return False
        # Any week_* subdir or *.html under the dir means content was emitted.
        for child in path.iterdir():
            if child.is_dir() and child.name.startswith("week"):
                return True
            if child.is_file() and child.suffix == ".html":
                return True
        # Recurse one level for week_*/*.html.
        for html in path.glob("*/*.html"):
            return True
        return False

    def _resolve_valid_ids(self, inputs: Dict[str, Any]) -> Set[str]:
        pre = inputs.get("valid_source_ids")
        if pre is not None:
            return {str(s) for s in pre}
        manifest_path = inputs.get("dart_chunks_manifest_path")
        if not manifest_path:
            return set()
        return _load_dart_chunks_universe(Path(manifest_path))

    def _resolve_skip_set(
        self, inputs: Dict[str, Any]
    ) -> tuple[frozenset, frozenset]:
        """Resolve (substantive, scaffolding) block-type sets.

        Falls back to the hardcoded §1.4 split. ``block_routing_path`` is
        accepted as a future seam, but the validator's skip-set decision is
        the hardcoded split today (the per-type ``validators`` matrix wiring is
        coupled to the deferred config/workflows.yaml gate wiring).
        """
        return SUBSTANTIVE_BLOCK_TYPES, SCAFFOLDING_BLOCK_TYPES

    _SCHEMA_CACHE: Optional[Dict[str, Any]] = None

    def _load_schema(self) -> Optional[Dict[str, Any]]:
        if ManifestCompletenessValidator._SCHEMA_CACHE is not None:
            return ManifestCompletenessValidator._SCHEMA_CACHE
        # repo_root / schemas / knowledge / block_synthesis_manifest.schema.json
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "schemas"
            / "knowledge"
            / "block_synthesis_manifest.schema.json"
        )
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(
                "ManifestCompletenessValidator: could not load schema at %s "
                "(%s); skipping schema validity check.",
                schema_path, exc,
            )
            return None
        ManifestCompletenessValidator._SCHEMA_CACHE = schema
        return schema

    @staticmethod
    def _schema_error(
        line: Dict[str, Any], schema: Dict[str, Any]
    ) -> Optional[str]:
        """Validate one manifest line against the schema; return error str or None.

        Uses ``jsonschema`` when available; degrades to a minimal required-key
        check when the dependency is absent (so the gate never hard-crashes on
        a CPU-only box without the validation extra).
        """
        try:
            import jsonschema  # type: ignore
        except ImportError:
            for key in schema.get("required", []):
                if key not in line:
                    return f"missing required key {key!r} (jsonschema absent)"
            return None
        try:
            jsonschema.validate(instance=line, schema=schema)
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            return exc.message
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return None
