"""Phase 4 Wave N1 — Category B SHACL outline validator.

Subtask 10 of ``plans/phase4_statistical_tier_detailed.md``. Wires the
canonical Phase-2 ``cfshapes:BlockShape`` + ``cfshapes:TouchShape`` (and
their cohort in ``schemas/context/courseforge_v1.shacl.ttl``) into a
``Validator``-protocol-compatible class so the workflow runner can fire
the SHACL graph against a Block-derived JSON-LD payload list at the
``inter_tier_validation`` and ``post_rewrite_validation`` seams.

Inputs (``inputs`` dict, mirroring the Phase 3.5
``inter_tier_gates`` adapters' shape):

    blocks: List[Dict[str, Any]]
        A list of Block-derived JSON-LD entry dicts (typically the
        output of ``Block.to_jsonld_entry()`` plus the ``@type``
        alias the SHACL shapes target). Preferred shape — the workflow
        runner emits this directly post-Phase-3.5 via
        ``_run_inter_tier_validation`` /
        ``_run_post_rewrite_validation``.

    blocks_path: Optional[str | Path]
        Alternative shape for stand-alone callers. JSONL of the same
        per-line entries (``Block.to_jsonld_entry()`` shape) that
        ``_run_post_rewrite_validation`` writes for downstream
        introspection. ``.json`` files (top-level list or
        ``{"blocks": [...]}`` envelope) are also accepted.

Pipeline parity with ``PageObjectivesShaclValidator`` (the §9 PoC
sibling at ``lib/validators/shacl_runner.py:473``):

    Block-derived JSON-LD payload(s)
        └── jsonld_payloads_to_graph (apply Wave 62 @context)
            └── rdflib.Graph
                └── run_shacl(courseforge_v1.shacl.ttl, graph)
                    └── ShaclViolation(...) per sh:ValidationResult
                        └── GateIssue per ShaclViolation.to_gate_issue()

Severity routing follows ``shacl_runner._SEVERITY_MAP``:

    sh:Violation -> "critical" -> GateResult(action="block")
    sh:Warning   -> "warning"  -> GateResult(action="regenerate")
    sh:Info      -> "info"     -> GateResult (no router action)

The dual mapping reflects Phase 3 §A: a structural SHACL violation
(missing required predicate, malformed block_id, escalation_marker
out of enum) is the kind of miss the rewrite tier cannot fix on a
re-roll, so it short-circuits to ``"block"``. A SHACL warning maps to
``"regenerate"`` because it's the same shape contract surface that the
outline tier could reasonably re-roll past on a second draft.

Shape-discrimination: dict entries are validated as-is; HTML strings
(rewrite-tier ``Block.content``) carry the JSON-LD ``blocks[]``
projection inside an embedded ``<script type="application/ld+json">``
block (Phase 2 emit at ``Courseforge/scripts/generate_course.py:2090``
when ``COURSEFORGE_EMIT_BLOCKS=true``); the validator scrapes those
blocks via the existing ``shacl_runner._extract_jsonld_blocks`` helper.

Graceful degradation: missing pyld/pyshacl/rdflib extras emit a single
warning issue with ``passed=True`` (mirroring Phase 4 Subtask 8's
``TRAINFORGE_REQUIRE_EMBEDDINGS`` opt-out pattern for embedding extras).
The Phase 4 PoC severity is intentionally `warning` so the gate never
blocks a workflow on a dev-extras gap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.validators.shacl_runner import (
    ShaclDepsMissing,
    _ensure_deps,
    _extract_jsonld_blocks,
    jsonld_payloads_to_graph,
    run_shacl,
)

logger = logging.getLogger(__name__)


#: Canonical multi-shape SHACL file the validator targets. Carries
#: BlockShape (sh:targetClass ed4all:Block) + TouchShape (sh:targetClass
#: ed4all:Touch) plus the legacy CourseModule / LearningObjective /
#: Section / TargetedConcept / Misconception / BloomDistribution / Chunk
#: / TypedEdge shapes — the graph fires every shape whose target class
#: matches a node in the data graph, so adding new block-side payloads
#: doesn't require re-pointing this constant.
DEFAULT_SHAPES_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "context"
    / "courseforge_v1.shacl.ttl"
)


def _coerce_block_payloads(
    inputs: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Optional[GateIssue]]:
    """Pull a list of Block JSON-LD payload dicts out of ``inputs``.

    Resolution priority (high → low):

    1. ``inputs["blocks"]`` — direct list of dicts (preferred — what
       the workflow runner passes post-Phase-3.5).
    2. ``inputs["blocks_path"]`` — JSONL or JSON file path. JSONL is
       one entry per line; JSON is either a top-level list or a
       ``{"blocks": [...]}`` envelope.

    Returns ``(payloads, error_issue)``. ``error_issue`` is non-None
    when the input shape is wrong; the caller wraps it into a
    ``passed=False`` ``GateResult`` and skips the SHACL run.
    """
    raw_blocks = inputs.get("blocks")
    if raw_blocks is not None:
        if not isinstance(raw_blocks, list):
            return [], GateIssue(
                severity="critical",
                code="INVALID_BLOCKS_INPUT",
                message=(
                    "inputs['blocks'] must be a list of Block JSON-LD "
                    f"entry dicts; got {type(raw_blocks).__name__}."
                ),
            )
        # Allow strings inside the list to carry HTML payloads
        # (rewrite-tier Block.content is sometimes serialised as
        # the page HTML envelope rather than the dict entry).
        payloads: List[Dict[str, Any]] = []
        for entry in raw_blocks:
            if isinstance(entry, dict):
                payloads.append(entry)
            elif isinstance(entry, str):
                payloads.extend(_extract_jsonld_blocks(entry))
        return payloads, None

    blocks_path_raw = inputs.get("blocks_path")
    if not blocks_path_raw:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "Either inputs['blocks'] (list of Block JSON-LD dicts) "
                "or inputs['blocks_path'] (JSONL/JSON file path) is "
                "required for CourseforgeOutlineShaclValidator."
            ),
        )

    blocks_path = Path(blocks_path_raw)
    if not blocks_path.exists():
        return [], GateIssue(
            severity="critical",
            code="BLOCKS_PATH_NOT_FOUND",
            message=f"blocks_path does not exist: {blocks_path}",
        )

    try:
        text = blocks_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], GateIssue(
            severity="critical",
            code="BLOCKS_PATH_READ_ERROR",
            message=f"Failed to read {blocks_path}: {exc}",
        )

    payloads = []
    try:
        if blocks_path.suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                payloads.append(json.loads(line))
        else:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                payloads = parsed
            elif isinstance(parsed, dict):
                inner = parsed.get("blocks")
                if isinstance(inner, list):
                    payloads = inner
    except json.JSONDecodeError as exc:
        return [], GateIssue(
            severity="critical",
            code="BLOCKS_PATH_PARSE_ERROR",
            message=f"Failed to parse {blocks_path}: {exc}",
        )

    return [p for p in payloads if isinstance(p, dict)], None


def _annotate_block_type(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Inject ``@type: "Block"`` when the payload carries a ``blockId``
    but no explicit ``@type``.

    The Phase-2 ``Block.to_jsonld_entry()`` shape only emits ``@type``
    on entries the SHACL shapes target via ``sh:targetClass`` (e.g.
    ``CourseModule`` / ``LearningObjective`` / ``Misconception``). For
    the new Phase-2 ``blocks[]`` array, the ``@type: "Block"`` alias
    has to be added by the consumer because the JSON-LD context maps
    ``ed4all:hasBlock`` / ``ed4all:Block`` but doesn't auto-promote
    set members. Mirrors the test fixture at
    ``Courseforge/scripts/tests/test_generate_course_shacl_validation.py:421-426``.
    """
    if not isinstance(payload, dict):
        return payload
    if "@type" in payload:
        return payload
    if "blockId" in payload:
        annotated = dict(payload)
        annotated["@type"] = "Block"
        return annotated
    return payload


def _decide_action(critical_count: int, warning_count: int) -> Optional[str]:
    """Map (critical, warning) violation counts onto the router's action enum.

    Aligns with the Phase 3 ``GateResult.action`` contract:
      - critical -> "block"   (structural miss; rewrite cannot fix)
      - warning  -> "regenerate" (rewrite-tier could re-roll past it)
      - neither  -> None      (let the router default to "pass")
    """
    if critical_count > 0:
        return "block"
    if warning_count > 0:
        return "regenerate"
    return None


class CourseforgeOutlineShaclValidator:
    """Phase 4 Wave N1 — SHACL gate for outline / rewrite Block payloads.

    Validator-protocol-compatible class wired into the workflow runner
    via ``inter_tier_validation::outline_shacl`` (Subtask 11) and
    ``post_rewrite_validation::rewrite_shacl`` (Subtask 12). Reuses the
    ``shacl_runner.run_shacl`` PoC machinery against the canonical
    multi-shape ``courseforge_v1.shacl.ttl`` graph so a single shape
    update propagates across both seams.

    Behavior contract (sub-plan §8 mirror):
        - SHACL deps missing -> single warning issue, ``passed=True``,
          ``action=None``. Cannot block the workflow during PoC.
        - Empty payload list -> ``passed=True``, ``action=None``, no
          issues.
        - Invalid input shape -> ``passed=False``, single critical
          issue, ``action="block"``.
        - Real critical violations -> ``passed=False``,
          ``action="block"``.
        - Real warning violations only -> ``passed=True``,
          ``action="regenerate"`` so the router re-rolls the outline.
    """

    name = "courseforge_outline_shacl"
    version = "0.1.0"  # Phase 4 Wave N1 PoC

    def __init__(
        self,
        *,
        shapes_path: Optional[Union[Path, str]] = None,
    ) -> None:
        self._shapes_path = (
            Path(shapes_path) if shapes_path is not None else DEFAULT_SHAPES_PATH
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)

        payloads, err = _coerce_block_payloads(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Empty input is a no-op pass — matches PageObjectivesShaclValidator
        # (sub-plan §8: empty corpus -> passed=True, no issues).
        if not payloads:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        # SHACL deps may not be installed in every environment (pyld /
        # pyshacl are dev-extras). Degrade gracefully so the PoC gate
        # never blocks a run on missing extras — mirrors Phase 4
        # Subtask 8's embedding-extras opt-out pattern.
        try:
            _ensure_deps()
        except ShaclDepsMissing as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="SHACL_DEPS_MISSING",
                        message=(
                            f"SHACL toolchain not importable: {exc}. "
                            "Phase 4 PoC gate skipped; install the "
                            "`shacl` extras to enable Block-shape validation."
                        ),
                    )
                ],
            )

        annotated = [_annotate_block_type(p) for p in payloads]
        graph = jsonld_payloads_to_graph(annotated)

        try:
            conforms, violations = run_shacl(self._shapes_path, graph)
        except FileNotFoundError as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="SHAPE_FILE_MISSING",
                        message=str(exc),
                    )
                ],
                action="block",
            )

        issues = [v.to_gate_issue() for v in violations]
        critical = sum(1 for i in issues if i.severity == "critical")
        warning = sum(1 for i in issues if i.severity == "warning")

        action = _decide_action(critical, warning)
        # `passed` follows the Python-validator convention: critical
        # violations fail the gate; warning-only runs still pass (the
        # router consumes ``action`` to decide whether to regenerate).
        passed = conforms or critical == 0
        score = (
            1.0
            if not violations
            else max(0.0, 1.0 - len(violations) / max(1, len(payloads)))
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


__all__ = [
    "CourseforgeOutlineShaclValidator",
    "DEFAULT_SHAPES_PATH",
]
