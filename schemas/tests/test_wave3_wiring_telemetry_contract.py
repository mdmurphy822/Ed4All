"""End-of-Wave-3 wiring + telemetry gate.

SELF-CONTAINED contract test: every fixture is inlined locally so a
future rename of a per-worker test file
(``lib/aggregators/tests/test_promotion_chain_report.py``,
``lib/governance/tests/test_calibration_gate.py``,
``Courseforge/router/tests/test_block_type_gate_matrix.py``, …) cannot
silently disable this gate.

Contracts pinned here:

* ``PromotionChainAggregator`` emit shape — exactly 9 arrows, a
  ``course_status`` from the canonical 5-value enum, a 64-char hex
  ``chain_hash``.
* Arrows 1-7 MUST carry ``source_coverage``; arrows 8-9 legitimately
  may be null (their inputs are hashed pair files / config dicts, not
  chunk-indexed artifacts).
* ``derive_course_status`` returns an enum value for all 8 cells of the
  ``(accessibility, instructional, trainable)`` table; only the
  critical cohort short-circuits to ``failed``.
* A missing per-stage report fails CLOSED (fail row + sentinel in
  ``validator_set``) rather than defaulting to pass — the load-bearing
  failure mode of any N-stage chain aggregator.
* Every ``BLOCK_TYPES`` entry has a validator-matrix row; a block type
  with no row would route through with NO validators.
* The calibration-gated severity flip reads
  ``answerability_alignment_rate`` — the "LLM judge agreed with the
  deterministic answer" signal. The sibling ``answerable_rate`` field
  means "the question was answerable in principle"; reading it instead
  would flip severity on the wrong evidence.
* ``build_source_coverage`` inserts an
  ``INTERNAL_DROP_REASON_MISSING`` bucket for unattributed drops, so
  ``dropped_count == sum(drop_reasons.values())`` holds in the emitted
  shape.
* Per-tree ``def test_*`` count ratchet plus Draft-2020-12
  ``check_schema`` cleanliness on the net-new schemas.

Test 1 uses an entirely synthetic in-process layout, so this public contract
never discovers or reads an operator corpus.
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Make the repo root importable so ``from lib...`` / ``from Courseforge...``
# resolve regardless of the pytest invocation cwd.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Block lives at Courseforge/scripts/blocks.py, which is not an
# importable package — bridge the directory onto sys.path.
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# --------------------------------------------------------------------------- #
# Per-tree test-count floors — a silent test removal trips the ratchet.
# Counts come from AST-walking each tree's ``test_*.py`` and tallying
# ``def test_`` declarations. AST counts are LOWER than
# ``pytest --collect-only`` counts (pytest expands parametrized cases at
# collection time), so the two algorithms must never be mixed across
# gates; each is monotonic on its own. The 7 trees cover every subtree
# this gate ratchets: validators, aggregators, the Trainforge eval
# surface, the broader Trainforge tree, the Courseforge router, the MCP
# tool / orchestrator surface, and the schemas tree.
# --------------------------------------------------------------------------- #

_MIN_VALIDATORS_TESTS = 523
_MIN_AGGREGATORS_TESTS = 72
_MIN_TRAINFORGE_EVAL_TESTS = 27
_MIN_TRAINFORGE_TESTS = 1658
_MIN_COURSEFORGE_ROUTER_TESTS = 162
_MIN_MCP_TESTS = 760
_MIN_SCHEMAS_TESTS = 352


# Canonical 5-value enum at ``schemas/governance/course_status.schema.json``.
_COURSE_STATUS_ENUM: frozenset = frozenset({
    "failed",
    "non_certified_archive",
    "certified_accessible",
    "certified_instructional",
    "certified_trainable",
})


# Net-new schemas covered by the cleanliness sweep in Test 8.
_W3_SCHEMA_PATHS: Tuple[Path, ...] = (
    PROJECT_ROOT / "schemas" / "library" / "chunkset_drift_report.schema.json",
    PROJECT_ROOT / "schemas" / "library" / "packaging_report.schema.json",
    PROJECT_ROOT / "schemas" / "training" / "synthesis_summary.schema.json",
    PROJECT_ROOT / "schemas" / "governance" / "promotion_chain.schema.json",
    PROJECT_ROOT / "schemas" / "aggregators" / "coverage_map.schema.json",
)


# --------------------------------------------------------------------------- #
# Fixture builders. These duplicate the helpers in
# lib/aggregators/tests/test_promotion_chain_report.py ON PURPOSE — a
# rename there must not be able to disable this gate.
# --------------------------------------------------------------------------- #


def _coverage_block(consumed: int, emitted: int) -> Dict[str, Any]:
    """Synthesise a canonical ``source_coverage`` block."""
    consumed_i = max(0, int(consumed))
    emitted_i = max(0, int(emitted))
    return {
        "consumed_count": consumed_i,
        "emitted_count": emitted_i,
        "dropped_count": max(0, consumed_i - emitted_i),
        "drop_reasons": {},
        "coverage_pct": (emitted_i / consumed_i) if consumed_i > 0 else 0.0,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _build_full_promotion_layout(tmp_path: Path) -> Dict[str, Path]:
    """Build a complete fixture layout with all 9 arrows present + passing."""
    course_dir = tmp_path / "libv2_course"
    project_dir = tmp_path / "courseforge_project"
    staging_dir = tmp_path / "staging"
    course_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Arrow 1 — staging manifest
    _write_json(
        staging_dir / "staging_manifest.json",
        {
            "run_id": "WF-W3-GATE",
            "course_name": "W3_GATE",
            "staged_files": ["a.html", "b.html"],
            "files": [{"path": "a.html"}, {"path": "b.html"}],
            "errors": None,
        },
    )
    # Arrow 2 — semantik_chunks/manifest.json
    _write_json(
        course_dir / "semantik_chunks" / "manifest.json",
        {
            "chunks_sha256": "a" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "semantik",
            "source_semantik_html_sha256": "b" * 64,
            "chunks_count": 100,
            "source_coverage": _coverage_block(100, 100),
        },
    )
    # Arrow 3 — inter-tier + post-rewrite reports + cf validation report
    _write_json(
        project_dir / "02_validation_report" / "report.json",
        {
            "schema_version": "1.0",
            "total_blocks": 50, "passed": 50, "failed": 0, "escalated": 0,
            "per_block": [],
            "source_coverage": _coverage_block(50, 50),
        },
    )
    _write_json(
        project_dir / "04_rewrite" / "02_validation_report" / "report.json",
        {
            "schema_version": "1.0",
            "total_blocks": 50, "passed": 50, "failed": 0, "escalated": 0,
            "per_block": [],
            "source_coverage": _coverage_block(50, 50),
        },
    )
    _write_json(
        project_dir / "courseforge_validation_report.json",
        {
            "schema_version": "1.1",
            "course_code": "W3_GATE",
            "run_id": "WF-W3-GATE",
            "generated_at": "2026-05-06T00:00:00Z",
            "status": "pass",
            "summary": {
                "total_gates": 0, "passed_count": 0, "failed_count": 0,
                "warning_count": 0, "skipped_count": 0,
            },
            "blocking_failures": [], "warnings": [], "per_phase": [],
            "final_promotion_decision": {
                "value": "certified_trainable",
                "rationale": "all gates passed",
                "contributing_gate_ids": [],
            },
        },
    )
    # Arrow 4 — packaging_report.json
    _write_json(
        project_dir / "05_final_package" / "packaging_report.json",
        {"source_coverage": _coverage_block(50, 50)},
    )
    # LibV2 manifest carrying imscc hash
    _write_json(
        course_dir / "manifest.json",
        {"imscc_sha256": "c" * 64},
    )
    # Arrow 5 — imscc_chunks/manifest.json
    _write_json(
        course_dir / "imscc_chunks" / "manifest.json",
        {
            "chunks_sha256": "d" * 64,
            "chunker_version": "v4",
            "chunkset_kind": "imscc",
            "source_imscc_sha256": "c" * 64,
            "chunks_count": 100,
            "source_coverage": _coverage_block(100, 100),
        },
    )
    # Arrow 6 — assessments.json
    _write_json(
        course_dir / "training_specs" / "assessments.json",
        {
            "assessment_id": "asmt-001",
            "title": "W3_GATE",
            "course_code": "W3_GATE",
            "questions": [],
            "source_coverage": _coverage_block(20, 20),
        },
    )
    # Arrow 7 — synthesis_summary.json + dataset_config.json
    _write_json(
        course_dir / "training_specs" / "synthesis_summary.json",
        {
            "schema_version": "1.0",
            "course_code": "W3_GATE",
            "provider": "local",
            "instruction_pairs_emitted": 100,
            "preference_pairs_emitted": 50,
            "source_coverage": _coverage_block(150, 150),
        },
    )
    _write_json(
        course_dir / "training_specs" / "dataset_config.json",
        {
            "statistics": {
                "promotion_ladder": {"draft": 200, "trainable": 150},
            },
        },
    )
    # Arrows 8 + 9 — model_card.json + eval_report.json
    model_dir = course_dir / "models" / "test-model-001"
    _write_json(
        model_dir / "model_card.json",
        {
            "model_id": "test-model-001",
            "provenance": {
                "instruction_pairs_hash": "e" * 64,
                "preference_pairs_hash": "f" * 64,
                "chunks_hash": "1" * 64,
                "pedagogy_graph_hash": "2" * 64,
                "concept_graph_hash": "3" * 64,
                "vocabulary_ttl_hash": "4" * 64,
                "holdout_graph_hash": "5" * 64,
            },
            "weights_sha256": "6" * 64,
        },
    )
    _write_json(
        model_dir / "eval" / "eval_report.json",
        {
            "eval_config_hash": "7" * 64,
            "status": "pass",
            "metrics": {"faithfulness": 0.85, "source_match": 0.65},
        },
    )
    return {
        "course_dir": course_dir,
        "project_dir": project_dir,
        "staging_dir": staging_dir,
    }


def _archive_passing_arrows() -> List[Dict[str, Any]]:
    """Build arrows 1-5 in a clean passing state, no critical-cohort gates."""
    from lib.aggregators.promotion_chain_report import ARROW_NAMES
    return [
        {
            "arrow_id": i,
            "name": ARROW_NAMES[i],
            "input_hash": None,
            "output_hash": None,
            "validator_set": [],
            "passed": True,
            "warnings_count": 0,
            "source_coverage": None,
            "promotion_decision": "pass",
        }
        for i in range(1, 6)
    ]


def _arrow_with_gate(
    arrow_id: int, gate_id: str, *, promotion: str = "pass",
) -> Dict[str, Any]:
    from lib.aggregators.promotion_chain_report import ARROW_NAMES
    return {
        "arrow_id": arrow_id,
        "name": ARROW_NAMES[arrow_id],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [gate_id],
        "passed": promotion in ("pass", "warn"),
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": promotion,
    }


def _make_decision_table_chain(
    *,
    accessibility_pass: bool,
    instructional_pass: bool,
    trainable_pass: bool,
) -> List[Dict[str, Any]]:
    """Compose a 9-arrow chain that exercises a row of the decision table."""
    from lib.aggregators.promotion_chain_report import ARROW_NAMES
    arrows = _archive_passing_arrows()
    # Embed accessibility gate on arrow 5 so when it fails the helper
    # short-circuits to "failed" via the critical-cohort branch.
    arrows[4] = _arrow_with_gate(
        5, "wcag_compliance",
        promotion="pass" if accessibility_pass else "fail",
    )
    instr_promotion = "pass" if instructional_pass else "fail"
    arrows.append({
        "arrow_id": 6,
        "name": ARROW_NAMES[6],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [
            "content_grounding", "oscqr_score",
            "page_objectives", "source_refs",
        ],
        "passed": instructional_pass,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": instr_promotion,
    })
    train_promotion = "pass" if trainable_pass else "fail"
    arrows.append({
        "arrow_id": 7,
        "name": ARROW_NAMES[7],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [
            "min_edge_count", "synthesis_diversity",
            "eval_gating", "family_completeness",
        ],
        "passed": trainable_pass,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": train_promotion,
    })
    arrows.append({
        "arrow_id": 8,
        "name": ARROW_NAMES[8],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [],
        "passed": True,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": "pass",
    })
    arrows.append({
        "arrow_id": 9,
        "name": ARROW_NAMES[9],
        "input_hash": None,
        "output_hash": None,
        "validator_set": [],
        "passed": True,
        "warnings_count": 0,
        "source_coverage": None,
        "promotion_decision": "pass",
    })
    return arrows


def _count_test_functions(tree: Path) -> Tuple[int, List[str]]:
    """AST-walk every ``test_*.py`` under ``tree`` and count
    ``def test_*`` declarations.

    Must stay byte-identical to the counting helper in the sibling
    wave-end gates — mixing counting algorithms breaks the monotonic
    ratchet contract across gates.
    """
    if not tree.exists():
        return 0, []
    total = 0
    seen_files: List[str] = []
    for path in sorted(tree.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            module = ast.parse(path.read_text())
        except (SyntaxError, OSError):
            continue
        seen_files.append(str(path.relative_to(PROJECT_ROOT)))
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    total += 1
    return total, seen_files


# --------------------------------------------------------------------------- #
# Test 1 — synthetic promotion-chain contract
# --------------------------------------------------------------------------- #


def test_1_full_pipeline_promotion_chain_emit_shape(tmp_path) -> None:
    """Promotion-chain master-report emit shape.

    Synthetic-build leg (always runs): a happy-path 9-arrow fixture
    layout through :class:`PromotionChainAggregator` must yield

    * exactly 9 arrows, each ``arrow_id`` matching its 1..9 index;
    * a ``course_status`` from the canonical 5-value enum in
      ``schemas/governance/course_status.schema.json``;
    * a 64-char hex SHA-256 ``chain_hash``;
    * ``promotion_decision`` / ``validator_set`` / ``source_coverage``
      on every arrow row.

    The fixture is generated entirely beneath ``tmp_path`` and never reads an
    operator corpus.
    """
    from lib.aggregators.promotion_chain_report import (
        PromotionChainAggregator,
        SCHEMA_VERSION,
    )

    layout = _build_full_promotion_layout(tmp_path)
    aggregator = PromotionChainAggregator(
        course_path=layout["course_dir"],
        project_path=layout["project_dir"],
        staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
        course_code="W3_GATE",
        run_id="WF-W3-GATE",
    )
    report = aggregator.build()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["course_code"] == "W3_GATE"
    assert report["run_id"] == "WF-W3-GATE"

    arrows = report.get("arrows") or []
    assert isinstance(arrows, list) and len(arrows) == 9, (
        f"expected exactly 9 arrows, got {len(arrows)}"
    )
    for index, arrow in enumerate(arrows, start=1):
        assert arrow["arrow_id"] == index, (
            f"arrow_id mismatch at index {index}: {arrow!r}"
        )
        assert isinstance(arrow.get("promotion_decision"), str)
        assert "validator_set" in arrow
        assert "source_coverage" in arrow

    # Top-level course_status is one of the canonical 5 enum values.
    assert report.get("course_status") in _COURSE_STATUS_ENUM, (
        f"course_status {report.get('course_status')!r} is not in the "
        f"canonical 5-value enum {sorted(_COURSE_STATUS_ENUM)}"
    )
    # Happy-path layout SHOULD certify trainable.
    assert report["course_status"] == "certified_trainable", (
        f"happy-path layout should certify trainable; got "
        f"{report['course_status']!r}. Re-check fixture builder."
    )

    chain_hash = report.get("chain_hash")
    assert isinstance(chain_hash, str), (
        f"chain_hash must be a string; got {type(chain_hash).__name__}"
    )
    assert len(chain_hash) == 64, (
        f"chain_hash must be 64-char SHA-256 hex; got len={len(chain_hash)}"
    )
    int(chain_hash, 16)  # raises ValueError if not hex

# --------------------------------------------------------------------------- #
# Test 2 — anti-silent-degradation: every arrow has coverage data
# --------------------------------------------------------------------------- #


def test_2_every_arrow_has_source_coverage_in_happy_path(tmp_path) -> None:
    """Anti-silent-degradation: every coverage-bearing arrow has coverage.

    Arrows 1-7 MUST carry a non-null ``source_coverage`` block with a
    numeric ``coverage_pct``. Arrows 8 (training_pairs → adapter) and 9
    (adapter → eval_report) MAY be null — their inputs are hashed pair
    files / config dicts, not chunk-indexed artifacts — but if a future
    wave does wire coverage there it must still be a mapping, not a
    bare int / string.

    Catches an arrow that silently stops emitting ``source_coverage``.
    """
    from lib.aggregators.promotion_chain_report import PromotionChainAggregator

    layout = _build_full_promotion_layout(tmp_path)
    aggregator = PromotionChainAggregator(
        course_path=layout["course_dir"],
        project_path=layout["project_dir"],
        staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
        course_code="W3_GATE",
        run_id="WF-W3-GATE",
    )
    report = aggregator.build()

    arrows_by_id = {a["arrow_id"]: a for a in report["arrows"]}
    for arrow_id in range(1, 8):  # 1..7 inclusive — coverage-bearing arrows
        arrow = arrows_by_id.get(arrow_id)
        assert arrow is not None, f"arrow {arrow_id} missing from report"
        coverage = arrow.get("source_coverage")
        assert coverage is not None, (
            f"arrow {arrow_id} ({arrow.get('name')}) "
            f"missing source_coverage"
        )
        assert isinstance(coverage, Mapping), (
            f"arrow {arrow_id} source_coverage is not a mapping: "
            f"{coverage!r}"
        )
        coverage_pct = coverage.get("coverage_pct")
        assert isinstance(coverage_pct, (int, float)), (
            f"arrow {arrow_id} coverage_pct is not numeric: "
            f"{coverage_pct!r} (type={type(coverage_pct).__name__})"
        )

    # Spot-check arrows 8 + 9 — null is fine, but if a future wave wires
    # coverage in those arrows the field MUST be a mapping (not e.g. a
    # bare int / string).
    for arrow_id in (8, 9):
        arrow = arrows_by_id.get(arrow_id)
        assert arrow is not None
        coverage = arrow.get("source_coverage")
        if coverage is not None:
            assert isinstance(coverage, Mapping), (
                f"arrow {arrow_id} source_coverage is non-null but not a "
                f"mapping: {coverage!r}"
            )


# --------------------------------------------------------------------------- #
# Test 3 — course_status decision-logic table is exhaustive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "accessibility_pass,instructional_pass,trainable_pass",
    [
        (False, False, False),
        (False, False, True),
        (False, True,  False),
        (False, True,  True),
        (True,  False, False),
        (True,  False, True),
        (True,  True,  False),
        (True,  True,  True),
    ],
)
def test_3_course_status_decision_yields_enum(
    accessibility_pass: bool,
    instructional_pass: bool,
    trainable_pass: bool,
) -> None:
    """The ``course_status`` decision table is exhaustive.

    Every cell of the 8-cell ``(accessibility_pass, instructional_pass,
    trainable_pass)`` table must yield one of the 5 enum values and
    never None.

    Load-bearing branches:

    * ``(True, True, True)`` → ``certified_trainable``
    * ``(True, True, False)`` → ``certified_instructional``
    * ``(True, False, *)`` → ``certified_accessible`` (instructional
      cohort fail does NOT short-circuit to failed; only the critical
      cohort does)
    * ``(False, *, *)`` → ``failed`` (accessibility is the critical
      cohort)

    Catches a silent default-to-failed where bad input yields no
    decision, and a new combination the helper doesn't handle.
    """
    from lib.governance.course_status import derive_course_status

    arrows = _make_decision_table_chain(
        accessibility_pass=accessibility_pass,
        instructional_pass=instructional_pass,
        trainable_pass=trainable_pass,
    )
    decision = derive_course_status(arrows)

    assert decision is not None, (
        f"derive_course_status returned None on combo "
        f"({accessibility_pass}, {instructional_pass}, {trainable_pass})"
    )
    assert decision in _COURSE_STATUS_ENUM, (
        f"({accessibility_pass}, {instructional_pass}, {trainable_pass}) "
        f"-> {decision!r} not in {sorted(_COURSE_STATUS_ENUM)}"
    )

    if accessibility_pass and instructional_pass and trainable_pass:
        assert decision == "certified_trainable", (
            f"all-pass combo should certify trainable; got {decision!r}"
        )
    if accessibility_pass and instructional_pass and not trainable_pass:
        assert decision == "certified_instructional", (
            f"trainable-fail-only combo should ceiling at instructional; "
            f"got {decision!r}"
        )
    if accessibility_pass and not instructional_pass and not trainable_pass:
        assert decision == "certified_accessible", (
            f"accessibility-only combo should certify accessible; got "
            f"{decision!r}"
        )
    if not accessibility_pass:
        assert decision == "failed", (
            f"accessibility hard-fail is critical; should yield failed. "
            f"Got {decision!r}"
        )


# --------------------------------------------------------------------------- #
# Test 4 — anti-silent-degradation: master aggregator integrity check
# --------------------------------------------------------------------------- #


def test_4_aggregator_fails_closed_on_missing_per_stage_report(
    tmp_path,
) -> None:
    """Master aggregator fails CLOSED on a missing per-stage report.

    Delete ``imscc_chunks/manifest.json`` from an otherwise-complete
    9-arrow layout; the aggregator must surface

    * arrow 5 ``promotion_decision == "fail"``,
    * ``"missing_stage_report"`` in arrow 5's ``validator_set``,
    * top-level ``course_status == "failed"``.

    Catches the load-bearing failure mode of any N-stage chain
    aggregator: silently defaulting to pass when a stage report is
    absent.
    """
    from lib.aggregators.promotion_chain_report import (
        MISSING_STAGE_REPORT,
        PromotionChainAggregator,
    )

    layout = _build_full_promotion_layout(tmp_path)
    deleted = layout["course_dir"] / "imscc_chunks" / "manifest.json"
    assert deleted.exists(), (
        "fixture build skipped imscc_chunks manifest; can't delete "
        "what isn't there"
    )
    deleted.unlink()

    aggregator = PromotionChainAggregator(
        course_path=layout["course_dir"],
        project_path=layout["project_dir"],
        staging_manifest_path=layout["staging_dir"] / "staging_manifest.json",
        course_code="W3_GATE",
        run_id="WF-W3-GATE",
    )
    report = aggregator.build()

    arrow_5 = next(
        (a for a in report["arrows"] if a["arrow_id"] == 5), None,
    )
    assert arrow_5 is not None, "arrow 5 missing from emit"
    assert arrow_5["promotion_decision"] == "fail", (
        f"arrow 5 should fail closed on missing imscc_chunks manifest; "
        f"got promotion_decision={arrow_5.get('promotion_decision')!r}. "
        "This is the silent-degradation regression class — a missing "
        "per-stage report MUST surface as a fail row."
    )
    assert MISSING_STAGE_REPORT in arrow_5["validator_set"], (
        f"arrow 5 validator_set must carry {MISSING_STAGE_REPORT!r}; "
        f"got {arrow_5.get('validator_set')!r}. The sentinel is the "
        "operator-visible signal that the underlying file was missing."
    )
    assert report["course_status"] == "failed", (
        f"course_status should fall through to 'failed' when a critical "
        f"cohort sentinel fires; got {report.get('course_status')!r}"
    )


# --------------------------------------------------------------------------- #
# Test 5 — per-block gate matrix completeness
# --------------------------------------------------------------------------- #


def test_5_every_block_type_has_validator_matrix_entry() -> None:
    """Per-block-type gate-matrix completeness.

    Every entry in ``Courseforge.scripts.blocks.BLOCK_TYPES`` must have
    a matching ``BlockRoutingPolicy.validators_by_block_type`` row whose
    ``fail_action`` is one of ``{regenerate, escalate, block}``.

    Catches a new block_type added to ``BLOCK_TYPES`` without a
    corresponding ``block_routing.yaml`` entry — it would silently route
    through with NO validators.
    """
    # Canonical block-types frozenset lives on the bridged
    # ``Courseforge/scripts/`` path, not an importable package.
    from blocks import BLOCK_TYPES  # noqa: WPS433
    from Courseforge.router.policy import load_block_routing_policy

    policy = load_block_routing_policy()

    missing: List[str] = []
    bad_fail_action: List[Tuple[str, Any]] = []
    bad_shape: List[str] = []
    canonical_actions = frozenset({"regenerate", "escalate", "block"})

    for block_type in sorted(BLOCK_TYPES):
        entry = policy.validators_by_block_type.get(block_type)
        if entry is None:
            missing.append(block_type)
            continue
        if "required" not in entry:
            bad_shape.append(f"{block_type}: missing 'required' field")
        if "fail_action" not in entry:
            bad_shape.append(f"{block_type}: missing 'fail_action' field")
            continue
        fail_action = entry["fail_action"]
        if fail_action not in canonical_actions:
            bad_fail_action.append((block_type, fail_action))

    assert not missing, (
        f"BLOCK_TYPES entries with no validator matrix entry: "
        f"{missing!r}. A block_type added to ``BLOCK_TYPES`` without a "
        "row in ``Courseforge/router/policy.py::DEFAULT_BLOCK_ROUTING`` "
        "(and the YAML overlay) would silently route through with NO "
        "validators — this gate fires before that ships."
    )
    assert not bad_shape, (
        f"validators_by_block_type entries with malformed shape: "
        f"{bad_shape!r}"
    )
    assert not bad_fail_action, (
        f"validators_by_block_type entries with non-canonical "
        f"fail_action: {bad_fail_action!r}. fail_action must be one of "
        f"{sorted(canonical_actions)}."
    )

    # Defence-in-depth: pin the BLOCK_TYPES cardinality so a drive-by
    # addition has to be a deliberate, audited change.
    assert len(BLOCK_TYPES) == 30, (
        f"BLOCK_TYPES drift: expected 30 canonical block types, got "
        f"{len(BLOCK_TYPES)}: {sorted(BLOCK_TYPES)!r}. If this is an "
        "intentional change, bump the assertion AND audit every "
        "block_type-aware surface (router policy, JSON-LD shape, "
        "Trainforge chunk extractor)."
    )


# --------------------------------------------------------------------------- #
# Test 6 — severity flip stays gated
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "alignment_rate,expected_apply,expected_reason",
    [
        # Below the 0.85 floor: flip held, reason recorded.
        (0.84, False, "alignment_rate_below_floor"),
        # Above the floor: flip applied; the applied payload carries no
        # reason key.
        (0.86, True, None),
    ],
)
def test_6_severity_flip_stays_gated_by_calibration_signal(
    tmp_path, alignment_rate: float, expected_apply: bool,
    expected_reason: Any,
) -> None:
    """The severity flip stays gated by the calibration signal.

    Two cases against
    :func:`lib.governance.calibration_gate.resolve_severity_flip`:
    below the floor the flip is held with
    ``ml_features.reason == "alignment_rate_below_floor"``; above it the
    flip is applied and the payload carries
    ``decision_type == "severity_flip_applied"``.

    The signal field is ``answerability_alignment_rate`` — the LLM-judge
    alignment metric. The sibling ``answerable_rate`` field emitted by
    the quality aggregator is the STRUCTURAL answerability metric;
    reading it here would gate the flip on the wrong evidence. This test
    calls the helper with the canonical constant so a rename trips the
    gate.

    Catches a hand-edit of ``workflows.yaml`` that flips severity while
    bypassing the calibration gate — the helper records the calibration
    source, so post-hoc audit can detect the bypass.
    """
    from lib.governance.calibration_gate import resolve_severity_flip
    from lib.validators.assessment_retrieval_grounding import (
        CALIBRATION_FLIP_THRESHOLD,
        CALIBRATION_SIGNAL_NAME,
    )

    # The parametrized expectations below hard-code the signal name and
    # the 0.85 floor, so pin both constants before using them.
    assert CALIBRATION_SIGNAL_NAME == "answerability_alignment_rate", (
        f"CALIBRATION_SIGNAL_NAME is {CALIBRATION_SIGNAL_NAME!r}. "
        "This gate hard-codes the canonical signal name; a rename must "
        "update this test in lockstep."
    )
    assert CALIBRATION_FLIP_THRESHOLD == pytest.approx(0.85), (
        f"CALIBRATION_FLIP_THRESHOLD is "
        f"{CALIBRATION_FLIP_THRESHOLD!r}; expected 0.85."
    )

    course_slug = "calibration-fixture"
    report_dir = tmp_path / "courses" / course_slug / "quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "trainforge_assessment_quality_report.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "course_code": course_slug,
            "summary": {
                CALIBRATION_SIGNAL_NAME: alignment_rate,
            },
        }),
        encoding="utf-8",
    )

    apply_flip, payload = resolve_severity_flip(
        calibration_signal=CALIBRATION_SIGNAL_NAME,
        threshold=CALIBRATION_FLIP_THRESHOLD,
        course_slug=course_slug,
        libv2_root=tmp_path,
    )

    assert apply_flip is expected_apply, (
        f"alignment_rate={alignment_rate}: expected apply_flip="
        f"{expected_apply}, got {apply_flip}. payload={payload!r}"
    )
    assert payload["ml_features"]["calibration_signal"] == (
        CALIBRATION_SIGNAL_NAME
    )
    assert payload["ml_features"]["threshold"] == pytest.approx(0.85)
    assert payload["ml_features"]["observed_rate"] == pytest.approx(
        alignment_rate
    )

    if expected_apply:
        assert payload["decision_type"] == "severity_flip_applied", (
            f"flip-applied case must emit severity_flip_applied event; "
            f"got decision_type={payload.get('decision_type')!r}"
        )
        assert payload["decision"] == "critical"
    else:
        assert payload["decision_type"] == "severity_flip_deferred", (
            f"flip-deferred case must emit severity_flip_deferred event; "
            f"got decision_type={payload.get('decision_type')!r}"
        )
        assert payload["decision"] == "warning"
        assert payload["ml_features"]["reason"] == expected_reason, (
            f"reason mismatch: expected {expected_reason!r}, got "
            f"{payload['ml_features'].get('reason')!r}"
        )


# --------------------------------------------------------------------------- #
# Test 7 — coverage-rate gaming detection
# --------------------------------------------------------------------------- #


def test_7_coverage_drop_reasons_filled_when_unattributed_drops_present(
) -> None:
    """Unattributed coverage drops surface in the histogram.

    When more artifacts were dropped than ``drop_reasons`` accounts for,
    :func:`lib.governance.source_coverage.build_source_coverage` inserts
    an :data:`INTERNAL_DROP_REASON_MISSING` bucket so the invariant
    ``dropped_count == sum(drop_reasons.values())`` holds in the emitted
    shape. ``dropped_count`` itself resolves to ``consumed - emitted``
    when omitted.

    Why this matters: ``coverage_pct`` alone cannot distinguish "a stage
    silently dropped half the input" from "the textbook is half
    image-only" — both read 0.5. ``drop_reasons`` is the load-bearing
    disambiguation, so an unexplained gap must never be invisible.
    """
    from lib.governance.source_coverage import (
        INTERNAL_DROP_REASON_MISSING,
        build_source_coverage,
    )

    coverage = build_source_coverage(
        consumed_count=20,
        emitted_count=10,
        drop_reasons={"boilerplate": 5},
    )

    assert coverage["dropped_count"] == 10, (
        f"dropped_count must equal consumed - emitted = 10; got "
        f"{coverage['dropped_count']}"
    )
    drop_reasons = coverage["drop_reasons"]
    assert INTERNAL_DROP_REASON_MISSING in drop_reasons, (
        f"unattributed drops must surface under "
        f"{INTERNAL_DROP_REASON_MISSING!r}; got drop_reasons="
        f"{drop_reasons!r}"
    )
    assert drop_reasons[INTERNAL_DROP_REASON_MISSING] == 5, (
        f"missing-bucket count must equal dropped - sum(attributed) = "
        f"10 - 5 = 5; got {drop_reasons[INTERNAL_DROP_REASON_MISSING]}"
    )
    assert drop_reasons["boilerplate"] == 5, (
        f"caller-provided buckets must persist intact; got "
        f"drop_reasons={drop_reasons!r}"
    )
    histogram_total = sum(drop_reasons.values())
    assert histogram_total == coverage["dropped_count"], (
        f"invariant violation: dropped_count={coverage['dropped_count']} "
        f"!= sum(drop_reasons.values())={histogram_total} after "
        "INTERNAL_DROP_REASON_MISSING insertion"
    )

    # coverage_pct is emitted/consumed.
    assert coverage["coverage_pct"] == pytest.approx(0.5), (
        f"coverage_pct must equal emitted/consumed = 0.5; got "
        f"{coverage['coverage_pct']}"
    )

    # Defence-in-depth: a clean run with no unattributed drops MUST
    # NOT introduce the bucket (otherwise we'd be polluting clean-
    # corpus emits with a false-positive sentinel).
    clean_coverage = build_source_coverage(
        consumed_count=20,
        emitted_count=10,
        drop_reasons={"boilerplate": 7, "duplicate": 3},
    )
    assert INTERNAL_DROP_REASON_MISSING not in clean_coverage["drop_reasons"], (
        "clean histogram (drop_reasons total == dropped_count) MUST "
        "NOT introduce internal_drop_reason_missing; got drop_reasons="
        f"{clean_coverage['drop_reasons']!r}"
    )


# --------------------------------------------------------------------------- #
# Test 8 — collected-count guard + DeprecationWarning-clean
# --------------------------------------------------------------------------- #


def test_8_collected_count_guard_and_check_schema_clean() -> None:
    """Test-count ratchet + schema cleanliness sweep.

    AST-walks seven trees and asserts each tree's ``def test_*`` count is
    at or above its ``_MIN_*_TESTS`` floor. The ratchet only moves up —
    adding tests means bumping the constant; a refactor that removes
    tests trips the gate.

    Also asserts ``Draft202012Validator.check_schema(...)`` runs clean
    (no ``DeprecationWarning``) on the net-new schemas in
    ``_W3_SCHEMA_PATHS``. A deprecated draft-2020-12 construct still
    LOADS, so without this sweep the only symptom would be a warning
    emitted by every consumer at validation time.
    """
    jsonschema = pytest.importorskip("jsonschema")

    # ---- AST count ratchet ------------------------------------------------
    trees = [
        ("validators",
         PROJECT_ROOT / "lib" / "validators" / "tests",
         _MIN_VALIDATORS_TESTS),
        ("aggregators",
         PROJECT_ROOT / "lib" / "aggregators" / "tests",
         _MIN_AGGREGATORS_TESTS),
        ("trainforge_eval",
         PROJECT_ROOT / "Trainforge" / "eval" / "tests",
         _MIN_TRAINFORGE_EVAL_TESTS),
        ("trainforge",
         PROJECT_ROOT / "Trainforge" / "tests",
         _MIN_TRAINFORGE_TESTS),
        ("courseforge_router",
         PROJECT_ROOT / "Courseforge" / "router" / "tests",
         _MIN_COURSEFORGE_ROUTER_TESTS),
        ("mcp",
         PROJECT_ROOT / "MCP" / "tests",
         _MIN_MCP_TESTS),
        ("schemas",
         PROJECT_ROOT / "schemas" / "tests",
         _MIN_SCHEMAS_TESTS),
    ]
    failures: List[str] = []
    for label, tree_path, floor in trees:
        actual, files = _count_test_functions(tree_path)
        if actual < floor:
            failures.append(
                f"{label} tree at {tree_path}: collected {actual} test "
                f"functions across {len(files)} files; floor is {floor}. "
                f"This gate ratchets upward — a future wave that "
                f"legitimately adds tests must bump _MIN_{label.upper()}_TESTS."
            )
    assert not failures, (
        "Wave-end test-count guard tripped (silent test removal "
        "regression suspected):\n  " + "\n  ".join(failures)
    )

    # ---- Schema cleanliness sweep -----------------------------------------
    schema_failures: List[str] = []
    for schema_path in _W3_SCHEMA_PATHS:
        if not schema_path.exists():
            schema_failures.append(
                f"{schema_path.relative_to(PROJECT_ROOT)}: file missing "
                "(every schema in _W3_SCHEMA_PATHS must exist for this "
                "sweep to be meaningful)"
            )
            continue
        try:
            schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            schema_failures.append(
                f"{schema_path.relative_to(PROJECT_ROOT)}: load failed: "
                f"{exc!r}"
            )
            continue
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                jsonschema.Draft202012Validator.check_schema(schema_doc)
            except jsonschema.SchemaError as exc:
                schema_failures.append(
                    f"{schema_path.relative_to(PROJECT_ROOT)}: "
                    f"check_schema raised SchemaError: {exc!r}"
                )
                continue
            deprecation_warnings = [
                w for w in captured
                if issubclass(w.category, DeprecationWarning)
            ]
        if deprecation_warnings:
            schema_failures.append(
                f"{schema_path.relative_to(PROJECT_ROOT)}: "
                f"check_schema emitted DeprecationWarning(s): "
                + "; ".join(str(w.message) for w in deprecation_warnings)
            )
    assert not schema_failures, (
        "Schema cleanliness sweep failed:\n  "
        + "\n  ".join(schema_failures)
    )
