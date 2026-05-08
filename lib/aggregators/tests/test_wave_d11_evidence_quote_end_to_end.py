"""Wave W-D11 T11.6 — end-to-end integration test for the evidence_quote chain.

This is the wave-end integration test that exercises the FULL T11.0 → T11.5
path against a real fixture pair carrying populated ``evidence_quote`` +
``evidence_char_span`` fields.

The test chain (mirroring the wave's per-task contracts):

  T11.0 — schema admits the new optional fields (chunk_v4 ``key_claims``
          structured arm + ``pair_audit_fields.PerClaimSupport``).
  T11.1 — block-side ``ClaimSupportValidator`` enforces substring +
          char_span containment (not directly under test here; covered by
          ``lib/validators/tests/test_claim_support_evidence_quote.py``).
  T11.2 — pair-side ``PairClaimSupportValidator`` enforces the same
          substring + char_span containment on the per_claim_support[]
          stamp and surfaces aggregate-rate metadata on
          ``GateResult.metadata``. Driven HERE through the gate-runner
          ``validate()`` surface against a fixture-pair JSONL.
  T11.3 — synthesis-side providers emit the ``evidence_quote`` field on
          per_claim_support[] (covered by Trainforge/tests/
          test_provider_evidence_quote_emit.py — not under test here).
  T11.4 — decision-capture rationale interpolates the three rates with
          a stable ``evidence_quote: coverage=...`` format string. This
          test asserts the format on the captured event from the
          ``validate()`` walk.
  T11.5 — promotion_chain_report aggregator + trainforge_assessment_quality
          aggregator surface the rates (per-arrow ``evidence_quote_metrics``
          + top-level ``evidence_quote_summary`` for the chain report;
          ``synthesis_quality.evidence_quote`` for the assessment-quality
          report). Driven HERE by feeding the GateResult from T11.2 into
          both aggregators and asserting the wire shape.
  T11.6 — fixtures + integration test (this file).

The fixtures under ``schemas/tests/fixtures/wave_d11/`` are the
shape-discriminating snapshots — they are loaded at runtime so the
integration test fails loudly if either the chunk-side ``key_claims`` or
the pair-side ``per_claim_support`` snapshot drifts away from the
T11.0 schema contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "schemas" / "tests" / "fixtures" / "wave_d11"
CHUNK_FIXTURE = FIXTURES_DIR / "chunk_with_evidence_quote_key_claims.json"
PAIR_FIXTURE = FIXTURES_DIR / (
    "instruction_pair_with_evidence_quote_per_claim_support.json"
)
PROMOTION_CHAIN_SCHEMA = (
    REPO_ROOT / "schemas" / "governance" / "promotion_chain.schema.json"
)
COURSE_STATUS_SCHEMA = (
    REPO_ROOT / "schemas" / "governance" / "course_status.schema.json"
)


# --------------------------------------------------------------------------
# Decision-capture spy — minimal log_decision recorder.
# --------------------------------------------------------------------------


class _SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# --------------------------------------------------------------------------
# Fixture loaders
# --------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture
def loaded_chunk_fixture() -> Dict[str, Any]:
    """Load the chunk fixture carrying structured ``key_claims`` with
    ``evidence_quote`` + ``evidence_char_span``."""
    fixture = _load_json(CHUNK_FIXTURE)
    # Sanity-check: fixture roundtrip — chunk.text[start:end] == quote.
    kc = fixture["key_claims"][0]
    start, end = kc["evidence_char_span"]
    assert fixture["text"][start:end] == kc["evidence_quote"], (
        "chunk fixture char_span must slice to evidence_quote — round-trip "
        "guard against fixture drift."
    )
    return fixture


@pytest.fixture
def loaded_pair_fixture() -> Dict[str, Any]:
    """Load the pair fixture carrying per_claim_support[] with
    ``evidence_quote`` + ``evidence_char_span``."""
    fixture = _load_json(PAIR_FIXTURE)
    pcs = fixture["per_claim_support"][0]
    assert "evidence_quote" in pcs
    assert "evidence_char_span" in pcs
    return fixture


# --------------------------------------------------------------------------
# (a) Validator-side: pair_claim_support fixture WITH evidence_quote
# --------------------------------------------------------------------------


def test_validator_with_evidence_quote_surfaces_metadata_rates(
    tmp_path: Path,
    loaded_chunk_fixture: Dict[str, Any],
    loaded_pair_fixture: Dict[str, Any],
) -> None:
    """T11.2 → T11.5 hop: a fixture pair with populated evidence_quote
    drives the pair-side ``PairClaimSupportValidator.validate()``; the
    resulting GateResult.metadata carries the three evidence_quote_*
    rates with non-zero coverage."""
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    inst_path = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst_path, [loaded_pair_fixture])

    chunk_id = loaded_chunk_fixture["id"]
    chunk_text = loaded_chunk_fixture["text"]

    capture = _SpyCapture()
    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst_path),
        "chunk_id_to_text_map": {chunk_id: chunk_text},
        "decision_capture": capture,
    })

    # Audit passes (only structural-key absence flips passed False).
    assert result.passed is True

    # T11.5 hand-off contract — three canonical rates on metadata.
    md = result.metadata
    assert md is not None
    for key in (
        "evidence_quote_coverage_rate",
        "evidence_quote_substring_fail_rate",
        "evidence_quote_char_span_mismatch_rate",
    ):
        assert key in md, f"GateResult.metadata missing {key!r}"
    assert md["evidence_quote_coverage_rate"] > 0, (
        "fixture carries a valid quote — coverage_rate must be > 0"
    )
    # The fixture's quote is a clean substring with a matching char_span,
    # so substring_fail and char_span_mismatch rates are 0.
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0


# --------------------------------------------------------------------------
# (b) Validator-side: legacy pair WITHOUT evidence_quote (graceful-degrade)
# --------------------------------------------------------------------------


def test_validator_without_evidence_quote_returns_zero_rates_passed_true(
    tmp_path: Path,
) -> None:
    """T11.2 graceful-degrade arm: a pair without evidence_quote on its
    per_claim_support[] entry hits the ``claims_without_quote`` counter,
    no GateIssue fires, and substring + char_span rates default to 0.0
    while coverage_rate also lands at 0.0 (no valid-quote claims).
    ``passed`` stays True (audit walks structural-key presence only)."""
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    legacy_pair = {
        "prompt": "Explain why working memory limits cognitive intake.",
        "completion": (
            "Working memory holds only a small number of items at once; "
            "instruction beyond that ceiling overwhelms the learner and "
            "degrades retention."
        ),
        "chunk_id": "wave_d11_legacy_chunk",
        "lo_refs": ["TO-01"],
        "bloom_level": "understand",
        "content_type": "explanation",
        "seed": 42,
        "decision_capture_id": "EVT_legacy",
        "per_claim_support": [
            {
                "sentence": (
                    "Working memory holds only a small number of items at "
                    "once."
                ),
                "entailment": 0.85,
                "contradiction": 0.04,
                "outcome": "entailed",
                # No evidence_quote / evidence_char_span — legacy shape.
            },
        ],
        "claim_support_rate": 1.0,
    }

    inst_path = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst_path, [legacy_pair])

    result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst_path),
        # Threading the chunk text would still find no quote (none set).
        "chunk_id_to_text_map": {
            "wave_d11_legacy_chunk": "any text since the entry has no quote",
        },
    })

    assert result.passed is True
    md = result.metadata
    assert md is not None
    # All three rates land at 0.0 — graceful-degrade contract.
    assert md["evidence_quote_coverage_rate"] == 0.0
    assert md["evidence_quote_substring_fail_rate"] == 0.0
    assert md["evidence_quote_char_span_mismatch_rate"] == 0.0
    # Counter sanity: total_claims == 1, claims_without_quote == 1.
    assert md["total_claims"] == 1
    assert md["claims_without_quote"] == 1
    assert md["claims_with_valid_quote"] == 0


# --------------------------------------------------------------------------
# (c) Aggregator-side: PromotionChainAggregator surfaces the rates on Arrow 7
# --------------------------------------------------------------------------


def _build_minimal_layout(tmp_path: Path) -> Dict[str, Path]:
    """Minimal LibV2 + Courseforge layout for the promotion chain build.

    The integration test cares about Arrow 7 (training pairs) wiring;
    other arrows resolve as ``missing_stage_report`` rows, which is fine
    — the aggregator's per-arrow read is best-effort and the missing
    rows don't prevent the evidence_quote_metrics block from landing on
    Arrow 7.
    """
    course_dir = tmp_path / "libv2_course"
    course_dir.mkdir(parents=True, exist_ok=True)
    # synthesis_summary.json is the Arrow 7 anchor (input_hash source).
    ts_dir = course_dir / "training_specs"
    ts_dir.mkdir(parents=True, exist_ok=True)
    (ts_dir / "synthesis_summary.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "course_code": "WAVE_D11_E2E",
            "provider": "local",
            "instruction_pairs_emitted": 1,
            "preference_pairs_emitted": 0,
            "source_coverage": {
                "consumed_count": 1,
                "emitted_count": 1,
                "dropped_count": 0,
                "drop_reasons": {},
                "coverage_pct": 1.0,
            },
        }),
        encoding="utf-8",
    )
    return {"course_dir": course_dir}


def test_promotion_chain_aggregator_carries_evidence_quote_metrics_arrow_7(
    tmp_path: Path,
    loaded_chunk_fixture: Dict[str, Any],
    loaded_pair_fixture: Dict[str, Any],
) -> None:
    """T11.5 hop: feeding a GateResult-shaped row carrying the three
    evidence_quote_* rates into ``PromotionChainAggregator`` surfaces the
    ``evidence_quote_metrics`` block on Arrow 7 (training_synthesis →
    pair_claim_support seam) and rolls it up into the top-level
    ``evidence_quote_summary.coverage_rate`` field."""
    from lib.aggregators.promotion_chain_report import PromotionChainAggregator
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    layout = _build_minimal_layout(tmp_path)

    # First run the actual validator against the fixture pair so the
    # GateResult.metadata payload comes from the real T11.2 code path.
    inst_path = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst_path, [loaded_pair_fixture])
    chunk_id = loaded_chunk_fixture["id"]
    chunk_text = loaded_chunk_fixture["text"]
    gate_result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst_path),
        "chunk_id_to_text_map": {chunk_id: chunk_text},
    })

    # Project GateResult to the dict shape phase_outputs expects (the
    # aggregator reads ``phase_outputs[phase]._gate_results`` as a list
    # of dict-shaped GateResult payloads).
    gate_dict = {
        "gate_id": gate_result.gate_id,
        "validator_name": gate_result.validator_name,
        "validator_version": gate_result.validator_version,
        "passed": gate_result.passed,
        "severity": "warning",
        "score": gate_result.score,
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "message": i.message,
                "location": i.location,
                "suggestion": i.suggestion,
            }
            for i in gate_result.issues
        ],
        "execution_time_ms": 0,
        "action": gate_result.action,
        "metadata": gate_result.metadata,
    }

    phase_outputs = {
        "training_synthesis": {
            "_completed": True,
            "_gates_passed": True,
            "_gate_results": [gate_dict],
        },
    }

    agg = PromotionChainAggregator(
        course_path=layout["course_dir"],
        course_code="WAVE_D11_E2E",
        run_id="WF-WAVE-D11-T11.6",
        phase_outputs=phase_outputs,
    )
    report = agg.build()

    # Arrow 7 carries the evidence_quote_metrics block.
    arrow_7 = next(a for a in report["arrows"] if a["arrow_id"] == 7)
    assert "evidence_quote_metrics" in arrow_7, (
        "Arrow 7 missing evidence_quote_metrics — T11.5 wiring regressed"
    )
    metrics = arrow_7["evidence_quote_metrics"]
    assert metrics["evidence_quote_coverage_rate"] == 1.0, (
        f"fixture pair had a clean valid quote — coverage_rate must be "
        f"1.0; got {metrics['evidence_quote_coverage_rate']}"
    )
    assert metrics["evidence_quote_substring_fail_rate"] == 0.0
    assert metrics["evidence_quote_char_span_mismatch_rate"] == 0.0
    assert "pair_claim_support" in arrow_7["validator_set"]

    # Top-level summary rolls up the unweighted mean across surfacing
    # arrows; with only Arrow 7 contributing, it equals Arrow 7's rate.
    assert report["evidence_quote_summary"] == {"coverage_rate": 1.0}


# --------------------------------------------------------------------------
# (d) Aggregator-side: TrainforgeAssessmentQualityReport propagates rates
# --------------------------------------------------------------------------


def test_trainforge_assessment_quality_report_evidence_quote_block(
    tmp_path: Path,
    loaded_chunk_fixture: Dict[str, Any],
    loaded_pair_fixture: Dict[str, Any],
) -> None:
    """T11.5 chunks-and-pairs surface: feeding the same GateResult into
    ``TrainforgeAssessmentQualityReport`` populates the
    ``synthesis_quality.evidence_quote`` block with the three canonical
    rate fields."""
    from lib.aggregators.trainforge_assessment_quality_report import (
        TrainforgeAssessmentQualityReport,
    )
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    inst_path = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst_path, [loaded_pair_fixture])
    chunk_id = loaded_chunk_fixture["id"]
    chunk_text = loaded_chunk_fixture["text"]
    gate_result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst_path),
        "chunk_id_to_text_map": {chunk_id: chunk_text},
    })

    gate_dict = {
        "gate_id": gate_result.gate_id,
        "validator_name": gate_result.validator_name,
        "validator_version": gate_result.validator_version,
        "passed": gate_result.passed,
        "severity": "warning",
        "score": gate_result.score,
        "issues": [],
        "execution_time_ms": 0,
        "action": gate_result.action,
        "metadata": gate_result.metadata,
    }
    phase_outputs = {
        "training_synthesis": {
            "_completed": True,
            "_gates_passed": True,
            "_gate_results": [gate_dict],
        },
    }

    agg = TrainforgeAssessmentQualityReport(
        phase_outputs=phase_outputs,
        course_code="WAVE_D11_E2E",
        run_id="WF-WAVE-D11-T11.6",
    )
    report = agg.build()

    block = report["synthesis_quality"]["evidence_quote"]
    assert block == {
        "evidence_quote_coverage_rate": 1.0,
        "evidence_quote_substring_fail_rate": 0.0,
        "evidence_quote_char_span_mismatch_rate": 0.0,
    }


# --------------------------------------------------------------------------
# (e) Schema validation: the freshly-aggregated promotion_chain_report.json
#     output validates against schemas/governance/promotion_chain.schema.json
# --------------------------------------------------------------------------


def test_aggregated_promotion_chain_report_validates_against_schema(
    tmp_path: Path,
    loaded_chunk_fixture: Dict[str, Any],
    loaded_pair_fixture: Dict[str, Any],
) -> None:
    """Anti-regression contract: the aggregator's emit MUST validate
    against the bumped schema (W-D11 T11.5 added EvidenceQuoteMetrics +
    EvidenceQuoteSummary as additive-optional). Skips when jsonschema is
    not installed (matches the rest of the schemas/tests/ skip pattern)."""
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator, RefResolver

    from lib.aggregators.promotion_chain_report import PromotionChainAggregator
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    layout = _build_minimal_layout(tmp_path)

    inst_path = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst_path, [loaded_pair_fixture])
    chunk_id = loaded_chunk_fixture["id"]
    chunk_text = loaded_chunk_fixture["text"]
    gate_result = PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst_path),
        "chunk_id_to_text_map": {chunk_id: chunk_text},
    })

    gate_dict = {
        "gate_id": gate_result.gate_id,
        "validator_name": gate_result.validator_name,
        "validator_version": gate_result.validator_version,
        "passed": gate_result.passed,
        "severity": "warning",
        "score": gate_result.score,
        "issues": [],
        "execution_time_ms": 0,
        "action": gate_result.action,
        "metadata": gate_result.metadata,
    }
    phase_outputs = {
        "training_synthesis": {
            "_completed": True,
            "_gates_passed": True,
            "_gate_results": [gate_dict],
        },
    }
    agg = PromotionChainAggregator(
        course_path=layout["course_dir"],
        course_code="WAVE_D11_E2E",
        run_id="WF-WAVE-D11-T11.6",
        phase_outputs=phase_outputs,
    )
    report = agg.build()

    # Build a Draft 2020-12 validator with the cross-schema $ref to
    # course_status.schema.json pre-resolved (mirrors the existing
    # test_promotion_chain_report.py::_load_schema_validator helper).
    schema = json.loads(PROMOTION_CHAIN_SCHEMA.read_text(encoding="utf-8"))
    course_status_schema = json.loads(
        COURSE_STATUS_SCHEMA.read_text(encoding="utf-8")
    )
    resolver = RefResolver.from_schema(schema)
    resolver.store[course_status_schema["$id"]] = course_status_schema
    validator = Draft202012Validator(schema, resolver=resolver)

    errors = list(validator.iter_errors(report))
    assert errors == [], (
        "aggregated promotion_chain_report.json failed schema validation: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )

    # Also assert the new W-D11 T11.5 fields are present + well-shaped.
    arrow_7 = next(a for a in report["arrows"] if a["arrow_id"] == 7)
    assert "evidence_quote_metrics" in arrow_7
    assert set(arrow_7["evidence_quote_metrics"].keys()) == {
        "evidence_quote_coverage_rate",
        "evidence_quote_substring_fail_rate",
        "evidence_quote_char_span_mismatch_rate",
    }
    assert report["evidence_quote_summary"]["coverage_rate"] >= 0.0


# --------------------------------------------------------------------------
# (f) Decision-capture: rationale interpolates the canonical format string.
# --------------------------------------------------------------------------


def test_decision_capture_rationale_carries_evidence_quote_format_string(
    tmp_path: Path,
    loaded_chunk_fixture: Dict[str, Any],
    loaded_pair_fixture: Dict[str, Any],
) -> None:
    """T11.4 contract: the ``pair_claim_support_check`` decision event
    fired by ``validate()`` carries an ``evidence_quote: coverage=`` token
    in its rationale so a single grep across the three call sites
    (block-side, per-pair, and gate-walk) finds them all."""
    from lib.validators.pair.claim_support import PairClaimSupportValidator

    inst_path = tmp_path / "instruction_pairs.jsonl"
    _write_jsonl(inst_path, [loaded_pair_fixture])
    chunk_id = loaded_chunk_fixture["id"]
    chunk_text = loaded_chunk_fixture["text"]

    capture = _SpyCapture()
    PairClaimSupportValidator().validate({
        "instruction_pairs_path": str(inst_path),
        "chunk_id_to_text_map": {chunk_id: chunk_text},
        "decision_capture": capture,
    })

    # Exactly one ``pair_claim_support_check`` event with the expected
    # rationale token.
    matching = [
        e for e in capture.events
        if e.get("decision_type") == "pair_claim_support_check"
    ]
    assert len(matching) == 1, (
        f"expected exactly one pair_claim_support_check event from the "
        f"validate() walk, got {len(matching)} (events: {capture.events})"
    )
    rationale = matching[0].get("rationale", "")
    assert "evidence_quote: coverage=" in rationale, (
        f"T11.4 format-string contract violated — rationale must carry "
        f"'evidence_quote: coverage=' for cross-surface grep parity. "
        f"Got rationale: {rationale!r}"
    )


# --------------------------------------------------------------------------
# (g) Fixture-shape sanity: snapshots validate against their target schemas.
# --------------------------------------------------------------------------


def test_chunk_fixture_validates_against_chunk_v4_schema(
    loaded_chunk_fixture: Dict[str, Any],
) -> None:
    """The wave_d11 chunk snapshot must validate against
    ``chunk_v4.schema.json`` so a future schema bump doesn't silently
    invalidate the integration fixture (and by extension the rest of
    this file)."""
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schemas_dir = REPO_ROOT / "schemas"
    id_to_schema: Dict[str, Any] = {}
    for path in schemas_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = payload.get("$id")
        if sid:
            id_to_schema[sid] = payload
    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)

    chunk_schema = json.loads(
        (schemas_dir / "knowledge" / "chunk_v4.schema.json")
        .read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(chunk_schema, registry=registry)
    errors = list(validator.iter_errors(loaded_chunk_fixture))
    assert errors == [], (
        "wave_d11 chunk fixture failed chunk_v4 validation: "
        f"{[(list(e.absolute_path), e.message) for e in errors]}"
    )


def test_pair_fixture_per_claim_support_validates_against_schema(
    loaded_pair_fixture: Dict[str, Any],
) -> None:
    """The wave_d11 pair snapshot's per_claim_support[] must validate
    against the canonical PerClaimSupport $defs in
    ``pair_audit_fields.schema.json``."""
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    schemas_dir = REPO_ROOT / "schemas"
    id_to_schema: Dict[str, Any] = {}
    for path in schemas_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = payload.get("$id")
        if sid:
            id_to_schema[sid] = payload
    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": (
            "https://ed4all.dev/ns/knowledge/v1/pair_audit_fields.schema.json"
            "#/$defs/PerClaimSupport"
        ),
    }
    validator = Draft202012Validator(wrapper, registry=registry)
    for entry in loaded_pair_fixture["per_claim_support"]:
        errors = list(validator.iter_errors(entry))
        assert errors == [], (
            "wave_d11 pair fixture per_claim_support entry failed "
            f"PerClaimSupport validation: "
            f"{[(list(e.absolute_path), e.message) for e in errors]}"
        )
